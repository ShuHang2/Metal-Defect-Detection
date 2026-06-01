import cv2
import numpy as np
import yaml
import time
from pathlib import Path
from tqdm import tqdm
from queue import Queue
from threading import Thread
from ais_bench.infer.interface import InferSession

# ==================== 配置区 ====================
OM_MODEL_PATH = "./512fp16.om"
IMAGE_FOLDER = "./GC10/test/images"      # 测试集图片路径
LABEL_FOLDER = "./GC10/test/labels"      # 测试集标签路径
YAML_PATH = "GC10.yaml"
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.5
INPUT_SIZE = 512               # 你的 OM 模型输入尺寸
DEVICE_ID = 0
PAD_COLOR = (114, 114, 114)
# ===============================================

def preprocess(img, input_size=640):
    h, w = img.shape[:2]
    r = input_size / max(h, w)
    new_w, new_h = int(w * r), int(h * r)
    resized = cv2.resize(img, (new_w, new_h))

    pad_w = (input_size - new_w) // 2
    pad_h = (input_size - new_h) // 2

    canvas = np.full((input_size, input_size, 3), PAD_COLOR, dtype=np.uint8)
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    blob = cv2.dnn.blobFromImage(canvas, scalefactor=1/255.0, swapRB=True)
    return blob, r, (pad_w, pad_h), (h, w)

def postprocess(outputs, r, pad, conf_thres, iou_thres, hw):
    h_img, w_img = hw
    pad_w, pad_h = pad

    output = np.squeeze(outputs[0])
    if output.shape[0] < output.shape[1]:
        output = output.T

    scores = np.max(output[:, 4:], axis=1)
    mask = scores > conf_thres
    output, scores = output[mask], scores[mask]

    if len(scores) == 0:
        return [], [], []

    class_ids = np.argmax(output[:, 4:], axis=1)
    boxes = output[:, :4]

    x1 = (boxes[:, 0] - boxes[:, 2] / 2 - pad_w) / r
    y1 = (boxes[:, 1] - boxes[:, 3] / 2 - pad_h) / r
    x2 = (boxes[:, 0] + boxes[:, 2] / 2 - pad_w) / r
    y2 = (boxes[:, 1] + boxes[:, 3] / 2 - pad_h) / r

    x1 = np.clip(x1, 0, w_img)
    y1 = np.clip(y1, 0, h_img)
    x2 = np.clip(x2, 0, w_img)
    y2 = np.clip(y2, 0, h_img)

    valid = (x2 > x1) & (y2 > y1)
    boxes = np.stack([x1, y1, x2, y2], axis=1)[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]

    if len(boxes) == 0:
        return [], [], []

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)
    if len(indices) == 0:
        return [], [], []

    idx = indices.flatten()
    return boxes[idx].tolist(), scores[idx].tolist(), class_ids[idx].tolist()

def compute_iou(box1, box2):
    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def evaluate_metrics(all_preds, all_gts, num_classes):
    aps, valid_ids = [], []
    total_tp, total_fp, total_fn = 0, 0, 0

    for cls_id in range(num_classes):
        dects, n_gt = [], 0
        for preds, gts in zip(all_preds, all_gts):
            cls_preds = [p for p in preds if p['label'] == cls_id]
            cls_gts = [g for g in gts if g['label'] == cls_id]
            n_gt += len(cls_gts)

            cls_preds.sort(key=lambda x: x['score'], reverse=True)
            matched_gt = [False] * len(cls_gts)
            for p in cls_preds:
                best_iou, match_idx = 0, -1
                for i, g in enumerate(cls_gts):
                    if matched_gt[i]:
                        continue
                    iou = compute_iou(p['box'], g['box'])
                    if iou > best_iou:
                        best_iou, match_idx = iou, i
                if best_iou >= 0.5:
                    matched_gt[match_idx] = True
                    dects.append([p['score'], 1])
                else:
                    dects.append([p['score'], 0])

        if n_gt == 0:
            for preds_img in all_preds:
                for p in preds_img:
                    if p['label'] == cls_id:
                        total_fp += 1
            continue

        tp_count = sum(1 for _, flag in dects if flag == 1)
        fp_count = len(dects) - tp_count
        fn_count = n_gt - tp_count

        total_tp += tp_count
        total_fp += fp_count
        total_fn += fn_count

        dects.sort(key=lambda x: x[0], reverse=True)
        tp_cum = np.cumsum([x[1] for x in dects])
        fp_cum = np.cumsum([1 - x[1] for x in dects])
        recall = tp_cum / n_gt
        precision = tp_cum / (tp_cum + fp_cum + 1e-6)

        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ap = np.sum((mrec[1:] - mrec[:-1]) * mpre[1:])
        aps.append(ap)
        valid_ids.append(cls_id)

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return aps, valid_ids, overall_precision, overall_recall

def preprocess_worker(img_paths, input_queue):
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        blob, r, pad, (h, w) = preprocess(img, INPUT_SIZE)
        input_queue.put((p, blob, r, pad, h, w))
    input_queue.put(None)

def main():
    with open(YAML_PATH, 'r') as f:
        class_names = yaml.safe_load(f)['names']

    session = InferSession(device_id=DEVICE_ID, model_path=OM_MODEL_PATH)
    img_paths = list(Path(IMAGE_FOLDER).glob("*.jpg")) + list(Path(IMAGE_FOLDER).glob("*.png"))

    input_queue = Queue(maxsize=20)
    t = Thread(target=preprocess_worker, args=(img_paths, input_queue))
    t.start()

    all_preds, all_gts = [], []
    pbar = tqdm(total=len(img_paths), desc="OM 推理")

    start_time = time.time()

    while True:
        data = input_queue.get()
        if data is None:
            break

        img_path, blob, r, pad, h, w = data

        # 🔧 修正：直接传入 numpy 数组，不要用字典
        outputs = session.infer(feeds=blob, mode="static")

        boxes, scores, labels = postprocess(
            outputs, r, pad, CONF_THRESHOLD, NMS_THRESHOLD, (h, w)
        )
        all_preds.append([
            {'box': b, 'score': s, 'label': l}
            for b, s, l in zip(boxes, scores, labels)
        ])

        label_file = Path(LABEL_FOLDER) / (img_path.stem + ".txt")
        gts = []
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    c, xc, yc, nw, nh = map(float, parts)
                    gts.append({
                        'box': [
                            (xc - nw / 2) * w,
                            (yc - nh / 2) * h,
                            (xc + nw / 2) * w,
                            (yc + nh / 2) * h
                        ],
                        'label': int(c)
                    })
        all_gts.append(gts)
        pbar.update(1)

    total_time = time.time() - start_time
    pbar.close()
    t.join()

    print("\n正在计算性能指标...")
    aps, valid_ids, precision, recall = evaluate_metrics(
        all_preds, all_gts, len(class_names)
    )

    print("-" * 50)
    print(f"模型: {OM_MODEL_PATH}")
    print(f"输入尺寸: {INPUT_SIZE}")
    print(f"置信度阈值: {CONF_THRESHOLD}, NMS IoU阈值: {NMS_THRESHOLD}")
    print(f"图片数量: {len(img_paths)}")
    print("-" * 50)
    print("逐类别 AP50:")
    for i, ap in zip(valid_ids, aps):
        print(f"  类别 {i} [{class_names[i]}]: AP50 = {ap:.4f}")
    print("-" * 50)
    print(f"整体 mAP50: {np.mean(aps):.4f}")
    print(f"整体 Precision (TP/(TP+FP)): {precision:.4f}")
    print(f"整体 Recall (TP/(TP+FN)): {recall:.4f}")
    if total_time > 0:
        fps = len(img_paths) / total_time
        avg_ms = total_time / len(img_paths) * 1000
        print(f"总推理时间: {total_time:.2f} 秒")
        print(f"平均每张时间: {avg_ms:.2f} 毫秒")
        print(f"FPS: {fps:.2f}")
    print("-" * 50)

if __name__ == "__main__":
    main()