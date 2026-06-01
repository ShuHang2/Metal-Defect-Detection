from ultralytics import YOLO
import pandas as pd
import numpy as np
from pathlib import Path

def evaluate_trained_model(model_path=None, data_yaml=None):

    # model_path = "./best.pt"
    data_yaml = "./GC10.yaml"
    model_path = "./last.pt"
    
    # 2. 加载模型
    print("加载模型:",model_path)
    model = YOLO(str(model_path))
    
    # 3. 在验证集上评估
    print("在验证集上评估模型...")
    metrics = model.val(
        data=data_yaml,
        split='test',           # 使用验证集
        imgsz=640,             # 与训练时保持一致
        batch=8,
        conf=0.25,             # 置信度阈值
        iou=0.5,              # NMS IoU阈值
        
        plots=True,            # 生成评估图表
        save_json=True,        # 保存JSON结果
        save_hybrid=True,      # 保存混合标签
        project='runs',    # 保存目录
        name='Severstal-steel-defect',
        exist_ok=True
    )
    
    # 4. 打印详细评估结果
    print("\n" + "="*60)
    print("模型性能评估结果")
    print("="*60)
    
    if hasattr(metrics, 'box'):
        print(f"平均精度 mAP50-95: {metrics.box.map:.4f}")
        print(f"平均精度 mAP50: {metrics.box.map50:.4f}")
        print(f"平均精度 mAP75: {metrics.box.map75:.4f}")
        
        # 修正这里：metrics.box.p 和 metrics.box.r 是数组
        if hasattr(metrics.box, 'p'):
            precision_value = metrics.box.p if isinstance(metrics.box.p, (int, float)) else metrics.box.p.mean()
            print(f"精确率 Precision: {precision_value:.4f}")
        
        if hasattr(metrics.box, 'r'):
            recall_value = metrics.box.r if isinstance(metrics.box.r, (int, float)) else metrics.box.r.mean()
            print(f"召回率 Recall: {recall_value:.4f}")
    
    # 5. 逐类别性能
    if hasattr(metrics, 'results_dict'):
        print("\n逐类别性能:")
        results_dict = metrics.results_dict
        if 'metrics' in results_dict and 'ap_class_index' in results_dict['metrics']:
            ap_per_class = results_dict['metrics']['ap50']
            class_names = results_dict['names']
            for i, (cls_idx, ap) in enumerate(zip(results_dict['metrics']['ap_class_index'], ap_per_class)):
                cls_name = class_names[cls_idx]
                print(f"  类别 {cls_idx} ({cls_name}): AP50 = {ap:.4f}")
    
    # 6. 推理速度
    if hasattr(metrics, 'speed'):
        print("\n 推理速度分析:")
        print(f"  预处理时间: {metrics.speed['preprocess']:.2f} ms/张")
        print(f"  推理时间: {metrics.speed['inference']:.2f} ms/张")
        print(f"  后处理时间: {metrics.speed['postprocess']:.2f} ms/张")
        total_time = sum(metrics.speed.values())
        print(f"  总时间: {total_time:.2f} ms/张")
        print(f"  帧率 FPS: {1000/total_time:.2f}")

if __name__ == "__main__":
    metrics = evaluate_trained_model()