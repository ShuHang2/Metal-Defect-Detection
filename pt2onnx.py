from ultralytics import YOLO

model = YOLO('./NEU.pt')   # 训练好的权重

success = model.export(
    format='onnx',
    imgsz=512,          # 必须等于训练时的尺寸
    dynamic=False,      # 固定尺寸，避免动态 shape 造成的精度波动
    simplify=True,      # 保持简化
    # half=False,         # 禁用半精度，使用 FP32
)
print("ONNX 模型保存至:", success)