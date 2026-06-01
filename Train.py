from ultralytics import YOLO

def main():
    model = YOLO("./last1.pt")
    results = model.train(
        data="./SSD/data.yaml",
        epochs=150,
        imgsz=640,
        batch=128,
        
        plots=True,
        save=True,
        val=True,
        
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        
        project="yolo", 
        amp=True,     
        workers=4,

        # ========== 增加泛化能力的参数 ==========
        weight_decay=0.0005,       # 权重衰减（L2正则化），常用0.0005
        label_smoothing=0.1,       # 标签平滑，防止对硬标签过自信
        dropout=0.1,               # 随机失活（如果模型支持，如YOLOv8可设）
        
        # 数据增强强度（值越大越多样，但要避免失真）
        hsv_h=0.015,               # 色调扰动范围
        hsv_s=0.7,                 # 饱和度扰动
        hsv_v=0.4,                 # 明度扰动
        degrees=5.0,               # 随机旋转角度（度）
        translate=0.1,             # 平移比例
        scale=0.5,                 # 缩放比例（0.5表示随机缩放0.5~1.5倍）
        shear=2.0,                 # 剪切变换角度
        perspective=0.0005,        # 透视变换程度（极小值，避免过度变形）
        flipud=0.0,                # 上下翻转概率（目标检测通常不用）
        fliplr=0.5,                # 左右翻转概率
        mosaic=1.0,                # mosaic概率（1.0表示每个epoch都用）
        mixup=0.2,                 # mixup混合概率
        copy_paste=0.1,            # copy-paste概率（适合实例分割/检测）
        erasing=0.4,               # 随机擦除概率（Cutout）
        
        # 学习率与训练策略（稳定收敛，间接提升泛化）
        warmup_epochs=3,           # 预热轮数
        warmup_momentum=0.8,       # 预热初始动量
        warmup_bias_lr=0.1,        # 预热初始偏置学习率
        cos_lr=True,               # 使用余弦退火学习率衰减
    )
if __name__ == '__main__':
    main()