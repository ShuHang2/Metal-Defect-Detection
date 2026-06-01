from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics import YOLO
import torch
import torch.nn.functional as F

# 继承官方的训练器，创建一个我们自己的“蒸馏训练器”
class KDTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        
        # ==========================================
        # 1. 加载“老师”模型 (YOLOv8m)
        # ==========================================
        print("🚀 正在加载 Teacher 模型 (YOLOv8m)...")
        # ⚠️ 注意：这里换成你刚刚训练好的 YOLOv8m 权重路径！
        teacher_path = 'best_yolov8m_teacher.pt' 
        self.teacher = YOLO(teacher_path).model.to(self.device)
        
        # 将老师设置为评估模式，并且冻结所有参数（老师不参与学习，只负责教）
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # 蒸馏强度系数 (设为 0.5 意味着老师的意见和真实标签一样重要)
        self.kd_weight = 0.5 

    def loss(self, batch):
        """重写单步训练的 Loss 计算逻辑"""
        # ==========================================
        # 2. 学生 (YOLOv8n) 尝试做题
        # ==========================================
        preds_student = self.model(batch['img'])
        # 计算学生和标准答案（真实标签）的常规差距
        loss_student, loss_items = self.criterion(preds_student, batch)

        with torch.no_grad():
            preds_teacher = self.teacher(batch['img'])

        kd_loss = 0.0
        # YOLOv8 的 preds 包含多尺度的特征输出，我们直接让学生去拟合老师的特征图
        if isinstance(preds_student, tuple) and isinstance(preds_teacher, tuple):
            feats_s = preds_student[1] # 学生的特征图集合
            feats_t = preds_teacher[1] # 老师的特征图集合
            for fs, ft in zip(feats_s, feats_t):
                # 使用 MSE (均方误差) 让特征图尽可能一致
                kd_loss += F.mse_loss(fs, ft)

        total_loss = loss_student + self.kd_weight * kd_loss

        return total_loss, loss_items

if __name__ == '__main__':
    # 这里的参数就和你平时在终端输入的一样
    args = dict(
        model='yolov8n.yaml',        # 学生要用的网络结构：轻量级的 YOLOv8n
        data='Severstal.yaml',       # ⚠️ 替换成你自己的数据集配置文件
        epochs=100,                  # 训练轮数
        imgsz=512,                   # 为了香橙派提速，建议用 512 或 416
        batch=16,                    
        device='0',                  # 使用第一块 GPU
        project='Yolov8_Distillation', # 保存结果的文件夹名称
        name='v8n_student_kd'        # 本次实验的名称
    )
    
    print("🌟 开始知识蒸馏训练...")
    # 使用我们自定义的蒸馏训练器启动训练！
    trainer = KDTrainer(overrides=args)
    trainer.train()