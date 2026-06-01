import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

# --- 配置路径 ---
# 确保这里指向你的 Severstal 数据集文件夹
ROOT_DIR = Path(r'./Severstal-steel-defect')
TEMP_DIR = ROOT_DIR / 'temp_all_data' # 临时中转目录，保证数据安全
SPLITS = ['train', 'val', 'test']

# --- 设置新的划分比例 ---
RATIOS = {'train': 0.8, 'val': 0.1, 'test': 0.1}

def resplit_severstal_811():
    if not ROOT_DIR.exists():
        print(f"❌ 错误：找不到目录 {ROOT_DIR}")
        return

    TEMP_DIR.mkdir(exist_ok=True)
    class_to_data = defaultdict(list)
    
    print("📥 正在从当前 Severstal 目录收集所有原始数据到临时文件夹...")
    valid_count = 0
    
    # 1. 安全收集所有数据到临时目录
    for s in SPLITS:
        img_dir = ROOT_DIR / s / 'images'
        lbl_dir = ROOT_DIR / s / 'labels'
        
        if not lbl_dir.exists():
            continue
            
        for lbl_file in lbl_dir.glob('*.txt'):
            # 过滤掉可能的缓存或增强文件
            if '_aug_' in lbl_file.name or lbl_file.name == 'classes.txt':
                continue
                
            # 寻找对应的图片 (兼容 .jpg 和 .png)
            img_file = img_dir / lbl_file.name.replace('.txt', '.jpg')
            if not img_file.exists():
                img_file = img_dir / lbl_file.name.replace('.txt', '.png')
                
            if img_file.exists():
                # 先复制到临时目录，防止后续删除时丢失
                shutil.copy(img_file, TEMP_DIR / img_file.name)
                shutil.copy(lbl_file, TEMP_DIR / lbl_file.name)
                
                # 读取类别信息，用于分层抽样
                with open(lbl_file, 'r') as f:
                    first_line = f.readline()
                    if first_line:
                        try:
                            # 提取类别 ID (例如 "0", "1", "2", "3")
                            cls_id = int(float(first_line.split()[0]))
                            class_to_data[cls_id].append(img_file.name)
                            valid_count += 1
                        except ValueError:
                            pass

    print(f"✅ 成功收集 {valid_count} 对有效的图片和标签。")
    if valid_count == 0:
        print("❌ 没有找到有效数据，请检查 Severstal-steel-defect 目录结构。")
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        return

    # 2. 清理旧目录并重建
    print("🧹 正在清空旧的 train/val/test 目录并重新组织结构...")
    for s in SPLITS:
        shutil.rmtree(ROOT_DIR / s, ignore_errors=True)
        (ROOT_DIR / s / 'images').mkdir(parents=True, exist_ok=True)
        (ROOT_DIR / s / 'labels').mkdir(parents=True, exist_ok=True)

    # 3. 按 8:1:1 分层重新分配
    print(f"🚀 正在按 {int(RATIOS['train']*10)}:{int(RATIOS['val']*10)}:{int(RATIOS['test']*10)} 比例对 Severstal 数据集进行分层重新划分...")
    random.seed(42) # 固定随机种子，保证每次实验的可重复性
    
    for cls_id, img_names in class_to_data.items():
        random.shuffle(img_names)
        total = len(img_names)
        
        t_end = int(total * RATIOS['train'])
        v_end = t_end + int(total * RATIOS['val'])
        
        tasks = {
            'train': img_names[:t_end],
            'val': img_names[t_end:v_end],
            'test': img_names[v_end:]
        }
        
        for split_name, names in tasks.items():
            for n in names:
                # 从临时文件夹移动回目标文件夹的对应子集
                shutil.move(str(TEMP_DIR / n), str(ROOT_DIR / split_name / 'images' / n))
                lbl_name = n.replace('.jpg', '.txt').replace('.png', '.txt')
                shutil.move(str(TEMP_DIR / lbl_name), str(ROOT_DIR / split_name / 'labels' / lbl_name))

    # 4. 清理临时目录
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print("🎉 Severstal 8:1:1 分层划分彻底完成！")
    print("💡 提示：你可以再次运行统计脚本 `plot_severstal_distribution.py`，查看最新的分布图。")

if __name__ == "__main__":
    resplit_severstal_811()