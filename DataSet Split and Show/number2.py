import os
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# --- 配置支持中文显示（宋体 + 高清论文字体）---
plt.rcParams['font.sans-serif'] = ['SimSun']    # 改为宋体
plt.rcParams['axes.unicode_minus'] = False       # 解决负号显示异常

# --- 全局字号统一设置（对应论文插入后小四字体）---
FONT_SIZE = 56

# --- 中英文类别映射字典 ---
EN_TO_ZH_MAP = {
    'crease': '折痕',
    'crescent_gap': '新月形缝隙',
    'inclusion': '夹杂物',
    'oil_spot': '油斑',
    'punching_hole': '冲孔',
    'rolled_pit': '轧坑',
    'silk_spot': '丝斑',
    'waist folding': '腰部折痕',
    'water_spot': '水斑',
    'welding_line': '焊缝'
}

# --- 配置路径 ---
DATASET_ROOT = Path(r'../GC10/train') 
YAML_PATH = Path(r'../GC10/data.yaml')
SAVE_PATH = 'GC10_Train_Distribution_CN.png'

def get_class_names(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data.get('names', [])

def count_folder_labels(folder_path):
    label_dir = folder_path / 'labels'
    stats = {}
    if not label_dir.exists():
        return stats
    for txt_file in label_dir.glob('*.txt'):
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                cls_id = int(float(parts[0]))
                stats[cls_id] = stats.get(cls_id, 0) + 1
    return stats

def plot_bar_chart(stats, class_names):
    data = []
    for cls_id, count in stats.items():
        en_name = class_names[cls_id] if cls_id < len(class_names) else f"ID {cls_id}"
        zh_name = EN_TO_ZH_MAP.get(en_name, en_name)
        data.append({'Class': zh_name, 'Count': count})
    
    df = pd.DataFrame(data)
    df = df.sort_values(by='Count', ascending=False)

    # --- 高清大图尺寸 ---
    plt.figure(figsize=(32, 18), dpi=300)
    plt.style.use('seaborn-v0_8-muted') 
    
    bars = plt.bar(df['Class'], df['Count'], color='steelblue', edgecolor='black', alpha=0.8, width=0.6)
    
    # --- 全部统一大字号 ---
    plt.title('GC10-DET数据集类别分布', fontsize=FONT_SIZE, fontweight='bold', pad=20)
    plt.ylabel('数量', fontsize=FONT_SIZE - 8)
    plt.xticks(rotation=45, ha='right', fontsize=FONT_SIZE - 12)
    plt.yticks(fontsize=FONT_SIZE - 12)
    
    # ====================== 固定Y轴最大值 800 ======================
    plt.ylim(0, 800)

    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # ====================== 数字再往上移，彻底不重叠 ======================
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.annotate(f'{int(height)}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 40),  # 从25 → 40，完全避开坐标轴
                         textcoords="offset points",
                         ha='center', va='bottom',
                         fontsize=FONT_SIZE - 16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ 高清统计图已保存至: {SAVE_PATH}")

if __name__ == "__main__":
    names = get_class_names(YAML_PATH)
    print(f"正在扫描: {DATASET_ROOT / 'labels'}")
    counts = count_folder_labels(DATASET_ROOT)
    
    if not counts:
        print("❌ 未发现标签数据，请确认路径是否正确")
    else:
        plot_bar_chart(counts, names)