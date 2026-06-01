import os
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# --- 配置支持中文显示 ---
plt.rcParams['font.sans-serif'] = ['SimSun']    # 改为宋体
plt.rcParams['axes.unicode_minus'] = False       # 解决负号 - 显示异常

# --- 全局字号统一设置（对应你要的 FONT_SIZE=56 效果）---
FONT_SIZE = 56

# --- Severstal 纯英文类别映射字典 ---
CLASS_MAP = {
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

# --- 数据集划分名称映射 ---
SPLIT_MAP = {
    'train': '训练集 (Train)',
    'val': '验证集 (Val)',
    'test': '测试集 (Test)'
}

# --- 配置路径 ---
DATASET_ROOT = Path(r'../GC10')
YAML_PATH = DATASET_ROOT / 'data.yaml'
SAVE_PATH = 'Severstal_Distribution.png'

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

def plot_comparison_chart(all_stats, class_names):
    df_list = []
    for split_name, stats in all_stats.items():
        mapped_split = SPLIT_MAP.get(split_name, split_name)
        for cls_id, count in stats.items():
            en_name = str(class_names[cls_id]) if cls_id < len(class_names) else f"{cls_id + 1}"
            final_class_name = CLASS_MAP.get(en_name.lower(), f'Class {cls_id + 1}')
            df_list.append({'Split': mapped_split, 'Class': final_class_name, 'Count': count})
    
    df = pd.DataFrame(df_list)
    pivot_df = df.pivot(index='Class', columns='Split', values='Count').fillna(0)
    
    train_col = SPLIT_MAP['train']
    if train_col in pivot_df.columns:
        pivot_df = pivot_df.sort_values(by=train_col, ascending=False)

    # --- 画布放大，保证高清 ---
    plt.figure(figsize=(32, 18), dpi=300)
    plt.style.use('seaborn-v0_8-muted')
    
    ax = pivot_df.plot(kind='bar', width=0.8, edgecolor='white', ax=plt.gca())
    
    # --- 全部统一大字号 ---
    plt.title('GC10-DET数据集划分', fontsize=FONT_SIZE, fontweight='bold', pad=20)
    plt.ylabel('目标数量', fontsize=FONT_SIZE - 8)
    plt.xticks(rotation=45, ha='right', fontsize=FONT_SIZE - 12)
    plt.yticks(fontsize=FONT_SIZE - 12)
    
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(title='数据集划分', fontsize=FONT_SIZE - 14, title_fontsize=FONT_SIZE - 10)

    # --- 柱子数值字号也放大 ---
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(
                str(int(p.get_height())),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom',
                fontsize=FONT_SIZE - 16
            )

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ 高清分布图已保存至: {SAVE_PATH}")

if __name__ == "__main__":
    if not YAML_PATH.exists():
        print(f"❌ 未找到 {YAML_PATH}")
    else:
        names = get_class_names(YAML_PATH)
        all_splits_stats = {
            'train': count_folder_labels(DATASET_ROOT / 'train'),
            'val': count_folder_labels(DATASET_ROOT / 'val'),
            'test': count_folder_labels(DATASET_ROOT / 'test')
        }
        if not any(all_splits_stats.values()):
            print("❌ 未检测到标签")
        else:
            plot_comparison_chart(all_splits_stats, names)