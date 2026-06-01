import os
import yaml
import random
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# --- 配置支持中文显示 (改为宋体) ---
# Windows系统下宋体通常为 simsun.ttc，若是其他系统请修改对应的宋体路径
FONT_PATH = r'C:\Windows\Fonts\simsun.ttc'  
FONT_SIZE = 56  # 字体调大

# --- GC10 类别映射字典 ---
# 将 yaml 中的标签转为论文配图规范格式 (去下划线、首字母大写)
CLASS_DISPLAY_MAP = {
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
# 指向你的 GC10 数据集文件夹，请根据实际路径修改
DATASET_ROOT = Path(r'../GC10')  
IMAGE_DIR = DATASET_ROOT / 'train' / 'images'
LABEL_DIR = DATASET_ROOT / 'train' / 'labels'
YAML_PATH = DATASET_ROOT / 'data.yaml'
SAVE_PATH = 'GC10_defect_gallery.jpg'

def get_metadata(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data.get('names', [])

def find_representative_images(names):
    found_examples = {}
    remaining_classes = set(range(len(names)))
    
    label_files = list(LABEL_DIR.glob('*.txt'))
    if not label_files:
        print("❌ 错误：未在 train/labels 中找到标签文件。")
        return None

    random.seed(42) 
    random.shuffle(label_files)
    
    print("🔍 正在为 GC10 寻找各类别示例图片...")
    for lbl_file in label_files:
        if not remaining_classes:
            break 
            
        # 兼容图片格式
        img_file = IMAGE_DIR / lbl_file.name.replace('.txt', '.jpg')
        if not img_file.exists():
            img_file = IMAGE_DIR / lbl_file.name.replace('.txt', '.png')
        if not img_file.exists(): continue
        
        with open(lbl_file, 'r') as f:
            lines = f.readlines()
            if not lines: continue
            
            # 读取该图片包含的所有类别
            classes_in_img = set([int(float(line.split()[0])) for line in lines])
            target_classes = classes_in_img.intersection(remaining_classes)
            
            if target_classes:
                chosen_cls = list(target_classes)[0]
                found_examples[chosen_cls] = img_file
                remaining_classes.remove(chosen_cls)
                print(f"✅ 已提取: [{chosen_cls}] {names[chosen_cls]}")

    return found_examples

def create_gallery(examples, names):
    if not examples: return

    sorted_cls_ids = sorted(examples.keys())
    
    # 统一尺寸保证清晰度
    grid_w, grid_h = 400, 400
    # GC10 有 10 个类别，采用 2行 x 5列 布局
    cols = 5
    rows = 2
    
    total_w = cols * grid_w
    total_h = rows * grid_h
    gallery_img = Image.new('RGB', (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(gallery_img)
    
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except IOError:
        print("⚠️ 警告：找不到宋体文件，将使用默认字体！")
        font = ImageFont.load_default()

    print("🧱 正在拼接 2x5 网格图 (白底黑字，宋体大字)...")
    for i, cls_id in enumerate(sorted_cls_ids):
        img_path = examples[cls_id]
        
        try:
            with Image.open(img_path) as img:
                img_resized = img.resize((grid_w, grid_h), Image.Resampling.LANCZOS)
                
                col = i % cols
                row = i // cols
                paste_x = col * grid_w
                paste_y = row * grid_h
                
                gallery_img.paste(img_resized, (paste_x, paste_y))
                
                # 获取展示名称 (字典映射或首字母大写)
                raw_name = str(names[cls_id]).lower()
                display_name = CLASS_DISPLAY_MAP.get(raw_name, raw_name.capitalize().replace('_', ' '))
                
                # 适配新版 Pillow 获取文字宽高
                bbox = font.getbbox(display_name)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                
                # 绘制纯白底色条 (因字体变大，高度增加到 80)
                text_bg_h = 85
                bg_y = paste_y + grid_h - text_bg_h
                draw.rectangle([paste_x, bg_y, paste_x + grid_w, paste_y + grid_h], fill=(255, 255, 255))
                
                # 绘制黑色文字居中
                text_x = paste_x + (grid_w - text_w) // 2
                text_y = bg_y + (text_bg_h - text_h) // 2 - 8 # 微调垂直居中
                draw.text((text_x, text_y), display_name, fill=(0, 0, 0), font=font)
                
        except Exception as e:
            print(f"⚠️ 处理失败 {img_path}: {e}")

    gallery_img.save(SAVE_PATH, quality=95)
    print(f"🎉 GC10 全家福生成完毕，已保存至: {SAVE_PATH}")

if __name__ == "__main__":
    if not YAML_PATH.exists():
        print(f"❌ 未找到 {YAML_PATH}，请检查 DATASET_ROOT 路径是否正确。")
    else:
        names = get_metadata(YAML_PATH)
        found_examples = find_representative_images(names)
        create_gallery(found_examples, names)