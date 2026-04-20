import json
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# ==================== 🛠️ 用户配置区域 ====================
CONFIG = {
    # 目标图片文件夹路径
    "image_dir": "data/train", 
    # 输出特征文件名
    "output_filename": "_visual_features.jsonl",
    
    # 模型路径
    "model_path": "./mobilevit-small", 
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
# =======================================================

def extract_visual_features():
    """执行视觉特征提取逻辑"""
    
    # 1. 路径初始化
    current_script = Path(__file__).resolve()
    # 假设项目结构中 data 文件夹在脚本的上一级或同级，此处采用相对路径逻辑
    project_root = current_script.parent.parent
    
    image_folder = project_root / CONFIG["image_dir"]
    output_file = image_folder / CONFIG["output_filename"]

    print(f"--- 路径调试信息 ---")
    print(f"📂 目标图片目录: {image_folder}")
    print(f"📄 结果保存路径: {output_file}")
    print(f"-------------------\n")

    if not image_folder.exists():
        print(f"❌ 错误：找不到图片文件夹！路径: {image_folder}")
        return

    # 2. 加载模型和处理器
    print(f"🚀 正在加载视觉模型 (设备: {CONFIG['device']})...")
    try:
        processor = AutoImageProcessor.from_pretrained(CONFIG["model_path"])
        model = AutoModel.from_pretrained(CONFIG["model_path"])
        model.to(CONFIG["device"])
        model.eval()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 3. 遍历图片并提取特征
    image_files = list(image_folder.glob("*.jpg"))
    print(f"📦 找到 {len(image_files)} 张图片，开始处理...")

    processed_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for img_path in image_files:
            try:
                # 读取并转换图片
                image = Image.open(img_path).convert("RGB")
                
                # 预处理
                inputs = processor(images=image, return_tensors="pt").to(CONFIG["device"])
                
                # 提取特征 (不计算梯度)
                with torch.no_grad():
                    outputs = model(**inputs)
                    
                    # MobileViT-small 默认输出为 640 维
                    # 优先获取 pooler_output (全局平均池化后的结果)
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        features = outputs.pooler_output
                    else:
                        # 兜底方案：对最后隐藏层取空间维度均值
                        features = outputs.last_hidden_state.mean(dim=1)
                
                # 转换为列表
                feature_vector = features.squeeze().tolist()
                
                # 写入结果
                output_obj = {
                    "image_filename": img_path.name,
                    "visual_embedding": feature_vector,
                    "dimension": len(feature_vector)
                }
                
                f_out.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
                
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"✅ 已处理 {processed_count} 张图片...")

            except Exception as e:
                print(f"⚠️ 处理图片 {img_path.name} 时出错: {e}")

    print(f"\n✨ 处理完成！共提取 {processed_count} 个视觉特征向量。")
    print(f"📂 结果保存在: {output_file}")

if __name__ == "__main__":
    extract_visual_features()