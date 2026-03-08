import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ==================== 🛠️ 用户配置区域 ====================
# 直接使用你从 VS Code 复制的路径结构（去掉了最后的文件名）
CONFIG = {
    "data_dir": "data/clothing-dataset-small-master/train/dress", 
    "input_filename": "_basic_data_dress.jsonl",
    "output_filename": "_text_features_dress.jsonl",
    
    "model_name": "all-MiniLM-L6-v2",
    "device": "cuda"                            # 'cpu' 或 'cuda'
}
# =======================================================

def encode_clothing_dataset():
    """执行编码逻辑"""
    
    # 1. 核心路径修复逻辑
    # 获取当前脚本的绝对路径 (例如: /Users/name/project/codes/extract_text_features.py)
    current_script = Path(__file__).resolve()
    
    # 因为脚本在 codes 文件夹下，parent 是 codes/，再 parent 就是项目根目录
    project_root = current_script.parent.parent
    
    # 拼接完整路径
    base_path = project_root / CONFIG["data_dir"]
    input_file = base_path / CONFIG["input_filename"]
    output_file = base_path / CONFIG["output_filename"]

    # 打印调试信息，方便你确认路径是否正确
    print(f"--- 路径调试信息 ---")
    print(f"📍 脚本位置: {current_script}")
    print(f"🏠 项目根目录: {project_root}")
    print(f"📂 目标数据目录: {base_path}")
    print(f"📄 尝试读取文件: {input_file}")
    print(f"-------------------\n")

    # 2. 检查输入是否存在
    if not input_file.exists():
        print(f"❌ 错误：依然找不到输入文件！")
        print(f"💡 请手动核对这个路径在电脑上是否存在: \n   {input_file.absolute()}")
        return

    # 3. 准备输出目录
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"🚀 正在加载模型 {CONFIG['model_name']}...")
    try:
        model = SentenceTransformer(CONFIG["model_name"], device=CONFIG["device"])
    except Exception:
        print("⚠️ GPU 加载失败，正在切换到 CPU...")
        model = SentenceTransformer(CONFIG["model_name"], device="cpu")
    
    processed_count = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            print(f"📦 开始处理: {CONFIG['input_filename']}")
            
            for line in f_in:
                line = line.strip()
                if not line: continue
                    
                data = json.loads(line)
                
                # 提取文本：将类别和场景组合
                combined_text = f"{data.get('clothing_type', '')}. {' '.join(data.get('usage_scenarios', []))}"
                
                # 生成向量
                embedding = model.encode(combined_text)
                
                # 写入结果
                output_obj = {
                    "image_filename": data.get("image_filename"),
                    "usage_scenarios": data.get("usage_scenarios"),
                    "temperature_range": data.get("temperature_range"),
                    "clothing_type": data.get("clothing_type"),
                    "embedding": embedding.tolist()
                }
                
                f_out.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
                
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"✅ 已处理 {processed_count} 条...")

        print(f"\n✨ 处理完成！")
        print(f"📂 结果保存在: {output_file}")

    except Exception as e:
        print(f"⚠️ 运行出错: {e}")

if __name__ == "__main__":
    encode_clothing_dataset()