import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ==================== 🛠️ 配置区域 ====================
CONFIG = {
    # 存放输入 JSONL 文件的目录
    "data_dir": "data/clothing-dataset-small-master/train/dress", 
    # 输入文件名：格式需为 {"temperature_range": "...", "usage_scenario": "..."}
    "input_filename": "scenarios_input.jsonl",
    # 输出文件名：包含原始数据和场景特征向量
    "output_filename": "_scenario_features_adaptation.jsonl",
    
    # 使用项目中已定义的文本编码模型
    "model_name": "all-MiniLM-L6-v2",
    "device": "cuda" # 若无 GPU 会自动切换到 cpu
}
# =======================================================

def extract_scenario_features():
    """执行场景适配的特征提取逻辑"""
    
    # 1. 路径初始化逻辑
    current_script = Path(__file__).resolve()
    project_root = current_script.parent.parent
    base_path = project_root / CONFIG["data_dir"]
    input_file = base_path / CONFIG["input_filename"]
    output_file = base_path / CONFIG["output_filename"]

    print(f"--- 场景适配路径调试 ---")
    print(f"📂 目标数据目录: {base_path}")
    print(f"📄 尝试读取文件: {input_file}")
    print(f"-----------------------\n")

    if not input_file.exists():
        print(f"❌ 错误：未找到输入文件 {input_file}")
        return

    # 2. 加载预训练模型
    print(f"🚀 正在加载模型 {CONFIG['model_name']}...")
    try:
        model = SentenceTransformer(CONFIG["model_name"], device=CONFIG["device"])
    except Exception:
        print("⚠️ GPU 加载失败，正在切换到 CPU...")
        model = SentenceTransformer(CONFIG["model_name"], device="cpu")
    
    processed_count = 0
    
    # 3. 处理数据
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            print(f"📦 开始提取场景向量...")
            
            for line in f_in:
                line = line.strip()
                if not line: continue
                    
                data = json.loads(line)
                
                # 提取特定字段：单一适用场景
                scenario_text = data.get('usage_scenario', '')
                
                if not scenario_text:
                    print(f"⚠️ 跳过空场景行: {line}")
                    continue

                # 生成文本特征向量
                embedding = model.encode(scenario_text)
                
                # 构建输出对象，保留原始字段并增加 embedding
                output_obj = {
                    "temperature_range": data.get("temperature_range"),
                    "usage_scenario": scenario_text,
                    "scenario_embedding": embedding.tolist() # 转换为列表以存入 JSON
                }
                
                f_out.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
                
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"✅ 已处理 {processed_count} 条场景适配数据...")

        print(f"\n✨ 处理完成！共生成 {processed_count} 条特征向量。")
        print(f"📂 结果保存在: {output_file}")

    except Exception as e:
        print(f"⚠️ 运行出错: {e}")

if __name__ == "__main__":
    extract_scenario_features()