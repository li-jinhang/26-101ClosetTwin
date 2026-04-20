import os
import json
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# ==================== 🛠️ 配置区域 ====================
CONFIG = {
    # 文件夹路径配置
    "input_dir": "./user_simulate",
    
    # 输入文件名
    "info_filename": "_info_simulate.json",
    "manifest_filename": "_candidate_jpg_manifest.txt",
    "annotation_filename": "_annotation_data.jsonl",
    
    # 输出文件名
    "output_filename": "_sorted_scenario_manifest.jsonl",
    
    # 模型配置
    "model_name": "all-MiniLM-L6-v2",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
# ====================================================

def calculate_and_sort_scenarios():
    # --- 1. 路径拼接 ---
    info_path = os.path.join(CONFIG["input_dir"], CONFIG["info_filename"])
    manifest_path = os.path.join(CONFIG["input_dir"], CONFIG["manifest_filename"])
    annotation_path = os.path.join(CONFIG["input_dir"], CONFIG["annotation_filename"])
    output_path = os.path.join(CONFIG["input_dir"], CONFIG["output_filename"])

    # --- 2. 获取用户需求场景并进行编码 ---
    print(f"📖 正在读取当日用户信息: {info_path}")
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            info_data = json.load(f)
            # 兼容读取列表或字典格式
            day_info = info_data[0] if isinstance(info_data, list) else info_data
            user_scenario = day_info.get('usage_scenario', '')
            
            if not user_scenario:
                print("❌ 未能在用户信息中找到 'usage_scenario' 字段！请检查 info JSON。")
                return
            print(f"🎯 用户目标场景: {user_scenario}")
    except Exception as e:
        print(f"❌ 读取用户信息失败: {e}")
        return

    print(f"🚀 正在加载文本模型 {CONFIG['model_name']}...")
    try:
        model = SentenceTransformer(CONFIG["model_name"], device=CONFIG["device"])
        # 将用户场景编码为 Tensor
        user_embedding = model.encode(user_scenario, convert_to_tensor=True)
    except Exception as e:
        print(f"❌ 模型加载或编码失败: {e}")
        return

    # --- 3. 获取通过温度筛选的候选衣物名单 ---
    candidate_set = set()
    try:
        print(f"📋 正在读取候选清单: {manifest_path}")
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                filename = line.strip()
                if filename:
                    candidate_set.add(filename)
        print(f"✅ 从温度筛选清单中共找到 {len(candidate_set)} 件候选衣物。")
    except Exception as e:
        print(f"❌ 读取候选清单失败: {e}")
        return

    # --- 4. 读取衣物特征并计算余弦相似度 ---
    results = []
    print(f"🔍 正在读取衣物标注数据以计算相似度: {annotation_path}")
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                img_name = item.get("image_filename")
                
                # 仅对温度筛选后的候选衣物进行计算
                if img_name in candidate_set:
                    item_embedding_list = item.get("scenario_embedding")
                    
                    if not item_embedding_list:
                        print(f"⚠️ 警告: {img_name} 缺少 scenario_embedding，已跳过。")
                        continue
                        
                    # 转换为 Tensor 以进行计算
                    item_embedding = torch.tensor(item_embedding_list, device=CONFIG["device"])
                    
                    # 使用 PyTorch 计算一维张量的余弦相似度
                    sim = F.cosine_similarity(user_embedding, item_embedding, dim=0).item()
                    
                    results.append({
                        "image_filename": img_name,
                        "similarity": sim
                    })
    except Exception as e:
        print(f"❌ 读取标注数据或计算相似度时出错: {e}")
        return

    # --- 5. 对结果按照余弦相似度进行降序排序 ---
    results.sort(key=lambda x: x["similarity"], reverse=True)

    # --- 6. 将排序后的结果输出为 JSONL ---
    print(f"💾 正在保存排序结果至: {output_path}")
    try:
        # 确保输出目录存在
        os.makedirs(CONFIG["input_dir"], exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for res in results:
                # 保留 4 位小数使数据更易读
                res["similarity"] = round(res["similarity"], 4)
                f_out.write(json.dumps(res, ensure_ascii=False) + '\n')
                
        print("✨ 任务完成！排名前 3 的推荐衣物如下：")
        for i, res in enumerate(results[:3]):
            print(f"  {i+1}. {res['image_filename']} (余弦相似度: {res['similarity']:.4f})")
            
    except Exception as e:
        print(f"❌ 写入输出文件失败: {e}")

if __name__ == "__main__":
    calculate_and_sort_scenarios()