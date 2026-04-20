import os
import json
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

def calculate_and_sort_scenarios(input_dir, 
                                 output_dir=None,
                                 info_filename="_info_simulate.json",
                                 manifest_filename="_candidate_jpg_manifest.txt",
                                 annotation_filename="_annotation_data.jsonl",
                                 output_filename="_sorted_scenario_manifest.jsonl",
                                 model_name="all-MiniLM-L6-v2"):
    """
    计算用户场景与候选衣物特征向量的相似度并排序
    :param input_dir: 输入文件所在目录
    :param output_dir: 输出文件所在目录 (默认为 input_dir)
    :param info_filename: 用户信息文件名 (包含目标场景)
    :param manifest_filename: 温度筛选后的候选清单名
    :param annotation_filename: 衣物特征标注文件名
    :param output_filename: 最终排序输出文件名
    :param model_name: SentenceTransformer 模型名称或路径
    """
    if output_dir is None:
        output_dir = input_dir
        
    info_path = os.path.join(input_dir, info_filename)
    manifest_path = os.path.join(input_dir, manifest_filename)
    annotation_path = os.path.join(input_dir, annotation_filename)
    output_path = os.path.join(output_dir, output_filename)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. 获取用户需求场景并进行编码 ---
    print(f"📖 正在读取当日用户信息: {info_path}")
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            info_data = json.load(f)
            day_info = info_data[0] if isinstance(info_data, list) else info_data
            user_scenario = day_info.get('usage_scenario', '')
            
            if not user_scenario:
                print("❌ 未能在用户信息中找到 'usage_scenario' 字段！")
                return
            print(f"🎯 用户目标场景: {user_scenario}")
    except Exception as e:
        print(f"❌ 读取用户信息失败: {e}")
        return

    print(f"🚀 正在加载文本模型 {model_name}...")
    try:
        model = SentenceTransformer(model_name, device=device)
        user_embedding = model.encode(user_scenario, convert_to_tensor=True)
    except Exception as e:
        print(f"❌ 模型加载或编码失败: {e}")
        return

    # --- 2. 获取候选名单 ---
    candidate_set = set()
    try:
        print(f"📋 正在读取候选清单: {manifest_path}")
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                filename = line.strip()
                if filename:
                    candidate_set.add(filename)
        print(f"✅ 共找到 {len(candidate_set)} 件候选衣物。")
    except Exception as e:
        print(f"❌ 读取候选清单失败: {e}")
        return

    # --- 3. 读取特征并计算余弦相似度 ---
    results = []
    print(f"🔍 正在计算场景相似度...")
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                img_name = item.get("image_filename")
                
                if img_name in candidate_set:
                    item_embedding_list = item.get("scenario_embedding")
                    if not item_embedding_list:
                        continue
                        
                    item_embedding = torch.tensor(item_embedding_list, device=device)
                    sim = F.cosine_similarity(user_embedding, item_embedding, dim=0).item()
                    
                    results.append({"image_filename": img_name, "similarity": sim})
    except Exception as e:
        print(f"❌ 计算相似度时出错: {e}")
        return

    # --- 4. 排序与输出 ---
    results.sort(key=lambda x: x["similarity"], reverse=True)

    print(f"💾 正在保存排序结果至: {output_path}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for res in results:
                res["similarity"] = round(res["similarity"], 4)
                f_out.write(json.dumps(res, ensure_ascii=False) + '\n')
                
        print("✨ 任务完成！排名前 3 的推荐衣物如下：")
        for i, res in enumerate(results[:3]):
            print(f"  {i+1}. {res['image_filename']} (余弦相似度: {res['similarity']:.4f})")
            
    except Exception as e:
        print(f"❌ 写入输出文件失败: {e}")

# 供自身独立运行测试的代码
if __name__ == "__main__":
    calculate_and_sort_scenarios(input_dir="./user_simulate")