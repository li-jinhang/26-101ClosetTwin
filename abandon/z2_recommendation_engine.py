import os
import json
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# ==================== 🛠️ 配置区域 ====================
CONFIG = {
    # 文件夹与文件路径
    "input_dir": "./user_simulate",
    "info_filename": "_info_simulate.json",
    # 读取上一步 (z1) 输出的标注文件，如果名称不同请自行修改
    "annotation_filename": "_z1_annotation_data.jsonl", 
    "output_filename": "_z2_recommendation_results.jsonl",
    
    # 文本编码模型
    "model_name": "all-MiniLM-L6-v2"
}
# ====================================================

class RecommendationEngine:
    """
    负责根据用户当前环境（温度、场景）筛选并排序衣物
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_model = SentenceTransformer(model_name, device=self.device)

    def get_recommendations(self, day_info, annotation_data_path):
        """
        核心推荐逻辑
        1. 温度过滤
        2. 场景相似度计算
        """
        temp_min = day_info['temp_range']['min']
        temp_max = day_info['temp_range']['max']
        target_scenario = day_info.get('usage_scenario', '')

        # 编码目标场景文本
        user_scenario_vec = self.text_model.encode(target_scenario, convert_to_tensor=True)

        recommendations = []
        with open(annotation_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                rec_temp = item.get('recommended_temperature')
                
                # 步骤 1: 温度硬筛选
                if rec_temp is not None and temp_min <= rec_temp <= temp_max:
                    # 步骤 2: 场景软排序（计算余弦相似度）
                    item_vec = torch.tensor(item['scenario_embedding'], device=self.device)
                    similarity = F.cosine_similarity(user_scenario_vec, item_vec, dim=0).item()
                    
                    recommendations.append({
                        "image_filename": item['image_filename'],
                        "score": round(similarity, 4),
                        "temp": rec_temp
                    })

        # 按相似度降序排序
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations

# --- 主程序执行入口 ---
if __name__ == "__main__":
    # 1. 路径拼接
    info_path = os.path.join(CONFIG["input_dir"], CONFIG["info_filename"])
    annotation_path = os.path.join(CONFIG["input_dir"], CONFIG["annotation_filename"])
    output_path = os.path.join(CONFIG["input_dir"], CONFIG["output_filename"])

    print("========================================")
    # 2. 读取当日用户信息
    print(f"📖 正在读取当日用户信息: {info_path}")
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            info_data = json.load(f)
            day_info = info_data[0] if isinstance(info_data, list) else info_data
            print(f"🎯 目标温度: {day_info['temp_range']['min']}°C - {day_info['temp_range']['max']}°C")
            print(f"🎯 目标场景: {day_info.get('usage_scenario', '未知')}")
    except Exception as e:
        print(f"❌ 读取用户信息失败: {e}")
        exit()

    # 3. 初始化并计算推荐
    print(f"🚀 正在加载推荐引擎 (模型: {CONFIG['model_name']})...")
    engine = RecommendationEngine(model_name=CONFIG["model_name"])
    
    print(f"🔍 正在从 {CONFIG['annotation_filename']} 中筛选推荐衣物...")
    try:
        recommendations = engine.get_recommendations(day_info, annotation_path)
    except Exception as e:
        print(f"❌ 推荐计算失败: {e}")
        exit()

    # 4. 写入输出文件
    print(f"💾 正在保存推荐结果至: {output_path}")
    try:
        os.makedirs(CONFIG["input_dir"], exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for res in recommendations:
                f_out.write(json.dumps(res, ensure_ascii=False) + '\n')
                
        print("\n✨ 任务完成！排名前 3 的推荐衣物如下：")
        for i, res in enumerate(recommendations[:3]):
            print(f"  {i+1}. {res['image_filename']} (匹配得分: {res['score']:.4f}, 适用温度: {res['temp']}°C)")
    except Exception as e:
        print(f"❌ 写入输出文件失败: {e}")
    print("========================================")