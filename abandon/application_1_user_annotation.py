import os
import json
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
# 从训练脚本中导入模型定义
from train_temperature_model import TempRegressor
from train_scenary_model import ScenarioRegressor  #

# ==================== 🛠️ 配置区域 ====================
CONFIG = {
    # 目标图片文件夹
    "image_dir": "user_simulate/", 
    # 标注结果保存的 JSONL 文件名
    "output_jsonl": "_annotation_data.jsonl",
    
    # 模型路径
    "vit_model_path": "./mobilevit-small", 
    "regressor_path": "models/temperature_regressor.pth",
    "scenario_regressor_path": "models/scenario_regressor.pth",  #
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
# ====================================================

class SupplementaryAnnotator:
    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent
        self.image_folder = project_root / CONFIG["image_dir"]
        self.output_file = self.image_folder / CONFIG["output_jsonl"]
        
        # 1. 加载视觉特征提取器 (MobileViT)
        print(f"🚀 正在加载视觉模型: {CONFIG['vit_model_path']}...")
        self.processor = AutoImageProcessor.from_pretrained(CONFIG["vit_model_path"])
        self.vit_model = AutoModel.from_pretrained(CONFIG["vit_model_path"]).to(CONFIG["device"])
        self.vit_model.eval()

        # 2. 加载温度回归模型
        print(f"🌡️ 正在加载温度预测模型: {CONFIG['regressor_path']}...")
        self.regressor = TempRegressor().to(CONFIG["device"])
        regressor_full_path = project_root / CONFIG["regressor_path"]
        if regressor_full_path.exists():
            self.regressor.load_state_dict(torch.load(regressor_full_path, map_location=CONFIG["device"]))
            self.regressor.eval()
        else:
            raise FileNotFoundError(f"❌ 未找到温度回归模型权重文件: {regressor_full_path}")

        # 3. 加载场景回归模型 (将视觉特征映射为 384 维文本语义特征)
        print(f"🎭 正在加载场景预测模型: {CONFIG['scenario_regressor_path']}...")
        self.scenario_regressor = ScenarioRegressor(input_dim=640, output_dim=384).to(CONFIG["device"])
        scenario_full_path = project_root / CONFIG["scenario_regressor_path"]
        if scenario_full_path.exists():
            self.scenario_regressor.load_state_dict(torch.load(scenario_full_path, map_location=CONFIG["device"]))
            self.scenario_regressor.eval()
        else:
            print(f"⚠️ 警告：未找到场景回归模型权重，将跳过文本特征转换。")
            self.scenario_regressor = None

    def get_annotated_files(self):
        """获取已经标注过的图片列表"""
        annotated = set()
        if self.output_file.exists():
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        annotated.add(data["image_filename"])
                    except:
                        continue
        return annotated

    def process(self):
        # 1. 扫描文件夹
        all_images = list(self.image_folder.glob("*.jpg"))
        annotated_set = self.get_annotated_files()
        
        # 2. 筛选未标注图片
        pending_images = [img for img in all_images if img.name not in annotated_set]
        
        if not pending_images:
            print("✨ 所有图片已完成标注，无需处理。")
            return

        print(f"📊 发现 {len(all_images)} 张图片，其中 {len(pending_images)} 张待标注。")

        # 3. 批量处理并追加写入
        with open(self.output_file, 'a', encoding='utf-8') as f_out:
            for img_path in pending_images:
                try:
                    # 提取特征与预测温度
                    image = Image.open(img_path).convert("RGB")
                    inputs = self.processor(images=image, return_tensors="pt").to(CONFIG["device"])
                    
                    with torch.no_grad():
                        outputs = self.vit_model(**inputs)
                        # 获取 640 维视觉特征向量
                        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                            embedding_tensor = outputs.pooler_output
                        else:
                            embedding_tensor = outputs.last_hidden_state.mean(dim=1)
                        
                        # 预测温度
                        pred_temp = self.regressor(embedding_tensor).item()
                        
                        # 使用场景模型转换特征向量
                        scenario_embedding_list = []
                        if self.scenario_regressor:
                            scenario_vec = self.scenario_regressor(embedding_tensor)
                            scenario_embedding_list = scenario_vec.squeeze().tolist()
                        
                        embedding_list = embedding_tensor.squeeze().tolist()

                    # 构建输出对象
                    result = {
                        "image_filename": img_path.name,
                        "recommended_temperature": round(pred_temp, 1),
                        "visual_embedding": embedding_list,
                        "scenario_embedding": scenario_embedding_list  # 新增字段：预测的文本特征向量
                    }
                    
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    print(f"✅ 已完成: {img_path.name} -> {result['recommended_temperature']}°C (已提取场景特征)")

                except Exception as e:
                    print(f"⚠️ 处理 {img_path.name} 出错: {e}")

        print(f"\n✨ 补充标注任务结束！结果保存在: {self.output_file}")

if __name__ == "__main__":
    annotator = SupplementaryAnnotator()
    annotator.process()