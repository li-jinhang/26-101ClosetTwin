import torch
import json
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
# 导入场景训练脚本中的模型定义
from train_scenary_model import ScenarioRegressor 

# ==================== 🛠️ 配置区域 ====================
CONFIG = {
    "model_path": "./mobilevit-small",              # 视觉模型路径
    "scenario_model_path": "models/scenario_regressor.pth",
    "reference_data": "data/clothing-dataset-small-master/train/dress/_text_features_dress.jsonl",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
# ====================================================

class ScenarioPredictor:
    def __init__(self):
        # 1. 加载特征提取器 (MobileViT)
        self.processor = AutoImageProcessor.from_pretrained(CONFIG["model_path"])
        self.vit_model = AutoModel.from_pretrained(CONFIG["model_path"]).to(CONFIG["device"])
        self.vit_model.eval()

        # 2. 加载场景回归模型 (640 -> 384)
        project_root = Path(__file__).resolve().parent.parent
        self.regressor = ScenarioRegressor(input_dim=640, output_dim=384).to(CONFIG["device"])
        self.regressor.load_state_dict(torch.load(project_root / CONFIG["scenario_model_path"]))
        self.regressor.eval()
        
        # 3. 加载参考数据集（用于将向量转回文字标签）
        self.reference_features = []
        self.reference_labels = []
        ref_path = project_root / CONFIG["reference_data"]
        if ref_path.exists():
            with open(ref_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    self.reference_features.append(item["embedding"])
                    # 组合类别和场景作为标签
                    label = f"{item['clothing_type']} ({', '.join(item['usage_scenarios'])})"
                    self.reference_labels.append(label)
            self.reference_features = torch.tensor(self.reference_features).to(CONFIG["device"])

    def predict(self, image_path, top_k=3):
        """输入图片路径，输出预测的语义向量及最匹配的场景"""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(CONFIG["device"])
        
        with torch.no_grad():
            # 提取视觉特征
            outputs = self.vit_model(**inputs)
            visual_embedding = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
            
            # 映射为语义向量 (384维)
            predicted_vec = self.regressor(visual_embedding)
            
            # 如果有参考数据，计算余弦相似度进行检索
            if len(self.reference_features) > 0:
                # 归一化以计算余弦相似度
                pred_norm = predicted_vec / predicted_vec.norm(dim=1, keepdim=True)
                ref_norm = self.reference_features / self.reference_features.norm(dim=1, keepdim=True)
                
                similarities = torch.mm(pred_norm, ref_norm.t()).squeeze()
                scores, indices = torch.topk(similarities, top_k)
                
                results = []
                for score, idx in zip(scores, indices):
                    results.append({
                        "label": self.reference_labels[idx.item()],
                        "confidence": score.item()
                    })
                return results
            
            return predicted_vec.tolist()

if __name__ == "__main__":
    predictor = ScenarioPredictor()
    # 测试图片路径
    test_img = "data/clothing-dataset-small-master/train/dress/0a69db60-c052-4b9a-a90d-e53120d091d5.jpg"
    try:
        print(f"🔍 正在分析图片: {Path(test_img).name}")
        matches = predictor.predict(test_img)
        
        print("✨ 最匹配的场景预测：")
        for i, res in enumerate(matches):
            print(f"  {i+1}. {res['label']} (相似度: {res['confidence']:.4f})")
            
    except Exception as e:
        print(f"❌ 预测失败: {e}")