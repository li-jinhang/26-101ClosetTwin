import torch
import json
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
# 导入训练脚本中的模型定义
from train_temperature_model import TempRegressor 

# ==================== 🛠️ 配置区域 ====================
CONFIG = {
    "model_path": "./mobilevit-small",              # 视觉模型路径
    "regressor_path": "models/temperature_regressor.pth",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
# ====================================================

class TemperaturePredictor:
    def __init__(self):
        # 1. 加载特征提取器
        self.processor = AutoImageProcessor.from_pretrained(CONFIG["model_path"])
        self.vit_model = AutoModel.from_pretrained(CONFIG["model_path"]).to(CONFIG["device"])
        self.vit_model.eval()

        # 2. 加载回归模型
        project_root = Path(__file__).resolve().parent.parent
        self.regressor = TempRegressor().to(CONFIG["device"])
        self.regressor.load_state_dict(torch.load(project_root / CONFIG["regressor_path"]))
        self.regressor.eval()

    def predict(self, image_path):
        """输入图片路径，输出预测温度"""
        # 提取视觉特征
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(CONFIG["device"])
        
        with torch.no_grad():
            outputs = self.vit_model(**inputs)
            # 获取 640 维特征向量
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embedding = outputs.pooler_output
            else:
                embedding = outputs.last_hidden_state.mean(dim=1)
            
            # 回归预测
            pred_temp = self.regressor(embedding)
            return pred_temp.item()

if __name__ == "__main__":
    # 测试预测
    predictor = TemperaturePredictor()
    test_img = "data/clothing-dataset-small-master/train/dress/0a69db60-c052-4b9a-a90d-e53120d091d5.jpg" # 替换为实际图片路径
    
    try:
        temp = predictor.predict(test_img)
        print(f"🖼️ 图片: {Path(test_img).name}")
        print(f"🌡️ 预测适用温度: {temp:.1f}°C")
    except Exception as e:
        print(f"❌ 预测失败: {e}")