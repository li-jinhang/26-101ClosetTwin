import os
import torch
import json
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# 修复：分别从定义了模型类的脚本中进行导入
from train_temperature_model import TempRegressor
from train_scenary_model import ScenarioRegressor

# ==================== 🛠️ 配置区域 ====================
CONFIG = {
    # 基础路径配置
    "project_root": ".", 
    "image_dir": "./user_simulate",
    "output_jsonl": "./user_simulate/_z1_annotation_data.jsonl",
    
    # 模型路径
    "vit_model_path": "./mobilevit-small",
    "temp_model_path": "models/temperature_regressor.pth",
    "scenario_model_path": "models/scenario_regressor.pth",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
# ====================================================

class WardrobeProcessor:
    """
    负责衣物图片的特征提取、温度预测及场景编码
    """
    def __init__(self, config):
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.project_root = Path(config["project_root"])
        
        # 1. 加载视觉模型
        print(f"🚀 正在加载视觉模型: {config['vit_model_path']}...")
        self.processor = AutoImageProcessor.from_pretrained(config["vit_model_path"])
        self.vit_model = AutoModel.from_pretrained(config["vit_model_path"]).to(self.device)
        self.vit_model.eval()

        # 2. 加载预测模型
        print(f"🌡️ 正在加载温度预测与场景回归模型...")
        self.temp_regressor = TempRegressor().to(self.device)
        self.temp_regressor.load_state_dict(torch.load(self.project_root / config["temp_model_path"], map_location=self.device))
        self.temp_regressor.eval()

        self.scenario_regressor = ScenarioRegressor(input_dim=640, output_dim=384).to(self.device)
        self.scenario_regressor.load_state_dict(torch.load(self.project_root / config["scenario_model_path"], map_location=self.device))
        self.scenario_regressor.eval()

    def process_images(self, image_dir, output_jsonl):
        """扫描目录并生成标注数据"""
        img_folder = Path(image_dir)
        output_path = Path(output_jsonl)
        
        # 获取已处理列表实现增量更新
        processed = self._get_processed_files(output_path)
        images = [f for f in img_folder.glob("*.jpg") if f.name not in processed]
        
        if not images:
            print("✨ 文件夹内没有需要新处理的图片。")
            return 0

        print(f"📊 发现 {len(images)} 张待处理的图片。")

        with open(output_path, 'a', encoding='utf-8') as f:
            for img_path in images:
                result = self._annotate_single_image(img_path)
                if result:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    print(f"✅ 已处理: {result['image_filename']} -> {result['recommended_temperature']}°C")
        return len(images)

    def _annotate_single_image(self, img_path):
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.vit_model(**inputs)
                # 提取 640 维视觉特征
                v_feat = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
                
                # 预测推荐温度与场景特征 (384维)
                pred_temp = self.temp_regressor(v_feat).item()
                s_feat = self.scenario_regressor(v_feat).squeeze().tolist()
                
            return {
                "image_filename": img_path.name,
                "recommended_temperature": round(pred_temp, 1),
                "scenario_embedding": s_feat
            }
        except Exception as e:
            print(f"⚠️ 处理出错 {img_path.name}: {e}")
            return None

    def _get_processed_files(self, path):
        if not path.exists(): return set()
        with open(path, 'r', encoding='utf-8') as f:
            return {json.loads(line)["image_filename"] for line in f if line.strip()}

# --- 主程序执行入口 ---
if __name__ == "__main__":
    # 确保输入/输出目录存在
    os.makedirs(CONFIG["image_dir"], exist_ok=True)
    
    print("========================================")
    processor = WardrobeProcessor(CONFIG)
    print(f"\n📂 开始扫描目录: {CONFIG['image_dir']}")
    processed_count = processor.process_images(CONFIG["image_dir"], CONFIG["output_jsonl"])
    
    print(f"\n✨ 任务结束！本次共新增处理 {processed_count} 件衣物。")
    print(f"💾 标注数据保存在: {CONFIG['output_jsonl']}")
    print("========================================")