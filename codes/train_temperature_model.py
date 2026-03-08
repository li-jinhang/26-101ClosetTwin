import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import re
from pathlib import Path

# ==================== 🛠️ 配置区域 ====================
CONFIG = {
    "data_dir": "data/clothing-dataset-small-master/train/dress",
    "basic_data": "_basic_data_dress.jsonl",      # 包含温度标注
    "visual_features": "_visual_features_dress.jsonl", # 包含 640 维特征
    "model_save_path": "models/temperature_regressor.pth",
    "epochs": 100,
    "lr": 0.001,
    "batch_size": 16
}
# ====================================================

class TempRegressor(nn.Module):
    """简单的全连接回归模型"""
    def __init__(self, input_dim=640):
        super(TempRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # 输出单个温度值
        )

    def forward(self, x):
        return self.net(x)

def parse_temp_median(temp_str):
    """解析 '15°C - 25°C' 为 20.0"""
    nums = re.findall(r"(-?\d+)", temp_str)
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2
    elif len(nums) == 1:
        return float(nums[0])
    return 20.0 # 默认兜底值

def train_model():
    project_root = Path(__file__).resolve().parent.parent
    base_path = project_root / CONFIG["data_dir"]
    
    # 1. 加载标注数据
    labels_map = {}
    with open(base_path / CONFIG["basic_data"], 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            temp_str = item["temperature_range"]["range"]
            labels_map[item["image_filename"]] = parse_temp_median(temp_str)

    # 2. 加载视觉特征并对齐标签
    features = []
    targets = []
    with open(base_path / CONFIG["visual_features"], 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            fname = item["image_filename"]
            if fname in labels_map:
                features.append(item["visual_embedding"])
                targets.append([labels_map[fname]])

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)

    # 3. 训练过程
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    
    model = TempRegressor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    print(f"🚀 开始训练... 样本数: {len(X)}")
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Loss: {total_loss/len(loader):.4f}")

    # 4. 保存模型
    save_path = project_root / CONFIG["model_save_path"]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✨ 模型已保存至: {save_path}")

if __name__ == "__main__":
    train_model()