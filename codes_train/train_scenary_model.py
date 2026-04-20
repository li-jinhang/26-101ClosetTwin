import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# ==================== 🛠️ 配置区域 ====================
CONFIG = {
    "data_dir": "data/train",
    "visual_features": "_visual_features.jsonl", # 输入：640维视觉特征
    "text_features": "_text_features.jsonl",     # 目标：384维语义特征
    "model_save_path": "models/scenario_regressor.pth",
    "input_dim": 640,   # MobileViT 输出维度
    "output_dim": 384,  # all-MiniLM-L6-v2 输出维度
    "epochs": 150,
    "lr": 0.0005,
    "batch_size": 16
}
# ====================================================

class ScenarioRegressor(nn.Module):
    """场景语义预测模型：将视觉向量映射为语义向量"""
    def __init__(self, input_dim=640, output_dim=384):
        super(ScenarioRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim) # 输出语义特征向量
        )

    def forward(self, x):
        return self.net(x)

def train_scenario_model():
    # 自动定位项目根目录
    project_root = Path(__file__).resolve().parent.parent
    base_path = project_root / CONFIG["data_dir"]
    
    # 1. 加载视觉特征 (Input X)
    visual_map = {}
    visual_path = base_path / CONFIG["visual_features"]
    if not visual_path.exists():
        print(f"❌ 错误：未找到视觉特征文件 {visual_path}")
        return

    with open(visual_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            visual_map[item["image_filename"]] = item["visual_embedding"]

    # 2. 加载文本语义特征 (Target Y) 并对齐
    features_x = []
    features_y = []
    text_path = base_path / CONFIG["text_features"]
    if not text_path.exists():
        print(f"❌ 错误：未找到文本特征文件 {text_path}")
        return

    with open(text_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            fname = item["image_filename"]
            # 只有当图片同时具有视觉和文本特征时才加入训练集
            if fname in visual_map:
                features_x.append(visual_map[fname])
                features_y.append(item["embedding"])

    X = torch.tensor(features_x, dtype=torch.float32)
    y = torch.tensor(features_y, dtype=torch.float32)

    # 3. 训练准备
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True,drop_last=True)
    
    model = ScenarioRegressor(CONFIG["input_dim"], CONFIG["output_dim"])
    # 使用 MSELoss 计算预测向量与真实向量之间的欧式距离损失
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    print(f"🚀 开始训练场景适配模型... 样本数: {len(X)}")
    model.train()
    for epoch in range(CONFIG["epochs"]):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Loss: {total_loss/len(loader):.6f}")

    # 4. 保存模型
    save_path = project_root / CONFIG["model_save_path"]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✨ 场景预测模型已保存至: {save_path}")

if __name__ == "__main__":
    train_scenario_model()