from google import genai
import os

# 初始化客户端
client = genai.Client(api_key="xxxxxxxxxxxxxxxxxxx")

print("--- 我可以使用的模型列表 ---")

# 使用新版 SDK 列出模型
for model in client.models.list():
    # 注意：这里从 supported_methods 改为了 supported_actions
    if model.supported_actions and 'generateContent' in model.supported_actions:
        print(f"模型 ID: {model.name}")
        print(f"描述: {model.description}")
        print("-" * 30)