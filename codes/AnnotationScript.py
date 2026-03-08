import os
import time
import json
from pathlib import Path
from itertools import cycle
from google import genai 
from google.genai import types

# ================= 配置区域 =================
KEY_FILE = "api_keys.txt"
IMAGE_FOLDER = "data\\clothing-dataset-small-master\\train\\shirt"
MODEL_NAME = "gemini-2.5-flash" 

PROMPT = """
Role：你是一位资深的时尚造型师和服装功能分析专家。
Task：分析上传图片并完成 JSON 标注：
1.使用场景：提供 3 个最适合的场景。
2.适用温度：给出摄氏度范围并简述理由。

Output Format：
{
  "clothing_type": "简要描述衣物名称/类型",
  "usage_scenarios": ["场景1", "场景2", "场景3"],
  "temperature_range": {
    "range": "XX°C - XX°C",
    "reason": "基于材质和款式的简要解释"
  }
}
"""

# 代理配置（根据你的环境保留或修改）
proxy_url = 'http://127.0.0.1:7897'
os.environ['HTTP_PROXY'] = proxy_url
os.environ['HTTPS_PROXY'] = proxy_url
os.environ['http_proxy'] = proxy_url
os.environ['https_proxy'] = proxy_url
# ===========================================

def load_keys(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 未找到 Key 文件: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        keys = [line.strip() for line in f if line.strip()]
    if not keys:
        raise ValueError("❌ Key 文件为空！")
    return keys

def batch_label_with_gemini(folder_path):
    # 1. 初始化 Key 池
    keys = load_keys(KEY_FILE)
    num_keys = len(keys)
    key_pool = cycle(keys)
    
    # 2. 初始化第一个 Client
    current_key = next(key_pool)
    client = genai.Client(api_key=current_key)
    print(f"🚀 初始化完成，加载 {num_keys} 个 Key。当前 Key: ...{current_key[-4:]}")

    # 3. 筛选未处理的图片
    path = Path(folder_path)
    if not path.exists():
        print(f"❌ 文件夹路径不存在: {folder_path}")
        return

    all_images = [f for f in path.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}]
    image_files = [f for f in all_images if not f.with_suffix('.txt').exists()]
    
    print(f"📊 统计: 总计 {len(all_images)} 张，需处理 {len(image_files)} 张。")

    # 4. 开始遍历处理
    for file_path in image_files:
        txt_path = file_path.with_suffix('.txt')
        success = False
        retry_count = 0
        
        # 针对单张图片的重试/换 Key 逻辑
        # 允许重试次数为 Key 总数的 2 倍，确保每个 Key 都有机会在冷却后再次尝试
        while not success and retry_count < (num_keys * 2):
            try:
                print(f"🔍 正在处理: {file_path.name} (Key: ...{current_key[-4:]})")
                
                with open(file_path, "rb") as f:
                    image_bytes = f.read()
                
                # 发起请求
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[
                        PROMPT,
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )

                # 写入结果
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                print(f"✅ 已保存: {txt_path.name}")
                success = True
                # 成功后短暂停顿，保护频率
                time.sleep(0.5) 

            except Exception as e:
                error_str = str(e)
                if "429" in error_str:
                    retry_count += 1
                    print(f"⚠️ 频率受限 (429)。尝试切换第 {retry_count} 次 Key...")
                    
                    # 轮换到下一个 Key 并重建 Client
                    current_key = next(key_pool)
                    client = genai.Client(api_key=current_key)
                    
                    # 如果所有 Key 已经轮了一遍，额外增加冷却时间
                    if retry_count % num_keys == 0:
                        print("💤 所有 Key 均已尝试一遍，进入全局冷却 30s...")
                        time.sleep(30)
                    else:
                        time.sleep(1) # 普通切换等待
                else:
                    print(f"❌ 识别失败 {file_path.name}: {error_str}")
                    # 非频率问题（如格式、网络断开等），跳过此文件处理下一个
                    break 

        if not success:
            print(f"⏭️ 无法处理文件 {file_path.name}，已跳过。")

    print("\n✨ 所有任务处理流程结束！")

if __name__ == "__main__":
    batch_label_with_gemini(IMAGE_FOLDER)