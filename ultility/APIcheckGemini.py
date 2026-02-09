import os
from google import genai

# 1. 初始化客户端
# 建议将 API Key 设置为环境变量 GEMINI_API_KEY
# 或者直接在代码中传入: client = genai.Client(api_key="你的_API_KEY")
client = genai.Client(api_key="xxxxxxxxxxxxxxxxxxx")

def test_gemini_connection():
    try:
        print("正在请求 Gemini...")
        
        # 2. 调用模型生成内容 (目前常用 gemini-2.0-flash 或 gemini-1.5-flash)
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents="你好！请用一句话证明你已经连接成功了。"
        )
        
        # 3. 打印结果
        print("-" * 20)
        print(f"Gemini 回复: {response.text}")
        print("-" * 20)
        print("测试成功！")
        
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    test_gemini_connection()