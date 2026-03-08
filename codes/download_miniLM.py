import os
from huggingface_hub import snapshot_download

# --- 第一步：设置镜像环境变量 ---
# 这会让 huggingface_hub 库优先通过国内镜像站下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def download_hf_model(model_name, save_dir):
    print(f"正在从镜像站下载模型: {model_name} ...")
    try:
        # --- 第二步：使用 snapshot_download 下载完整模型 ---
        path = snapshot_download(
            repo_id=model_name,
            local_dir=save_dir,
            local_dir_use_symlinks=False, # 建议设为 False，直接下载文件到文件夹
            resume_download=True,         # 支持断点续传
            token=None                    # 公开模型不需要 token
        )
        print(f"\n下载成功！模型已保存至: {path}")
    except Exception as e:
        print(f"下载失败，错误信息: {e}")

if __name__ == "__main__":
    MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    LOCAL_SAVE_PATH = "./all-MiniLM-L6-v2" # 你想保存的本地路径
    
    download_hf_model(MODEL_ID, LOCAL_SAVE_PATH)