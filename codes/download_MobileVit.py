import os
from huggingface_hub import snapshot_download

# --- 第一步：设置镜像环境变量 ---
# 使用国内镜像站加速下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def download_hf_model(model_name, save_dir):
    print(f"正在从镜像站下载 Apple MobileViT 模型: {model_name} ...")
    try:
        # --- 第二步：使用 snapshot_download 下载完整模型 ---
        path = snapshot_download(
            repo_id=model_name,
            local_dir=save_dir,
            local_dir_use_symlinks=False, # 直接下载文件到文件夹，方便后续移动和使用
            resume_download=True,         # 支持断点续传
            token=None                    # MobileViT 是公开模型，不需要 token
        )
        print(f"\n下载成功！模型已保存至: {path}")
    except Exception as e:
        print(f"下载失败，错误信息: {e}")

if __name__ == "__main__":
    # 指定 Apple 的 MobileViT-small 模型 ID
    MODEL_ID = "apple/mobilevit-small"
    # 指定本地保存路径
    LOCAL_SAVE_PATH = "./mobilevit-small" 
    
    download_hf_model(MODEL_ID, LOCAL_SAVE_PATH)