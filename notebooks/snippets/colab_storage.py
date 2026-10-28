# ============================================================
# Colab 儲存工具：Google Drive（隨身行李箱）+ Hugging Face Hub（保險箱）
# 用法：每本筆記本最上面貼「段 A、段 B」，需要時再貼段 C、D；
#       收工前貼段 E。已經 code-review 審查。
# ============================================================

# ===== 段 A｜連接 Google Drive，建立專案資料夾 =====
from google.colab import drive
from pathlib import Path

drive.mount("/content/drive")   # 第一次會跳出授權視窗，選你的 Google 帳號

PROJECT_DIR = Path("/content/drive/MyDrive/grid-dispatch-assistant")
DIRS = {
    "checkpoints": PROJECT_DIR / "checkpoints",   # 訓練中途存檔（斷線可接續）
    "adapters":    PROJECT_DIR / "adapters",      # 訓練完成的 LoRA 外掛
    "results":     PROJECT_DIR / "results",       # 考試成績、評估報告
    "data":        PROJECT_DIR / "data",          # 合成資料集（只放公開或合成資料！）
}
for p in DIRS.values():
    p.mkdir(parents=True, exist_ok=True)

print(f"✅ Drive 已連接：{PROJECT_DIR}")
print("   實際剩餘空間請看：https://drive.google.com/settings/storage")


# ===== 段 B｜登入 Hugging Face（token 存在 Colab 密鑰，不寫進程式） =====
from google.colab import userdata
from huggingface_hub import login, whoami

try:
    hf_token = userdata.get("HF_TOKEN")
except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
    raise RuntimeError(
        "讀不到 HF_TOKEN！請到左側 🔑「密鑰」：① 新增名稱 HF_TOKEN ② 打開「筆記本存取權」開關"
    ) from None

login(token=hf_token, add_to_git_credential=False)
HF_USER = whoami()["name"]
print(f"✅ Hugging Face 已登入：{HF_USER}")
# 安全提醒：絕對不要 print(hf_token)；分享筆記本前先「編輯 → 清除所有輸出」


# ===== 段 C｜存檔小工具：成績存 Drive、成果推上 Hub =====
from datetime import datetime
from zoneinfo import ZoneInfo
from huggingface_hub import HfApi

def _now():
    """Colab 的時鐘是 UTC，這裡換成台灣時間"""
    return datetime.now(ZoneInfo("Asia/Taipei"))

def save_results(df, name: str) -> Path:
    """把成績表存成 CSV 到 Drive；檔名含台灣時間到秒，不會覆蓋舊的"""
    path = DIRS["results"] / f"{_now():%Y%m%d-%H%M%S}_{name}.csv"
    df.to_csv(path, encoding="utf-8-sig")   # utf-8-sig：用 Excel 打開中文不會亂碼
    print(f"💾 已存到 Drive：{path}")
    return path

def push_to_hub(local_dir, repo_name: str, repo_type: str = "model") -> str:
    """把資料夾推到 Hugging Face 私人 repo。repo_type 可為 'model' 或 'dataset'
    ⚠️ 台電內部資料不可上傳，即使是私人 repo 也算資料離開公司"""
    api = HfApi()
    repo_id = f"{HF_USER}/{repo_name}"
    api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
    # 安全檢查：如果這個 repo 以前就存在而且是公開的，停下來不上傳
    if not api.repo_info(repo_id, repo_type=repo_type).private:
        raise RuntimeError(f"{repo_id} 是公開 repo！為安全起見已停止上傳，請先到 Hub 改成私人")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=f"Upload from Colab {_now():%Y-%m-%d %H:%M}",
        ignore_patterns=["checkpoint-*", "*.pt", "optimizer*", "rng_state*"],  # 中途存檔不上傳
    )
    url = f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"
    print(f"☁️ 已上傳（私人）：{url}")
    return url


# ===== 段 D｜斷線後接續訓練：找出「完整的」最新存檔點 =====
REQUIRED_FILES = ["trainer_state.json", "adapter_model.safetensors"]

def find_resume_point(run_name: str):
    """回傳最新且檔案齊全的 checkpoint；沒有就回傳 None（從頭開始）。
    為什麼要檢查齊全：Colab 存檔存到一半斷線，最新那個可能是壞的，要退回上一個"""
    run_dir = DIRS["checkpoints"] / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(
        (p for p in run_dir.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()),
        key=lambda p: int(p.name.split("-")[-1]),
        reverse=True,   # 從最新的開始檢查
    )
    for ckpt in ckpts:
        missing = [f for f in REQUIRED_FILES if not (ckpt / f).exists()]
        if not missing:
            print(f"🔁 找到完整存檔點，會從這裡接續：{ckpt.name}")
            return str(ckpt)
        print(f"⚠️ {ckpt.name} 不完整（缺 {missing}），跳過")
    print("🆕 沒有可用的存檔點，從頭開始訓練")
    return None

# 之後訓練筆記本的用法（示意，現在不用執行）：
#   RUN_NAME = "A-instruct-lora-v1"
#   config = SFTConfig(
#       output_dir=str(DIRS["checkpoints"] / RUN_NAME),
#       save_steps=200,         # 每個存檔約 0.5 GB，別存太頻繁
#       save_total_limit=2,     # 最多留 2 個（舊的會被刪，但會進 Drive 垃圾桶！）
#       ...
#   )
#   trainer.train(resume_from_checkpoint=find_resume_point(RUN_NAME))


# ===== 段 E｜收工：確保檔案真的寫進 Drive =====
drive.flush_and_unmount()
print("✅ 已同步並中斷 Drive 連線，可以安心關閉 Colab")
print("🧹 記得偶爾到 Drive 清空「垃圾桶」：被刪掉的舊存檔還會佔空間")
