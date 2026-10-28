# ============================================================
# 筆記本 02｜把合成資料集上傳到 Hugging Face（私人）
# 不需要 GPU：Colab「執行階段類型」選 CPU 即可
# 事前準備：把 grid-dispatch-data 資料夾放到
#           Google Drive / MyDrive / grid-dispatch-assistant / data /
# ============================================================

# ===== 第 1 段｜安裝套件 =====
# !pip install -q -U datasets huggingface_hub

# ===== 第 2 段｜連接 Drive、登入 Hugging Face =====
from google.colab import drive, userdata
from huggingface_hub import login, whoami
from pathlib import Path

drive.mount("/content/drive")
DATA_DIR = Path("/content/drive/MyDrive/grid-dispatch-assistant/data/grid-dispatch-data")

try:
    login(token=userdata.get("HF_TOKEN"), add_to_git_credential=False)
except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
    raise RuntimeError("讀不到 HF_TOKEN！請到左側 🔑 新增 HF_TOKEN 並打開「筆記本存取權」") from None
HF_USER = whoami()["name"]
print(f"✅ Hugging Face 已登入：{HF_USER}")

# ===== 第 3 段｜確認檔案都在 Drive，先在本地看一眼資料長怎樣 =====
from datasets import load_dataset

SPLIT_FILES = ["train.jsonl", "validation.jsonl", "test.jsonl", "test_unseen_phrasing.jsonl"]
missing = [f for f in SPLIT_FILES + ["README.md"] if not (DATA_DIR / f).exists()]
if missing:
    raise FileNotFoundError(f"Drive 裡找不到：{missing}，請確認資料夾位置是 {DATA_DIR}")

# 用 README 裡寫好的設定載入：這就是別人之後從 Hub 載入時會看到的樣子
ds = load_dataset(str(DATA_DIR))
print(ds)                                   # 各切分的題數與欄位

ex = ds["train"][0]
print("\n----- 第 1 題 -----")
print("題型：", ex["category"])
print("【使用者輸入】\n" + ex["messages"][1]["content"])
print("【標準答案】\n" + ex["messages"][2]["content"])
print("【規則軌跡】", ex["trace"])

# ===== 第 4 段｜上傳到 Hugging Face（私人 dataset） =====
from huggingface_hub import HfApi

REPO_ID = f"{HF_USER}/grid-dispatch-notes-zh"
api = HfApi()
api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
if not api.repo_info(REPO_ID, repo_type="dataset").private:     # 防呆：公開的就停
    raise RuntimeError(f"{REPO_ID} 是公開的！為安全起見停止上傳，請先到 Hub 改成私人")

api.upload_folder(
    folder_path=str(DATA_DIR),
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns=SPLIT_FILES + ["README.md"],   # 只上傳這 5 個檔；真人考卷不上傳
    commit_message="Upload synthetic dataset v1 (seed 42)",
)
print(f"☁️ 已上傳（私人）：https://huggingface.co/datasets/{REPO_ID}")

# ===== 第 5 段｜從 Hub 重新下載，確認上傳成功 =====
hub_ds = load_dataset(REPO_ID)
print(hub_ds)
assert {k: len(v) for k, v in hub_ds.items()} == {k: len(v) for k, v in ds.items()}, "題數對不上！"
print("✅ Hub 上的資料與 Drive 上的一致")
