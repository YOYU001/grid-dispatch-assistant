# ============================================================
# 筆記本 02（Kaggle 版）｜把合成資料集上傳到 Hugging Face（私人）
# 不需要 GPU：Settings → Accelerator 選 None（不扣 GPU 時數）
# 事前準備：
#   - Settings → Internet 打開
#   - Add-ons → Secrets 新增 HF_TOKEN（要 Write 權限）並打勾
#   - 右側 Add Input → Upload：上傳 grid-dispatch-data.zip（Kaggle 會自動解壓縮），設成 Private
# ============================================================

# ===== 第 1 段｜安裝套件 =====
# !pip install -q -U datasets huggingface_hub

# ===== 第 2 段｜登入 Hugging Face，找到上傳的資料夾 =====
import glob
from pathlib import Path
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, whoami

try:
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    raise RuntimeError("讀不到 HF_TOKEN！請到 Add-ons → Secrets 新增 HF_TOKEN 並打勾") from None
login(token=hf_token, add_to_git_credential=False)   # 這裡出錯＝token 打錯或過期，請到 HF 重新產生
HF_USER = whoami()["name"]
print(f"✅ Hugging Face 已登入：{HF_USER}")

found = glob.glob("/kaggle/input/**/train.jsonl", recursive=True)
assert found, "找不到 train.jsonl！請在右側 Add Input → Upload 上傳 grid-dispatch-data.zip"
if len(found) > 1:
    print(f"⚠️ 找到 {len(found)} 份資料，使用第一份；如果不對，請到右側 Input 移除舊的上傳：{found}")
DATA_DIR = Path(found[0]).parent
print(f"✅ 資料夾：{DATA_DIR}")

# ===== 第 3 段｜確認檔案都在，先在本地看一眼資料長怎樣 =====
from datasets import load_dataset

SPLIT_FILES = ["train.jsonl", "validation.jsonl", "test.jsonl", "test_unseen_phrasing.jsonl"]
missing = [f for f in SPLIT_FILES + ["README.md"] if not (DATA_DIR / f).exists()]
if missing:
    raise FileNotFoundError(f"找不到：{missing}，請確認上傳的是 grid-dispatch-data.zip")

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
    commit_message="Upload synthetic dataset (from Kaggle)",
)
print(f"☁️ 已上傳（私人）：https://huggingface.co/datasets/{REPO_ID}")

# ===== 第 5 段｜從 Hub 重新下載，確認上傳成功 =====
hub_ds = load_dataset(REPO_ID)
print(hub_ds)
assert {k: len(v) for k, v in hub_ds.items()} == {k: len(v) for k, v in ds.items()}, "題數對不上！"
print("✅ Hub 上的資料與上傳的一致")
