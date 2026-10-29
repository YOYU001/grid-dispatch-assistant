# ============================================================
# Kaggle 版替換段落（Colab GPU 額度用完時使用）
# 與 Colab 的差異：
#   - 密鑰：Kaggle「Add-ons → Secrets」，用 kaggle_secrets 讀取
#   - 存檔：/kaggle/working（關掉 session 就會清空）→ 最後把結果推到 HF 私人 repo
#   - 輸入檔：右側「Add Input → Upload」，路徑在 /kaggle/input/ 底下
#   - T4 × 2：只用第 1 張卡（device_map={"": 0}），4B 模型一張就放得下
# ============================================================

# ===== Kaggle 第 2 段（取代 Colab 第 2 段）｜登入 Hugging Face、找到真人考卷 =====
import glob
from pathlib import Path
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, whoami

try:
    login(token=UserSecretsClient().get_secret("HF_TOKEN"), add_to_git_credential=False)
except Exception:
    raise RuntimeError("讀不到 HF_TOKEN！請到上方選單 Add-ons → Secrets 新增 HF_TOKEN，並打勾讓這本筆記本使用") from None
HF_USER = whoami()["name"]
print(f"✅ Hugging Face 已登入：{HF_USER}")

RESULTS_DIR = Path("/kaggle/working/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
found = glob.glob("/kaggle/input/**/human_test_v2.jsonl", recursive=True)
assert found, "找不到真人考卷！請在右側 Add Input → Upload 上傳 human_test_v2.jsonl"
HUMAN_TEST = Path(found[0])
print(f"✅ 真人考卷：{HUMAN_TEST}")

# ===== Kaggle 第 5 段（取代 Colab 第 5 段）｜載入模型到第 1 張 GPU =====
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map={"": 0})  # 只用第 1 張 T4
model.eval()
print(f"GPU 記憶體用量：{torch.cuda.memory_allocated(0) / 1e9:.1f} GB")

# ===== Kaggle 第 10 段（取代 Colab 第 10 段）｜把結果推到 Hugging Face 私人 repo =====
from huggingface_hub import HfApi

RESULTS_REPO = f"{HF_USER}/grid-dispatch-results"
api = HfApi()
api.create_repo(RESULTS_REPO, repo_type="dataset", private=True, exist_ok=True)
if not api.repo_info(RESULTS_REPO, repo_type="dataset").private:
    raise RuntimeError(f"{RESULTS_REPO} 是公開的！停止上傳")
api.upload_folder(folder_path=str(RESULTS_DIR), repo_id=RESULTS_REPO, repo_type="dataset",
                  path_in_repo=RUN_NAME, commit_message=f"{RUN_NAME} results")
print(f"☁️ 結果已存到（私人）：https://huggingface.co/datasets/{RESULTS_REPO}/tree/main/{RUN_NAME}")
