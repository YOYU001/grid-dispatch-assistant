# ============================================================
# 筆記本 07（Kaggle 版）｜GSM8K 退化檢查：原模型 vs 微調後模型
# 目的：確認學了「調度」之後，原本的數學能力沒有明顯變差（災難性遺忘檢查）
# 判斷標準：微調後分數下降超過 5 個百分點 → 標紅、要處理
# 工具：lighteval 0.13.0（transformers 後端；T4 上最穩，不用裝 vLLM）
# 事前準備：
#   - 先跑完筆記本 06（合併模型要已經在 HF 上）
#   - 這本要用「新的」Kaggle 筆記本，不要接在 06 後面跑（套件版本不同）
#   - Settings → Accelerator 選 GPU T4；Internet 打開；Secrets 加 HF_TOKEN 並打勾
#   - 第 1 段 pip 那行要「去掉開頭的 #」才會安裝
# 先用 MAX_SAMPLES = 20 試跑（約 5～10 分鐘），確認沒問題再改 None 考全部 1,319 題
# ============================================================

# ===== 第 1 段｜安裝 lighteval（官方 PyPI 最新版 0.13.0） =====
# !pip install -q "lighteval==0.13.0"
# 注意：lighteval 0.13 的任務名稱格式改成「gsm8k|0」（課程範例的 "lighteval|gsm8k|0|0" 在新版會報錯）

# ===== 第 2 段｜主要設定（只需要改這裡） =====
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # 只用第 1 張 T4

BASE_MODEL  = "Qwen/Qwen3-4B-Instruct-2507"
MERGED_NAME = "grid-dispatch-A-instruct-merged-v1"   # 筆記本 06 推上去的合併模型
TASK        = "gsm8k|0"      # GSM8K、0-shot（不給範例題）
BATCH_SIZE  = 16             # 記憶體不足（CUDA out of memory）就改 8
MAX_SAMPLES = 20             # 試跑只考前 20 題；確認沒問題後改成 None 考全部
RUN_NAME    = "07_gsm8k" + ("-trial" if MAX_SAMPLES else "")

# ===== 第 3 段｜登入、確認 GPU 與套件版本 =====
import json, subprocess, sys, glob
from pathlib import Path
import torch, transformers, lighteval
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, whoami, HfApi

token = UserSecretsClient().get_secret("HF_TOKEN")
login(token=token, add_to_git_credential=False)
os.environ["HF_TOKEN"] = token        # lighteval 是另外開的程式，要靠環境變數讀私人模型
HF_USER = whoami()["name"]
api = HfApi()
FT_MODEL = f"{HF_USER}/{MERGED_NAME}"
MODELS = {"base": BASE_MODEL, "finetuned": FT_MODEL}

assert torch.cuda.is_available(), "沒有 GPU！請到 Settings → Accelerator 選 GPU T4"
DTYPE = "bfloat16" if torch.cuda.get_device_capability(0)[0] >= 8 else "float16"
print(f"GPU：{torch.cuda.get_device_name(0)}｜精度：{DTYPE}")
print(f"套件：lighteval {lighteval.__version__}｜transformers {transformers.__version__}｜torch {torch.__version__}")
assert api.file_exists(FT_MODEL, "config.json"), f"找不到合併模型 {FT_MODEL}，請先跑筆記本 06"
FT_SHA = api.model_info(FT_MODEL).sha
print(f"✅ 合併模型：{FT_MODEL}（版本 {FT_SHA[:8]}）")

OUT_DIR = Path(f"/kaggle/working/{RUN_NAME}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_REPO = f"{HF_USER}/grid-dispatch-results"

# ===== 第 4 段｜評估函式：呼叫 lighteval 指令，回傳分數 =====
def run_gsm8k(tag, model_id, revision="main"):
    """用 lighteval 考 GSM8K；兩個模型用完全相同的設定，比較才公平"""
    out = OUT_DIR / tag
    done = glob.glob(f"{out}/results/**/results_*.json", recursive=True)
    if done:                                   # 已經考過（例如重跑筆記本）就直接讀結果
        print(f"⏭️ {tag} 已有結果，跳過")
        return read_score(done[-1])
    model_args = f"model_name={model_id},revision={revision},dtype={DTYPE},batch_size={BATCH_SIZE}"
    cmd = ["lighteval", "accelerate", model_args, TASK, "--output-dir", str(out), "--save-details"]
    if MAX_SAMPLES:
        cmd += ["--max-samples", str(MAX_SAMPLES)]
    print("執行：", " ".join(cmd))
    subprocess.run(cmd, check=True)            # 失敗會直接停下，錯誤訊息印在上方
    files = sorted(glob.glob(f"{out}/results/**/results_*.json", recursive=True))
    assert files, f"{tag} 跑完但找不到結果檔"
    return read_score(files[-1])

def read_score(path):
    """從結果檔找出 GSM8K 的 extractive_match（答對率）"""
    res = json.load(open(path, encoding="utf-8"))["results"]
    key = next(k for k in res if "gsm8k" in k)   # 新版是 "gsm8k|0"；用模糊比對避免版本差異
    r = res[key]
    return {"task_key": key, "acc": r["extractive_match"] * 100,
            "stderr": r.get("extractive_match_stderr", 0) * 100, "file": path}

# ===== 第 5 段｜考原模型（訓練前） =====
base = run_gsm8k("base", BASE_MODEL)
print(f"原模型 GSM8K：{base['acc']:.1f}% ± {1.96 * base['stderr']:.1f}")
torch.cuda.empty_cache()

# ===== 第 6 段｜考微調後模型（鎖定 06 上傳的版本） =====
ft = run_gsm8k("finetuned", FT_MODEL, revision=FT_SHA)
print(f"微調後 GSM8K：{ft['acc']:.1f}% ± {1.96 * ft['stderr']:.1f}")

# ===== 第 7 段｜比較：有沒有忘掉數學？ =====
diff = ft["acc"] - base["acc"]
n = MAX_SAMPLES or 1319
print(f"\n{'':10s}{'GSM8K 答對率':>14s}")
print(f"{'原模型':10s}{base['acc']:>13.1f}%")
print(f"{'微調後':10s}{ft['acc']:>13.1f}%")
print(f"{'差距':10s}{diff:>+13.1f} 個百分點（{n} 題）")
if MAX_SAMPLES:
    print("🧪 試跑題數太少，差距不可信；只用來確認流程能跑")
elif diff < -5:
    print("🔴 下降超過 5 個百分點：有災難性遺忘，需要處理（例如減少 epoch、混入一般資料）")
else:
    print("✅ 沒有明顯退化（下降不到 5 個百分點）")

# ===== 第 8 段｜看幾題微調後模型的作答（確認不是格式問題造成的分數） =====
import pandas as pd
det = sorted(glob.glob(f"{OUT_DIR}/finetuned/details/**/*.parquet", recursive=True))
if det:
    d = pd.read_parquet(det[-1])
    print(f"細節欄位：{list(d.columns)[:12]}")
    for _, row in d.head(2).iterrows():
        print("-" * 60)
        print(str(row.to_dict())[:1200])
else:
    print("找不到細節檔（--save-details），略過")

# ===== 第 9 段｜存成績、推到 HF 私人 repo =====
summary = {
    "run_name": RUN_NAME, "task": TASK, "max_samples": MAX_SAMPLES, "batch_size": BATCH_SIZE, "dtype": DTYPE,
    "base": {"model": BASE_MODEL, **{k: v for k, v in base.items() if k != "file"}},
    "finetuned": {"model": FT_MODEL, "revision": FT_SHA, **{k: v for k, v in ft.items() if k != "file"}},
    "diff_points": round(diff, 2),
    "versions": {"lighteval": lighteval.__version__, "transformers": transformers.__version__, "torch": torch.__version__},
    "gpu": torch.cuda.get_device_name(0),
    "command": f'lighteval accelerate "model_name=<model>,revision=<sha>,dtype={DTYPE},batch_size={BATCH_SIZE}" "{TASK}" --output-dir <dir> --save-details'
               + (f" --max-samples {MAX_SAMPLES}" if MAX_SAMPLES else ""),
}
(OUT_DIR / f"{RUN_NAME}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
api.upload_folder(folder_path=str(OUT_DIR), repo_id=RESULTS_REPO, repo_type="dataset",
                  path_in_repo=RUN_NAME, commit_message=f"{RUN_NAME} results")
print(f"☁️ 結果已存到（私人）：https://huggingface.co/datasets/{RESULTS_REPO}/tree/main/{RUN_NAME}")
if MAX_SAMPLES:
    print("🧪 試跑結束。確認沒問題後，把 MAX_SAMPLES 改成 None，Restart 後重跑全部")
