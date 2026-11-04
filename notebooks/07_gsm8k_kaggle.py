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
import json, subprocess, sys, glob, shutil
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
BASE_SHA = api.model_info(BASE_MODEL).sha      # 原模型也鎖版本，以後才能重現
print(f"✅ 合併模型：{FT_MODEL}（版本 {FT_SHA[:8]}）｜原模型版本 {BASE_SHA[:8]}")

# 公平性檢查：兩個模型套上 chat template 後，題目必須一模一樣
from transformers import AutoTokenizer
m = [{"role": "user", "content": "Question: 1+1?\nAnswer:"}]
tb = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_SHA)
tf = AutoTokenizer.from_pretrained(FT_MODEL, revision=FT_SHA)
assert tb.apply_chat_template(m, add_generation_prompt=True, tokenize=False) == \
       tf.apply_chat_template(m, add_generation_prompt=True, tokenize=False), "兩個模型的 chat template 不同，比較不公平，先停下來"
EOS_ID = tb.eos_token_id
print("✅ 兩個模型的 chat template 相同")

OUT_DIR = Path(f"/kaggle/working/{RUN_NAME}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_REPO = f"{HF_USER}/grid-dispatch-results"
api.create_repo(RESULTS_REPO, repo_type="dataset", private=True, exist_ok=True)
assert api.repo_info(RESULTS_REPO, repo_type="dataset").private, f"{RESULTS_REPO} 不是私人的！請到 HF 網頁改成 Private"
subprocess.run(f"{sys.executable} -m pip freeze > {OUT_DIR}/requirements.txt", shell=True)   # 記下套件版本

# ===== 第 4 段｜評估函式：呼叫 lighteval 指令，回傳分數 =====
def run_gsm8k(tag, model_id, revision="main"):
    """用 lighteval 考 GSM8K；兩個模型用完全相同的設定，比較才公平"""
    out = OUT_DIR / tag
    done = sorted(glob.glob(f"{out}/results/**/results_*.json", recursive=True))
    if done:                                   # 已經考過（例如重跑筆記本）：設定一樣才沿用
        cfg = json.load(open(done[-1], encoding="utf-8"))["config_general"]
        if cfg.get("max_samples") == MAX_SAMPLES and cfg.get("model_config", {}).get("revision") == revision:
            print(f"⏭️ {tag} 已有相同設定的結果，跳過")
            return read_score(done[-1])
        print(f"⚠️ {tag} 舊結果的設定不同（題數或模型版本），重新考")
        shutil.rmtree(out)
    model_args = f"model_name={model_id},revision={revision},dtype={DTYPE},batch_size={BATCH_SIZE}"
    cmd = [sys.executable, "-m", "lighteval", "accelerate", model_args, TASK, "--output-dir", str(out), "--save-details"]
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
base = run_gsm8k("base", BASE_MODEL, revision=BASE_SHA)
print(f"原模型 GSM8K：{base['acc']:.1f}% ± {1.96 * base['stderr']:.1f}")

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

# ===== 第 8 段｜截斷檢查＋看幾題作答（確認分數差距不是「回答長度」造成的） =====
# lighteval 的 GSM8K 每題最多只能寫 256 個 token（寫死在程式裡）。寫太長的回答會被切掉、抽不到答案
# 如果微調讓模型變得比較簡短，兩邊被切的比例就不同 → 分數差距會混進「囉嗦程度」，不只是數學能力
import pandas as pd
cut_rate = {}
for tag in ["base", "finetuned"]:
    f = sorted(glob.glob(f"{OUT_DIR}/{tag}/details/**/*.parquet", recursive=True))
    if not f:
        print(f"{tag}：找不到細節檔，略過")
        continue
    d = pd.read_parquet(f[-1])
    try:
        cut = sum(EOS_ID not in list(r["output_tokens"][0]) for r in d["model_response"])
        cut_rate[tag] = cut / len(d) * 100
        print(f"{tag}：被 256 token 截斷 {cut}/{len(d)} 題（{cut_rate[tag]:.0f}%）")
    except Exception as e:
        print(f"{tag}：細節檔格式跟預期不同（{e}），欄位：{list(d.columns)}")
    if tag == "finetuned":
        for _, row in d.head(2).iterrows():
            print("-" * 60)
            print(str(row.to_dict())[:1200])
if len(cut_rate) == 2 and abs(cut_rate["base"] - cut_rate["finetuned"]) > 5:
    print("⚠️ 兩邊截斷比例差超過 5 個百分點：報告分數時要一併說明")

# ===== 第 9 段｜存成績、推到 HF 私人 repo =====
summary = {
    "run_name": RUN_NAME, "task": TASK, "max_samples": MAX_SAMPLES, "batch_size": BATCH_SIZE, "dtype": DTYPE,
    "base": {"model": BASE_MODEL, "revision": BASE_SHA, **{k: v for k, v in base.items() if k != "file"}},
    "truncated_pct_at_256": {k: round(v, 1) for k, v in cut_rate.items()},
    "finetuned": {"model": FT_MODEL, "revision": FT_SHA, **{k: v for k, v in ft.items() if k != "file"}},
    "diff_points": round(diff, 2),
    "versions": {"lighteval": lighteval.__version__, "transformers": transformers.__version__, "torch": torch.__version__},
    "gpu": torch.cuda.get_device_name(0),
    "command": f'lighteval accelerate "model_name=<model>,revision=<sha>,dtype={DTYPE},batch_size={BATCH_SIZE}" "{TASK}" --output-dir <dir> --save-details'
               + (f" --max-samples {MAX_SAMPLES}" if MAX_SAMPLES else ""),
}
(OUT_DIR / f"{RUN_NAME}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
try:
    api.upload_folder(folder_path=str(OUT_DIR), repo_id=RESULTS_REPO, repo_type="dataset",
                      path_in_repo=RUN_NAME, commit_message=f"{RUN_NAME} results")
except Exception as e:   # 細節檔檔名含「|」，萬一 HF 不接受，就只傳成績、不傳細節檔
    print(f"⚠️ 完整上傳失敗（{e}），改成不含細節檔再傳一次")
    api.upload_folder(folder_path=str(OUT_DIR), repo_id=RESULTS_REPO, repo_type="dataset",
                      path_in_repo=RUN_NAME, commit_message=f"{RUN_NAME} results", ignore_patterns=["*.parquet"])
print(f"☁️ 結果已存到（私人）：https://huggingface.co/datasets/{RESULTS_REPO}/tree/main/{RUN_NAME}")
if MAX_SAMPLES:
    print("🧪 試跑結束。確認沒問題後，把 MAX_SAMPLES 改成 None，Restart 後重跑全部")
