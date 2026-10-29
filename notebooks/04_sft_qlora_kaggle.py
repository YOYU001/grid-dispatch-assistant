# ============================================================
# 筆記本 04｜SFT + QLoRA 訓練（Kaggle 背景執行版）
# 模型：Qwen/Qwen3-4B-Instruct-2507 ｜ 資料：yoyu001/grid-dispatch-notes-zh
# 硬體：Kaggle GPU T4（fp16）
#
# 事前設定：Settings → Accelerator 選 GPU T4 x2、Settings → Internet 打開（需手機驗證）、
#           Add-ons → Secrets 加 HF_TOKEN 並打勾；第 1 段 pip 那行要「去掉開頭的 #」才會安裝
# 使用流程：
#   ① MODE = "trial"：互動模式 Run All → 跑 10 步，看 loss 有沒有下降、每步幾秒、記憶體夠不夠
#   ② MODE = "trial_full"：互動模式 Run All → 跑 20 步，順便測「存檔、推上 HF、評估」整條流程
#   ③ MODE = "full"：右上角 Save Version → Save & Run All（背景執行，可關瀏覽器）
#   ④ 跑完到 Hugging Face 看 LoRA 外掛與 loss 圖
# 斷線／失敗：MODE 維持 "full"，再 Save & Run All 一次，會從 HF 上最後的 checkpoint 接續
# ============================================================

# ===== 第 1 段｜安裝套件 =====
# !pip install -q -U "transformers>=4.56,<6" "trl>=0.20,<2" "peft>=0.17" "bitsandbytes>=0.46" accelerate datasets huggingface_hub matplotlib

# ===== 第 2 段｜主要設定（只需要改這裡） =====
MODE = "trial"                   # "trial"＝試跑 10 步｜"trial_full"＝測整條流程 20 步｜"full"＝正式訓練

MODEL_ID   = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_ID = "yoyu001/grid-dispatch-notes-zh"
RUN_NAME   = "A-instruct-qlora-v1"           # A 版本＝無思考；之後 C、B 版本換名字
# ⚠️ 改了任何超參數想重新訓練，一定要換 RUN_NAME，否則會接續舊的 checkpoint（舊設定）

# --- LoRA 外掛 ---
LORA_R       = 16      # 外掛大小：越大學得越多、檔案越大（常見 8～64）
LORA_ALPHA   = 32      # 外掛音量：通常設成 r 的 2 倍
LORA_DROPOUT = 0.05    # 訓練時隨機遮掉 5%，防止死背

# --- 訓練節奏 ---
EPOCHS        = 2      # 整本題本讀幾遍
LR            = 1e-4   # 學習率：TRL 建議 adapter 用 1e-4；loss 爆掉就調小（5e-5）
BATCH_SIZE    = 4      # 一次看幾題（T4 記憶體限制）
GRAD_ACCUM    = 4      # 累積幾次才更新一次 → 有效批次 = 4 × 4 = 16 題
MAX_LENGTH    = 1024   # 每題最多幾個 token（第 4 段會印出實際最長多少）
EVAL_SUBSET   = 200    # 驗證集只取 200 題算 eval loss，省時間
SAVE_EVERY    = 100    # 每幾步存一次 checkpoint 並推到 HF（斷線可接續）；全部約 500 步

TRIAL = MODE != "full"
if MODE == "trial_full":
    RUN_NAME += "-pipeline-test"   # 測試流程用的臨時 repo，不會汙染正式訓練
HUB_MODEL_ID = None    # 第 3 段登入後自動設成「你的帳號/grid-dispatch-{RUN_NAME}」
                       # 想從頭重新訓練（不接續）→ 換一個 RUN_NAME 即可

# ===== 第 3 段｜登入 Hugging Face、確認 GPU =====
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # Kaggle 是 T4 x2：只讓程式看到第 1 張，避免 Trainer 自動用兩張卡
import torch, transformers, trl, peft
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, whoami

try:
    login(token=UserSecretsClient().get_secret("HF_TOKEN"), add_to_git_credential=False)
except Exception:
    raise RuntimeError("讀不到 HF_TOKEN！請到 Add-ons → Secrets 新增 HF_TOKEN 並打勾") from None
HF_USER = whoami()["name"]
HUB_MODEL_ID = f"{HF_USER}/grid-dispatch-{RUN_NAME}"
print(f"✅ Hugging Face：{HF_USER}｜模型會存到（私人）：{HUB_MODEL_ID}")

assert torch.cuda.is_available(), "沒有 GPU！請到 Settings → Accelerator 選 GPU T4 x2"
CC = torch.cuda.get_device_capability(0)
USE_BF16 = CC[0] >= 8                       # T4 是 7.5 → False → 用 fp16
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
print(f"GPU：{torch.cuda.get_device_name(0)}（compute {CC[0]}.{CC[1]}）｜計算精度：{COMPUTE_DTYPE}")
print({"trial": "🧪 試跑 10 步", "trial_full": "🧪 測整條流程 20 步", "full": "🚀 正式訓練"}[MODE])
print(f"套件版本：transformers {transformers.__version__}｜trl {trl.__version__}｜peft {peft.__version__}")

# ===== 第 4 段｜載入資料，轉成「題目／答案」格式 =====
# prompt＝system＋user（題目），completion＝assistant（答案）
# 這樣 TRL 只會對「答案」計算 loss：模型專心學怎麼回答，不用背題目
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# Qwen 的 pad token 若被設成別的特殊符號，4-bit 訓練可能出現 NaN（已知問題）
assert tokenizer.pad_token == "<|endoftext|>", f"pad token 不對：{tokenizer.pad_token}"
raw = load_dataset(DATASET_ID)

def to_prompt_completion(ex):
    return {"prompt": ex["messages"][:2], "completion": [ex["messages"][2]]}

keep = ["prompt", "completion"]
train_ds = raw["train"].map(to_prompt_completion, remove_columns=[c for c in raw["train"].column_names if c not in keep])
eval_ds = raw["validation"].shuffle(seed=42).select(range(EVAL_SUBSET))
eval_ds = eval_ds.map(to_prompt_completion, remove_columns=[c for c in eval_ds.column_names if c not in keep])

# 看看每題有多長，確認 MAX_LENGTH 夠用
def n_tokens(ex):
    text = tokenizer.apply_chat_template(ex["prompt"] + ex["completion"], tokenize=False)
    return len(tokenizer(text)["input_ids"])
lens = sorted(n_tokens(ex) for ex in train_ds) + [n_tokens(ex) for ex in eval_ds]   # 全部檢查，約十幾秒
lens.sort()
print(f"訓練題數：{len(train_ds)}｜驗證（抽樣）：{len(eval_ds)}")
print(f"每題長度：中位數 {lens[len(lens)//2]}、最長 {lens[-1]} tokens（MAX_LENGTH={MAX_LENGTH}）")
assert lens[-1] < MAX_LENGTH, "有題目超過 MAX_LENGTH，答案結尾會被截斷！請調大 MAX_LENGTH"

# ===== 第 5 段｜載入 4-bit 模型，準備 QLoRA =====
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import LoraConfig, prepare_model_for_kbit_training

set_seed(42)   # 固定隨機種子：LoRA 初始值、資料順序都可重現

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # 專為模型權重設計的 4-bit 格式
    bnb_4bit_compute_dtype=COMPUTE_DTYPE, # 計算時還原成 fp16
    bnb_4bit_use_double_quant=True,       # 再壓一次，多省一點記憶體
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb, device_map={"": 0},   # 只用第 1 張 T4
    dtype=COMPUTE_DTYPE, attn_implementation="sdpa",
)
model.config.use_cache = False            # 訓練時關掉快取（搭配 gradient checkpointing）
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

peft_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules="all-linear",          # 外掛裝在所有線性層，效果最好
    bias="none", task_type="CAUSAL_LM",
)
print(f"GPU 記憶體用量：{torch.cuda.memory_allocated(0) / 1e9:.1f} GB")

# ===== 第 6 段｜訓練設定 =====
from trl import SFTConfig

OUT_DIR = f"/kaggle/working/{RUN_NAME}"
TOTAL_STEPS = EPOCHS * -(-len(train_ds) // (BATCH_SIZE * GRAD_ACCUM))   # 無條件進位
MAX_STEPS = {"trial": 10, "trial_full": 20, "full": -1}[MODE]
STEP_EVERY = 10 if MODE == "trial_full" else SAVE_EVERY
config = SFTConfig(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    max_steps=MAX_STEPS,                  # 試跑只跑 10／20 步；-1＝照 EPOCHS 跑完
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",           # 學習率先升後慢慢降
    warmup_steps=max(1, int(0.05 * TOTAL_STEPS)),   # 前 5% 的步數慢慢加速
    max_grad_norm=0.5,                    # 梯度太大就截斷，防止 fp16 爆掉
    optim="paged_adamw_8bit",             # 省記憶體的優化器
    fp16=not USE_BF16, bf16=USE_BF16,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=MAX_LENGTH,
    packing=False,
    completion_only_loss=True,            # 只對答案算 loss
    logging_steps=1 if TRIAL else 10,
    eval_strategy="no" if MODE == "trial" else "steps",
    eval_steps=STEP_EVERY,
    save_strategy="no" if MODE == "trial" else "steps",
    save_steps=STEP_EVERY,
    save_total_limit=2,
    push_to_hub=MODE != "trial",          # trial_full 推到臨時 repo，full 推到正式 repo
    hub_model_id=HUB_MODEL_ID,
    hub_private_repo=True,
    hub_strategy="checkpoint",            # 每次存檔都把最新 checkpoint 推到 HF（可接續）
    report_to="none",
    seed=42,
)

# ===== 第 7 段｜建立 Trainer；如果 HF 上有之前的 checkpoint 就接著訓練 =====
from trl import SFTTrainer
from huggingface_hub import snapshot_download, HfApi

trainer = SFTTrainer(
    model=model, args=config,
    train_dataset=train_ds, eval_dataset=None if MODE == "trial" else eval_ds,
    processing_class=tokenizer, peft_config=peft_config,
)
trainer.model.print_trainable_parameters()   # 只有約 1% 的參數要訓練

resume = None
if MODE != "trial":
    api = HfApi()
    if api.repo_exists(HUB_MODEL_ID):
        files = api.list_repo_files(HUB_MODEL_ID)
        if any(f.startswith("last-checkpoint/") for f in files):
            local = snapshot_download(HUB_MODEL_ID, allow_patterns=["last-checkpoint/*"], local_dir="/kaggle/working/resume")
            resume = os.path.join(local, "last-checkpoint")
print(("🔁 從 HF 的 checkpoint 接續：" + resume) if resume else "🆕 從頭開始訓練")

# ===== 第 8 段｜開始訓練 =====
import time
t0 = time.time()
trainer.train(resume_from_checkpoint=resume)
mins = (time.time() - t0) / 60
steps = trainer.state.global_step
print(f"✅ 訓練完成：{steps} 步，花 {mins:.1f} 分鐘（平均每步 {mins * 60 / max(steps, 1):.1f} 秒）")

peak_gb = torch.cuda.max_memory_allocated(0) / 1e9
print(f"💾 GPU 記憶體最高用到 {peak_gb:.1f} GB（T4 約 15 GB）" + ("　⚠️ 超過 12 GB，建議 BATCH_SIZE=2、GRAD_ACCUM=8" if peak_gb > 12 else ""))
if TRIAL:
    est_h = TOTAL_STEPS * (mins * 60 / max(steps, 1)) / 3600
    print(f"📏 正式訓練約 {TOTAL_STEPS} 步，預估 {est_h:.1f} 小時（另加 eval 與上傳，約多 10%）")

# ===== 第 9 段｜存 LoRA 外掛、推到 Hugging Face（先做最重要的事） =====
if MODE != "trial":
    # trainer.push_to_hub 會先等背景的 checkpoint 推送完成，再推最終外掛，不會互相搶
    trainer.push_to_hub(commit_message=f"{RUN_NAME}: final adapter")   # 外掛＋tokenizer＋model card

    # 把這次訓練的所有設定存成 run_config.json，以後才知道這個外掛是怎麼練出來的
    import json, bitsandbytes
    run_config = {
        "run_name": RUN_NAME, "base_model": MODEL_ID,
        "dataset": DATASET_ID, "dataset_sha": HfApi().dataset_info(DATASET_ID).sha,   # 鎖定資料版本
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "target_modules": "all-linear"},
        "train": {"epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
                  "max_length": MAX_LENGTH, "warmup_steps": config.warmup_steps, "max_grad_norm": config.max_grad_norm,
                  "optim": config.optim, "seed": 42, "fp16": config.fp16, "steps_done": trainer.state.global_step},
        "hardware": torch.cuda.get_device_name(0), "train_minutes": round(mins, 1),
        "versions": {"transformers": transformers.__version__, "trl": trl.__version__, "peft": peft.__version__,
                     "bitsandbytes": bitsandbytes.__version__, "torch": torch.__version__},
    }
    HfApi().upload_file(path_or_fileobj=json.dumps(run_config, ensure_ascii=False, indent=2).encode(),
                        path_in_repo="training_logs/run_config.json", repo_id=HUB_MODEL_ID,
                        commit_message=f"{RUN_NAME}: run config")
    print(f"☁️ LoRA 外掛與設定已上傳（私人）：https://huggingface.co/{HUB_MODEL_ID}")
else:
    print("🧪 試跑模式不上傳")

# ===== 第 10 段｜畫 loss 曲線、存訓練紀錄（出錯也不影響已上傳的外掛） =====
try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = pd.DataFrame(trainer.state.log_history)
    hist.to_csv(f"{OUT_DIR}_log_history.csv", index=False)
    train_log = hist.dropna(subset=["loss"]) if "loss" in hist else hist.iloc[0:0]
    eval_log = hist.dropna(subset=["eval_loss"]) if "eval_loss" in hist else hist.iloc[0:0]

    plt.figure(figsize=(7, 4))
    plt.plot(train_log["step"], train_log["loss"], label="train loss")
    if len(eval_log):
        plt.plot(eval_log["step"], eval_log["eval_loss"], "o-", label="eval loss")
    plt.xlabel("step"); plt.ylabel("loss"); plt.title(RUN_NAME); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f"{OUT_DIR}_loss.png", dpi=120, bbox_inches="tight")
    print(train_log[["step", "loss"]].tail(5).to_string(index=False))
    if len(eval_log):
        print(eval_log[["step", "eval_loss"]].to_string(index=False))
    import math
    raw_losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    if any(math.isnan(x) or x == 0 for x in raw_losses):
        print("⚠️ loss 出現 NaN 或 0：可能是 fp16 溢位，請把 LR 調成 5e-5、換新的 RUN_NAME 再試")
    if MODE != "trial":
        api = HfApi()
        for f in [f"{OUT_DIR}_loss.png", f"{OUT_DIR}_log_history.csv"]:
            api.upload_file(path_or_fileobj=f, path_in_repo=f"training_logs/{os.path.basename(f)}",
                            repo_id=HUB_MODEL_ID, commit_message=f"{RUN_NAME}: training logs")
        print("☁️ loss 圖與訓練紀錄已上傳")
except Exception as e:
    print(f"⚠️ 畫圖或上傳紀錄失敗（外掛已安全上傳，不影響）：{e}")

if TRIAL:
    print("🧪 試跑結束。確認 loss 有下降、沒有 NaN 後，把 MODE 改成下一階段")
