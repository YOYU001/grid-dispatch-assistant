# ============================================================
# 筆記本 06（Kaggle 版）｜把 LoRA 外掛併進原模型，推成「完整模型」到 HF
# 為什麼：GSM8K 評估工具 lighteval 與 smol-course 排行榜，都需要一個完整模型 repo
# 比喻：把貼在字典上的便利貼內容直接抄進字典，變成一本新字典
# 事前準備：
#   - Settings → Accelerator 選 GPU T4；Internet 打開
#   - Add-ons → Secrets 加 HF_TOKEN 並打勾（要有 write 權限）
#   - 第 1 段 pip 那行要「去掉開頭的 #」才會安裝
# 預估時間：約 15～20 分鐘（大部分是下載 8 GB 原模型與上傳 8 GB 合併模型）
# ============================================================

# ===== 第 1 段｜安裝套件（版本跟筆記本 04、05 一樣） =====
# !pip install -q "transformers==5.22.0" "peft==0.21.2" "accelerate>=1.0" datasets huggingface_hub
# Kaggle 預裝的 torchao 太舊，新版 peft 載入外掛時會報錯；我們用不到它，直接移除
# !pip uninstall -y -q torchao

# ===== 第 2 段｜主要設定（只需要改這裡） =====
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # Kaggle T4 x2 只用第 1 張（一定要在 import torch 之前）

MODEL_ID     = "Qwen/Qwen3-4B-Instruct-2507"            # 原模型
ADAPTER_NAME = "grid-dispatch-A-instruct-qlora-v1"      # 筆記本 04 訓練出的外掛
MERGED_NAME  = "grid-dispatch-A-instruct-merged-v1"     # 合併後的新模型 repo 名稱
DATASET_ID   = "yoyu001/grid-dispatch-notes-zh"          # 只用來取訓練時的 system prompt
PRIVATE      = True    # 先設私人；要交 smol-course 時再決定是否公開

# ===== 第 3 段｜登入、確認 GPU、鎖定外掛版本（跟筆記本 05 同樣的檢查） =====
import json
import torch, transformers, peft
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, whoami, HfApi, hf_hub_download

try:
    login(token=UserSecretsClient().get_secret("HF_TOKEN"), add_to_git_credential=False)
except Exception:
    raise RuntimeError("讀不到 HF_TOKEN！請到 Add-ons → Secrets 新增 HF_TOKEN 並打勾") from None
HF_USER = whoami()["name"]
api = HfApi()
print(f"✅ Hugging Face 已登入：{HF_USER}")

assert torch.cuda.is_available(), "沒有 GPU！請到 Settings → Accelerator 選 GPU T4"
DTYPE = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
print(f"GPU：{torch.cuda.get_device_name(0)}｜精度：{DTYPE}")
print(f"套件版本：transformers {transformers.__version__}｜peft {peft.__version__}")

ADAPTER_ID   = f"{HF_USER}/{ADAPTER_NAME}"
MERGED_ID    = f"{HF_USER}/{MERGED_NAME}"
ADAPTER_SHA  = api.model_info(ADAPTER_ID).sha
ADAPTER_HASH = api.get_paths_info(ADAPTER_ID, ["adapter_model.safetensors"], revision=ADAPTER_SHA)[0].lfs.sha256
rc_path = hf_hub_download(ADAPTER_ID, "training_logs/run_config.json", revision=ADAPTER_SHA)
RUN_CONFIG = json.load(open(rc_path, encoding="utf-8"))
steps = RUN_CONFIG["train"]["steps_done"]
print(f"✅ 外掛：{ADAPTER_ID}（訓練 {steps} 步｜指紋 {ADAPTER_HASH[:8]}）")
assert steps >= 400, f"只訓練了 {steps} 步，不像是正式版外掛"
# 筆記本 05 正式重考用的就是這個外掛；指紋不同代表外掛被換過，成績就對不上了
assert ADAPTER_HASH.startswith("1d4b4d4a"), "外掛指紋跟筆記本 05 考試用的不同！請先確認"

# ===== 第 4 段｜載入 fp16 原模型 ＋ 外掛，先記下「合併前」的回答 =====
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map={"": 0})
model = PeftModel.from_pretrained(base, ADAPTER_ID, revision=ADAPTER_SHA)
model.eval()

# 自編的檢查題（不是考卷裡的題目），只用來比對合併前後回答是否相同
PROBE = ("【現在時間】2026-11-02（一）15:00\n"
         "【電網狀況】燈號：黃燈；預估尖峰時段：18:00-21:00；預估備轉容量率：7.20%\n"
         "【操作員備註】今晚批次摘要可以多配合降載，但客服不能降速")
from datasets import load_dataset
first = load_dataset(DATASET_ID, split="validation[:1]")[0]["messages"][0]
assert first["role"] == "system", "資料集第一則訊息不是 system prompt，請檢查"
SYSTEM = first["content"]   # 跟訓練時一模一樣的 system prompt
msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROBE}]

def ask(m):
    enc = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(m.device)
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=256, do_sample=False)
    return tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

before = ask(model)
print("合併前回答：", before)

# ===== 第 5 段｜合併外掛，確認合併前後幾乎一樣（跟筆記本 05 同一套標準） =====
# 不要求逐字相同：fp16 合併會多一次捨入，長回答可能在某個字分岔；改看「每個位置最有把握的字」一致率
text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + before
ids = tokenizer(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    lg_before = model(**ids).logits
    model.merge_adapter()                      # 先「可還原」地合併
    lg_after = model(**ids).logits
agree = (lg_before.argmax(-1) == lg_after.argmax(-1)).float().mean().item() * 100
print(f"合併前後一致率 {agree:.2f}%")
assert torch.isfinite(lg_after).all(), "合併後出現 inf/NaN（fp16 溢位），不要上傳"
assert agree >= 99.9, "合併前後差異偏大！先不要上傳，把數字貼給 Claude 看"
del lg_before, lg_after
model = model.merge_and_unload()               # 已經合併過，這裡只是拿掉外掛的殼
del base                                       # base 和 model 是同一份 GPU 權重，刪名字避免誤用
after = ask(model)
print("合併後回答：", after)
if before == after:
    print("✅ 合併前後回答逐字相同")
else:
    print("⚠️ 逐字不同（fp16 捨入造成），一致率已達標，可以繼續；請把兩段回答貼給 Claude 看")

# ===== 第 6 段｜存成完整模型（safetensors）＋ 寫模型說明卡 =====
from pathlib import Path
OUT = Path(f"/kaggle/working/{MERGED_NAME}")
model.save_pretrained(OUT, safe_serialization=True, max_shard_size="2GB")
tokenizer.save_pretrained(OUT)

card = f"""---
license: apache-2.0
base_model: {MODEL_ID}
language: [zh]
tags: [qlora, sft, trl, smol-course, grid-dispatch]
---

# {MERGED_NAME}

{MODEL_ID} 以 QLoRA 微調後**合併**的完整模型（外掛：`{ADAPTER_ID}`，指紋 `{ADAPTER_HASH[:8]}`）。

任務：讀懂電網調度操作員的中文備註，轉成限制條件 JSON，並拒絕違反 SLA 的指令。模型**不做調度決策**。

- 訓練資料：`{HF_USER}/grid-dispatch-notes-zh`（以規則產生的合成資料，非真實操作員紀錄）
- 訓練：{steps} 步，Kaggle T4，fp16；合併精度 {str(DTYPE).replace('torch.', '')}
- 自訂考卷（exact，訓練前 → 後）：test 19.6% → 98.6%｜沒看過的說法 19.8% → 81.2%｜真人 20 題 30% → 75%
- 限制：對沒看過的說法較不穩定；輸出必須經安全驗證與人員核准才可執行

Hugging Face smol-course 第一章作業。
"""
(OUT / "README.md").write_text(card, encoding="utf-8")
size_gb = sum(f.stat().st_size for f in OUT.glob("*.safetensors")) / 1e9
print(f"✅ 已存到 {OUT}（權重 {size_gb:.1f} GB）")
assert 7 < size_gb < 9, "檔案大小不像 4B fp16 模型（約 8 GB），請檢查"

# ===== 第 7 段｜推上 HF，並重新下載設定檔確認 =====
api.create_repo(MERGED_ID, private=PRIVATE, exist_ok=True)
if PRIVATE:
    assert api.repo_info(MERGED_ID).private, f"{MERGED_ID} 不是私人 repo！請到 HF 網頁改成 Private 再重跑這段"
info = api.upload_folder(folder_path=str(OUT), repo_id=MERGED_ID,
                         commit_message=f"merge {ADAPTER_ID}@{ADAPTER_SHA[:8]} into {MODEL_ID}")
print(f"上傳版本：{info.oid[:8]}（筆記本 07 會顯示同一個版本號）")
files = api.list_repo_files(MERGED_ID)
print(f"HF 上的檔案：{sorted(files)}")
assert any(f.endswith(".safetensors") for f in files) and "config.json" in files, "上傳不完整！"
print(f"☁️ 合併模型已上傳（{'私人' if PRIVATE else '公開'}）：https://huggingface.co/{MERGED_ID}")
print("下一步：開新的 Kaggle 筆記本跑 07（GSM8K），不要在這本裡接著跑")
