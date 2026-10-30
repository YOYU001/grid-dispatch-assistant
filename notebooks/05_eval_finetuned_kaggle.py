# ============================================================
# 筆記本 05（Kaggle 版）｜微調後重考：Qwen3-4B-Instruct-2507 ＋ LoRA 外掛（A 版）
# 跟筆記本 03（基準考試）用「同一份考卷、同樣生成設定、同一個評分器」，比較訓練前後
# 事前準備：
#   - Settings → Accelerator 選 GPU T4；Internet 打開
#   - Add-ons → Secrets 加 HF_TOKEN 並打勾
#   - 右側 Add Input → Upload：上傳 human_test_v2.jsonl（真人考卷，設成 Private）
#   - 第 1 段 pip 那行要「去掉開頭的 #」才會安裝
#   - 這本筆記本的輸出會印出真人題的標準答案 → 筆記本一定要維持 Private
# 先用 SAMPLE_N = 20 試跑（存檔名稱自動加 -trial），確認沒問題再改 None 考全部
# ============================================================

# ===== 第 1 段｜安裝套件（版本跟筆記本 04 訓練時一樣） =====
# !pip install -q "transformers==5.22.0" "peft==0.21.2" accelerate datasets huggingface_hub

# ===== 第 2 段｜主要設定（只需要改這裡） =====
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # Kaggle T4 x2 只用第 1 張（一定要在 import torch 之前）

MODEL_ID     = "Qwen/Qwen3-4B-Instruct-2507"          # 原模型（跟基準考試同一個）
ADAPTER_NAME = "grid-dispatch-A-instruct-qlora-v1"    # 筆記本 04 訓練出的外掛（HF 上的 repo 名稱）
DATASET_ID   = "yoyu001/grid-dispatch-notes-zh"
RUN_NAME     = "05_A-instruct-qlora-v1"               # 這次考試的存檔名稱；換外掛就要換名字
BASELINE_RUN = "03_baseline_instruct"                 # 要比較的基準考試

BATCH_SIZE     = 16     # 一次考幾題；記憶體不足就改 8
MAX_NEW_TOKENS = 512    # 每題最多寫幾個 token（必須跟基準考試一樣）
SAMPLE_N       = 20     # 試跑：每份考卷只考前 20 題；確認沒問題後改成 None 考全部

if SAMPLE_N:
    RUN_NAME += "-trial"   # 試跑的結果另外存，不會混進正式成績

# ===== 第 3 段｜登入、確認 GPU、讀外掛的訓練紀錄（確認是最終版外掛） =====
import glob, json
from pathlib import Path
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

found = glob.glob("/kaggle/input/**/human_test_v2.jsonl", recursive=True)
assert found, "找不到真人考卷！請在右側 Add Input → Upload 上傳 human_test_v2.jsonl"
HUMAN_TEST = Path(found[0])
print(f"✅ 真人考卷：{HUMAN_TEST}")

ADAPTER_ID  = f"{HF_USER}/{ADAPTER_NAME}"
ADAPTER_SHA = api.model_info(ADAPTER_ID).sha                       # repo 的版本（載入時鎖定）
# 外掛「檔案本身」的指紋：repo 只改 README 不會變，所以用它判斷能不能沿用之前的作答
ADAPTER_HASH = api.get_paths_info(ADAPTER_ID, ["adapter_model.safetensors"], revision=ADAPTER_SHA)[0].lfs.sha256

# run_config.json 是筆記本 04 在「最終外掛上傳之後」才寫的 → 有它才代表外掛是最終版
try:
    rc_path = hf_hub_download(ADAPTER_ID, "training_logs/run_config.json", revision=ADAPTER_SHA)
except Exception:
    raise RuntimeError("找不到 training_logs/run_config.json：外掛可能不是最終版，請確認筆記本 04 第 9 段有跑完") from None
RUN_CONFIG = json.load(open(rc_path, encoding="utf-8"))
steps = RUN_CONFIG["train"]["steps_done"]
print(f"✅ 外掛：{ADAPTER_ID}（訓練 {steps} 步、{RUN_CONFIG['train_minutes']} 分鐘｜指紋 {ADAPTER_HASH[:8]}）")
assert steps >= 400, f"只訓練了 {steps} 步，不像是正式版外掛"

RESULTS_REPO = f"{HF_USER}/grid-dispatch-results"
RESULTS_DIR  = Path(f"/kaggle/working/results/{RUN_NAME}")   # 試跑與正式分開放
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ===== 第 4 段｜載入考卷（鎖定訓練時的資料版本）與基準作答 =====
from datasets import load_dataset

DATASET_SHA = RUN_CONFIG["dataset_sha"]
hub = load_dataset(DATASET_ID, revision=DATASET_SHA)    # 資料集之後更新成 v2，這裡仍考同一份
print(f"✅ 資料版本：{DATASET_SHA[:8]}（跟訓練時相同）")
SYSTEM_PROMPT = hub["test"][0]["messages"][0]["content"]   # 跟訓練資料用同一份工作說明書

def from_hub(split):
    """合成考卷：輸入 = system + user，標準答案 = assistant"""
    rows = []
    for r in hub[split]:
        rows.append({"id": r["id"], "split": split, "category": r["category"],
                     "messages": r["messages"][:2],
                     "gold": json.loads(r["messages"][2]["content"])})
    return rows

def from_human(path):
    """真人考卷：把情境組成跟合成資料一模一樣的輸入格式"""
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        c, g = r["context"], r["context"]["grid"]
        date, time_ = c["now"].split(" ")
        user = (f"【現在時間】{date}（{c['weekday']}）{time_}\n"
                f"【電網狀況】燈號：{g['light']}；預估尖峰時段：{g['peak_hours']}；預估備轉容量率：{g['reserve_rate']}%\n"
                f"【操作員備註】{r['note']}")
        rows.append({"id": r["id"], "split": "human", "category": r["category"],
                     "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": user}],
                     "gold": r["expect"]})
    return rows

EXAMS = {"test": from_hub("test"),
         "test_unseen_phrasing": from_hub("test_unseen_phrasing"),
         "human": from_human(HUMAN_TEST)}
if SAMPLE_N:
    EXAMS = {k: v[:SAMPLE_N] for k, v in EXAMS.items()}
for k, v in EXAMS.items():
    print(f"{k:22s} {len(v):4d} 題")
print(f"🧪 試跑模式（存成 {RUN_NAME}）" if SAMPLE_N else f"🚀 正式考試（存成 {RUN_NAME}）")

# 基準作答：從 HF 下載；沒有的話從 Kaggle Input 找（GitHub 的 results/ 資料夾有這個檔）
base_name = f"{BASELINE_RUN}_predictions.jsonl"
try:
    base_path = hf_hub_download(RESULTS_REPO, f"{BASELINE_RUN}/{base_name}", repo_type="dataset")
except Exception:
    cand = glob.glob(f"/kaggle/input/**/{base_name}", recursive=True)
    assert cand, f"找不到基準作答 {base_name}！請到右側 Add Input → Upload 上傳"
    base_path = cand[0]
base_preds = {}
for line in open(base_path, encoding="utf-8"):
    r = json.loads(line)
    assert r["model"] == MODEL_ID and r["max_new"] == MAX_NEW_TOKENS, "基準作答的模型或設定不同，不能比較！"
    base_preds[(r["split"], r["id"])] = r
# 確認基準考的是同一份考卷：每題的題型要對得上
mismatch = [r["id"] for rows in EXAMS.values() for r in rows
            if (r["split"], r["id"]) in base_preds and base_preds[(r["split"], r["id"])]["category"] != r["category"]]
assert not mismatch, f"基準考卷跟這次不一樣（例如 {mismatch[:3]}）！"
print(f"✅ 基準作答：{len(base_preds)} 題，題型與這次考卷一致")

# ===== 第 5 段｜載入 fp16 原模型 ＋ LoRA 外掛，確認外掛完整且有訓練過 =====
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left"      # 批次生成要從左邊補空白，答案才會接在題目正後方
assert tokenizer.pad_token == "<|endoftext|>", f"pad token 跟訓練時不同：{tokenizer.pad_token}"

# 考試用 fp16 原模型（不壓 4-bit）：跟基準考試同精度，比較才公平；考試不用梯度，T4 放得下
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map={"": 0})
model = PeftModel.from_pretrained(base, ADAPTER_ID, revision=ADAPTER_SHA)
model.eval()

# 檢查 1：外掛的 B 片一開始全是 0，訓練過才會變成非 0 → 全是 0 代表載到空外掛
lora_b = [p for n, p in model.named_parameters() if "lora_B" in n]
nonzero = sum(int(p.abs().sum() > 0) for p in lora_b)
print(f"LoRA B 片：{len(lora_b)} 片，其中 {nonzero} 片非 0")
assert len(lora_b) == 252 and nonzero == 252, "外掛不完整或沒有訓練過！請確認 ADAPTER_NAME"

# 檢查 2：設定跟筆記本 04 一致
cfg = model.peft_config["default"]
short = sorted({t.split(".")[-1] for t in cfg.target_modules})   # all-linear 會展開成完整路徑，只看最後一段
print(f"外掛設定：r={cfg.r}、alpha={cfg.lora_alpha}、貼在 {short}")
assert (cfg.r, cfg.lora_alpha) == (RUN_CONFIG["lora"]["r"], RUN_CONFIG["lora"]["alpha"]), "外掛設定跟訓練紀錄不同！"
print(f"GPU 記憶體用量：{torch.cuda.memory_allocated(0) / 1e9:.1f} GB")

# ===== 第 6 段｜環境檢查：先「關掉外掛」重考 32 題，看跟基準考試的答案是否一樣 =====
# 目的：確認 Colab／Kaggle、套件版本的差異可以忽略 → 之後的「進步」才都是微調的功勞
import time

def generate_batch(batch):
    texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True) for r in batch]
    enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tokenizer.pad_token_id)
    sec = time.time() - t0
    new = out[:, enc["input_ids"].shape[1]:]                     # 只取模型新寫的部分
    outs = tokenizer.batch_decode(new, skip_special_tokens=True)
    n_tok = [int((row != tokenizer.pad_token_id).sum()) for row in new]
    return outs, n_tok, sec

check_rows = EXAMS["human"][:20] + EXAMS["test"][:12]
same = []
with model.disable_adapter():                 # 暫時拔掉外掛 = 未訓練的原模型
    for i in range(0, len(check_rows), BATCH_SIZE):
        batch = check_rows[i:i + BATCH_SIZE]
        outs, _, _ = generate_batch(batch)
        same += [o.strip() == base_preds[(r["split"], r["id"])]["output"] for r, o in zip(batch, outs)]
rate = sum(same) / len(same) * 100
print(f"關掉外掛時，跟基準考試答案逐字相同：{sum(same)}/{len(same)} 題（{rate:.0f}%）")
print("✅ 環境差異可以忽略" if rate >= 90 else
      "⚠️ 差異偏大：可能是套件或 GPU 不同造成的。成績仍可比較，但請把這個數字告訴 Claude，報告要註明")

# ===== 第 7 段｜把外掛併進原模型（先確認合併前後輸出幾乎一樣） =====
# 合併 = 把便利貼的修正直接寫進字典；生成時少算一次外掛，比較快。但 fp16 捨入可能造成微小差異
probe = [EXAMS["human"][i] for i in range(4)]
texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True) for r in probe]
enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
with torch.no_grad():
    before = model(**enc).logits
    model.merge_adapter()                     # 先「可還原」地合併
    after = model(**enc).logits
mask = enc["attention_mask"].bool()
agree = (before.argmax(-1) == after.argmax(-1))[mask].float().mean().item() * 100
print(f"合併前後：每個位置最有把握的字一致 {agree:.2f}%｜最大差值 {(before - after).abs().max().float().item():.3f}")
del before, after
if agree >= 99.9:
    model = model.merge_and_unload()          # 已經合併過，這裡只是把外掛的殼拿掉
    MERGED = True
    print("✅ 已合併外掛")
else:
    model.unmerge_adapter()                   # 還原，改用「不合併」（慢一點，但最準）
    MERGED = False
    print("⚠️ 合併差異偏大，改用不合併的方式考試")
torch.cuda.empty_cache()

# ===== 第 8 段｜先考 1 題真人題，親眼看看微調後的輸出 =====
demo = EXAMS["human"][0]
outs, n_tok, sec = generate_batch([demo])
print("【操作員備註】" + demo["messages"][1]["content"].split("【操作員備註】")[1])
print(f"\n【微調後輸出】（{n_tok[0]} tokens，{sec:.1f} 秒）\n{outs[0]}")
print(f"\n【基準輸出】\n{base_preds[('human', demo['id'])]['output']}")
print(f"\n【標準答案】\n{json.dumps(demo['gold'], ensure_ascii=False)}")

# ===== 第 9 段｜批次作答；每 10 批推一次到 HF（Kaggle 斷線重跑會從 HF 接續） =====
PRED_NAME = f"{RUN_NAME}_predictions.jsonl"
PRED_FILE = RESULTS_DIR / PRED_NAME
api.create_repo(RESULTS_REPO, repo_type="dataset", private=True, exist_ok=True)
if not api.repo_info(RESULTS_REPO, repo_type="dataset").private:
    raise RuntimeError(f"{RESULTS_REPO} 是公開的！停止")

# 接續：本地沒有作答檔，但 HF 上有 → 先下載回來（Kaggle 重開後 /kaggle/working 會清空）
if not PRED_FILE.exists() and api.file_exists(RESULTS_REPO, f"{RUN_NAME}/{PRED_NAME}", repo_type="dataset"):
    p = hf_hub_download(RESULTS_REPO, f"{RUN_NAME}/{PRED_NAME}", repo_type="dataset")
    PRED_FILE.write_bytes(Path(p).read_bytes())
    print("🔁 從 HF 下載之前的作答，接著考")

def drop_partial_line():
    """被中斷時最後一行可能只寫一半；不先截掉，新答案會黏在後面，整行都讀不到"""
    if PRED_FILE.exists() and PRED_FILE.stat().st_size:
        data = PRED_FILE.read_bytes()
        if not data.endswith(b"\n"):
            PRED_FILE.write_bytes(data[:data.rfind(b"\n") + 1])

def load_done():
    """讀出已經考完的題目。只採用「同一個原模型＋同一份外掛檔＋同樣設定」的作答"""
    if not PRED_FILE.exists():
        return {}
    done = {}
    for line in open(PRED_FILE, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (r.get("model") == MODEL_ID and r.get("adapter_hash") == ADAPTER_HASH
                and r.get("max_new") == MAX_NEW_TOKENS):
            done[(r["split"], r["id"])] = r
    return done

def push_preds(msg):
    api.upload_file(path_or_fileobj=str(PRED_FILE), path_in_repo=f"{RUN_NAME}/{PRED_NAME}",
                    repo_id=RESULTS_REPO, repo_type="dataset", commit_message=msg)

def run_exam(name, rows):
    drop_partial_line()
    done = load_done()
    todo = [r for r in rows if (r["split"], r["id"]) not in done]
    todo.sort(key=lambda r: len(r["messages"][1]["content"]))   # 長度相近的放同一批，比較快
    print(f"【{name}】共 {len(rows)} 題，已完成 {len(rows) - len(todo)}，這次要考 {len(todo)} 題")
    if not todo:
        return
    t_start = time.time()
    with open(PRED_FILE, "a", encoding="utf-8") as f:
        for b, i in enumerate(range(0, len(todo), BATCH_SIZE), start=1):
            batch = todo[i:i + BATCH_SIZE]
            outs, n_tok, sec = generate_batch(batch)
            for r, o, n in zip(batch, outs, n_tok):
                f.write(json.dumps({"split": r["split"], "id": r["id"], "category": r["category"],
                                    "model": MODEL_ID, "adapter": ADAPTER_ID, "adapter_hash": ADAPTER_HASH,
                                    "merged": MERGED, "max_new": MAX_NEW_TOKENS,
                                    "output": o.strip(), "gen_tokens": n, "batch_sec": round(sec, 2),
                                    "batch_size": len(batch)}, ensure_ascii=False) + "\n")
            f.flush()
            pct = (i + len(batch)) / len(todo)
            eta = (time.time() - t_start) / pct * (1 - pct)
            print(f"  {i + len(batch):4d}/{len(todo)}　這批 {sec:5.1f} 秒　預估剩 {eta / 60:4.1f} 分鐘")
            if b % 10 == 0:
                push_preds(f"{RUN_NAME}: {name} {i + len(batch)}/{len(todo)}")
    push_preds(f"{RUN_NAME}: {name} done")
    print("  ☁️ 已推到 HF")

for name, rows in EXAMS.items():
    run_exam(name, rows)
print(f"💾 全部作答已存到：{PRED_FILE}")

# ===== 第 10 段｜評分規則（跟筆記本 03 一字不差，比較才公平） =====
import re

def parse_json(text):
    """從輸出抓出 JSON；抓不到回傳 None"""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None

def norm_value(v):
    """把 'false'/False、'0'/0、'2026-10-28 9:00'/'2026-10-28 09:00' 等寫法統一"""
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})[ T](\d{1,2}):(\d{2})", s)
        if m:
            return f"{m.group(1)} {int(m.group(2)):02d}:{m.group(3)}"
        return s
    return v

def ckey(c, with_time=True):
    """一條限制條件的『指紋』。型別也放進去，因為 Python 裡 False == 0"""
    action = str(c.get("action", "set")).lower()
    val = None if action == "clear" else norm_value(c.get("value"))
    k = (c.get("workload_id"), c.get("field"), action, type(val).__name__, val)
    return k + ((norm_value(c.get("valid_until")),) if with_time else ())

def grade(output, gold):
    pred = parse_json(output)
    s = {"pure_json": output.startswith("{") and output.endswith("}"),
         "json_ok": pred is not None and isinstance(pred.get("constraints", []), list),
         "pred_intent": None, "intent_ok": False, "core_ok": False, "exact_ok": False}
    if not s["json_ok"]:
        return s
    s["pred_intent"] = pred.get("intent")
    s["intent_ok"] = s["pred_intent"] == gold["intent"]
    try:
        p_full = {ckey(c) for c in pred.get("constraints", [])}
        p_core = {ckey(c, False) for c in pred.get("constraints", [])}
    except (AttributeError, TypeError):
        return s
    g_full = {ckey(c) for c in gold["constraints"]}
    g_core = {ckey(c, False) for c in gold["constraints"]}
    s["core_ok"] = s["intent_ok"] and p_core == g_core          # 工作、欄位、數值都對（不看有效期限）
    s["exact_ok"] = s["intent_ok"] and p_full == g_full         # 連有效期限也全對（最嚴格）
    return s

# ===== 第 11 段｜成績單：微調後 vs 基準（同一個評分器重新算兩邊） =====
import pandas as pd

METRICS = ["pure_json", "json_ok", "intent_ok", "core_ok", "exact_ok"]
SPLITS  = ["test", "test_unseen_phrasing", "human"]

def score(preds, tag):
    """把作答檔評分成表格；題目以這次的考卷為準，缺題就停"""
    missing = [(r["split"], r["id"]) for rows in EXAMS.values() for r in rows if (r["split"], r["id"]) not in preds]
    assert not missing, f"[{tag}] 還有 {len(missing)} 題沒作答（例如 {missing[:3]}），請重跑第 9 段"
    recs = []
    for rows in EXAMS.values():
        for r in rows:
            p = preds[(r["split"], r["id"])]
            recs.append({"split": r["split"], "category": r["category"], "id": r["id"],
                         "gold_intent": r["gold"]["intent"], "gen_tokens": p["gen_tokens"],
                         "output": p["output"], **grade(p["output"], r["gold"])})
    return pd.DataFrame(recs)

ft_df   = score(load_done(), "微調後")
base_df = score(base_preds, "基準")

def summary_of(df):
    return (df.groupby("split")[METRICS].mean() * 100).round(1).loc[SPLITS]

ft_sum, base_sum = summary_of(ft_df), summary_of(base_df)
n_by_split = ft_df.groupby("split").size()
tables = []
print("===== 各考卷成績（%）：基準 → 微調後 =====")
for m in METRICS:
    t = pd.DataFrame({"基準": base_sum[m], "微調後": ft_sum[m]})
    t["進步"] = (t["微調後"] - t["基準"]).round(1)
    t["題數"] = n_by_split
    print(f"\n【{m}】\n{t.to_string()}")
    tables.append(t.assign(指標=m))
compare = pd.concat(tables)          # 存檔用：每列 = 一份考卷 × 一項指標

print("\n===== 各題型 exact_ok（test + unseen，%）=====")
by_cat = pd.DataFrame({
    "基準":   base_df[base_df.split != "human"].groupby("category").exact_ok.mean() * 100,
    "微調後": ft_df[ft_df.split != "human"].groupby("category").exact_ok.mean() * 100,
}).round(1)
by_cat["進步"] = (by_cat["微調後"] - by_cat["基準"]).round(1)
by_cat["題數"] = ft_df[ft_df.split != "human"].groupby("category").size()
print(by_cat.sort_values("微調後").to_string())

def safety(df):
    refuse, normal = df[df.gold_intent == "refuse"], df[df.gold_intent != "refuse"]
    return refuse.intent_ok.mean() * 100, (normal.pred_intent == "refuse").mean() * 100, len(refuse), len(normal)

(b_rec, b_fr, n_ref, n_norm), (f_rec, f_fr, _, _) = safety(base_df), safety(ft_df)
print(f"\n🛡️ 拒絕召回率（越高越好）：{b_rec:.1f}% → {f_rec:.1f}%（{n_ref} 題）")
print(f"🛡️ 誤拒率（越低越好）　　：{b_fr:.1f}% → {f_fr:.1f}%（{n_norm} 題）")

repeated = ft_df.output.apply(lambda s: bool(re.fullmatch(r"(.)\1{10,}", s or "")))   # 用 Python re（新版 pandas 不支援 \1）
broken = ft_df[(ft_df.gen_tokens <= 1) | repeated]
print(f"⚠️ 疑似 fp16 溢位的輸出：{len(broken)} 題" + ("（請把這些題目列出來給 Claude 看）" if len(broken) else ""))
print(f"⏱️ 平均每題輸出 token：基準 {base_df.gen_tokens.mean():.0f} → 微調後 {ft_df.gen_tokens.mean():.0f}")

# ===== 第 12 段｜錯題分析：真人 20 題逐題對照 ＋ 微調後還錯什麼 =====
rows_by_key = {(r["split"], r["id"]): r for rows in EXAMS.values() for r in rows}
ft_preds = load_done()

print("===== 真人題逐題：基準 → 微調後（○ = 完全正確，△ = 只錯有效期限，✗ = 錯）=====")
def mark(r):
    return "○" if r.exact_ok else ("△" if r.core_ok else "✗")
hb, hf = base_df[base_df.split == "human"], ft_df[ft_df.split == "human"]
h = pd.DataFrame({"題型": hf.category.values,
                  "基準": hb.apply(mark, axis=1).values,
                  "微調後": hf.apply(mark, axis=1).values}, index=hf.id.values)
print(h.to_string())

def error_type(r):
    """把錯題分成幾種：方便決定資料 v2 要補什麼"""
    if not r.json_ok:     return "JSON 壞掉"
    if not r.intent_ok:   return f"意圖錯（{r.gold_intent} → {r.pred_intent}）"
    if r.core_ok:         return "只錯有效期限"
    return "限制條件錯（工作／欄位／數值）"

wrong = ft_df[~ft_df.exact_ok].copy()
if len(wrong):
    wrong["錯誤類型"] = wrong.apply(error_type, axis=1)
    print(f"\n===== 微調後還錯 {len(wrong)} 題：錯誤類型統計 =====")
    print(wrong.groupby(["split", "錯誤類型"]).size().to_string())
else:
    print("\n🎉 微調後全部答對")

def diff(output, gold):
    """列出模型多寫了哪些限制、漏寫了哪些限制"""
    pred = parse_json(output) or {}
    try:
        p = {ckey(c) for c in pred.get("constraints", [])}
    except (AttributeError, TypeError):
        return "    （限制格式錯誤）"
    g = {ckey(c) for c in gold["constraints"]}
    fmt = lambda k: f"{k[0]}.{k[1]}={k[4]}（{k[2]}，期限 {k[5]}）"
    return "\n".join([f"    多寫：{fmt(k)}" for k in p - g] + [f"    漏寫：{fmt(k)}" for k in g - p]) or "    （限制相同，錯在意圖）"

for split in SPLITS:
    w = wrong[wrong.split == split] if len(wrong) else wrong
    print(f"\n########## {split}：錯 {len(w)} 題（列出前 5 題）##########")
    for _, x in w.head(5).iterrows():
        r = rows_by_key[(x.split, x.id)]
        print(f"\n--- {x.id}（{x.category}｜{x['錯誤類型']}）---")
        print(r["messages"][1]["content"].split("【操作員備註】")[1])
        print(diff(ft_preds[(x.split, x.id)]["output"], r["gold"]))

# ===== 第 13 段｜存成績、推到 HF 私人 repo =====
stamp = pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y%m%d-%H%M")
ft_df.drop(columns=["output"]).to_csv(RESULTS_DIR / f"{stamp}_{RUN_NAME}_scores.csv", index=False, encoding="utf-8-sig")
compare.to_csv(RESULTS_DIR / f"{stamp}_{RUN_NAME}_vs_{BASELINE_RUN}.csv", encoding="utf-8-sig")
by_cat.to_csv(RESULTS_DIR / f"{stamp}_{RUN_NAME}_by_category.csv", encoding="utf-8-sig")

eval_config = {
    "run_name": RUN_NAME, "baseline_run": BASELINE_RUN, "sample_n": SAMPLE_N,
    "model": MODEL_ID, "adapter": ADAPTER_ID, "adapter_sha": ADAPTER_SHA, "adapter_hash": ADAPTER_HASH,
    "adapter_steps": steps, "merged": MERGED, "merge_argmax_agree": round(agree, 2),
    "env_check_same_as_baseline": f"{sum(same)}/{len(same)}",
    "dataset": DATASET_ID, "dataset_sha": DATASET_SHA, "dtype": str(DTYPE),
    "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "batch_size": BATCH_SIZE, "padding_side": "left"},
    "safety": {"refuse_recall": round(f_rec, 1), "false_refusal": round(f_fr, 1),
               "baseline_refuse_recall": round(b_rec, 1), "baseline_false_refusal": round(b_fr, 1)},
    "versions": {"transformers": transformers.__version__, "peft": peft.__version__, "torch": torch.__version__},
    "gpu": torch.cuda.get_device_name(0),
}
(RESULTS_DIR / f"{RUN_NAME}_eval_config.json").write_text(json.dumps(eval_config, ensure_ascii=False, indent=2), encoding="utf-8")

api.upload_folder(folder_path=str(RESULTS_DIR), repo_id=RESULTS_REPO, repo_type="dataset",
                  path_in_repo=RUN_NAME, commit_message=f"{RUN_NAME} results")
print(f"☁️ 結果已存到（私人）：https://huggingface.co/datasets/{RESULTS_REPO}/tree/main/{RUN_NAME}")
if SAMPLE_N:
    print("🧪 試跑結束。確認沒問題後，把 SAMPLE_N 改成 None，Restart 後重跑全部")
