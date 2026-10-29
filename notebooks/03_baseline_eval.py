# ============================================================
# 筆記本 03｜正式基準考試：未訓練的 Qwen3-4B-Instruct-2507
# 執行環境：Colab T4 GPU（執行階段 → 變更執行階段類型 → T4 GPU）
# 考三份考卷：test（500）、test_unseen_phrasing（394）、真人題（20）
# 每段之間用「# ===== 第 N 段」分隔，一段貼一格
# ============================================================

# ===== 第 1 段｜安裝套件 =====
# !pip install -q -U "transformers>=4.56" accelerate datasets huggingface_hub

# ===== 第 2 段｜連接 Drive、登入 Hugging Face =====
from google.colab import drive, userdata
from huggingface_hub import login, whoami
from pathlib import Path

drive.mount("/content/drive")
PROJECT_DIR = Path("/content/drive/MyDrive/grid-dispatch-assistant")
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HUMAN_TEST = PROJECT_DIR / "data/grid-dispatch-data/human_test_v2.jsonl"

try:
    login(token=userdata.get("HF_TOKEN"), add_to_git_credential=False)
except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
    raise RuntimeError("讀不到 HF_TOKEN！請到左側 🔑 新增 HF_TOKEN 並打開「筆記本存取權」") from None
print(f"✅ Hugging Face 已登入：{whoami()['name']}")
assert HUMAN_TEST.exists(), f"找不到真人考卷：{HUMAN_TEST}"
print("✅ 真人考卷在 Drive 上")

# ===== 第 3 段｜設定與 GPU 檢查 =====
import torch

MODEL_ID   = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_ID = "yoyu001/grid-dispatch-notes-zh"
RUN_NAME   = "03_baseline_instruct"   # 存檔名稱；訓練後重考時換成別的名字
BATCH_SIZE = 16                       # 一次考幾題；記憶體不足就改 8
MAX_NEW_TOKENS = 512                  # 每題最多寫幾個 token
SAMPLE_N   = None                     # None = 全部考；想先快速試跑可設 50

assert torch.cuda.is_available(), "沒有 GPU！請把執行階段改成 T4 GPU"
# 看 GPU 世代判斷：A100（8.0）以上才真的支援 bf16；T4 是 7.5 → 用 fp16
DTYPE = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
print(f"GPU：{torch.cuda.get_device_name()}｜精度：{DTYPE}")

# ===== 第 4 段｜載入三份考卷，統一成同一種格式 =====
import json
from datasets import load_dataset

hub = load_dataset(DATASET_ID)
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
        date, time = c["now"].split(" ")
        user = (f"【現在時間】{date}（{c['weekday']}）{time}\n"
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
print("\n----- 真人第 1 題，模型會看到的輸入 -----")
print(EXAMS["human"][0]["messages"][1]["content"])

# ===== 第 5 段｜載入模型（左側補齊，才能一次考很多題） =====
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left"      # 批次生成要從左邊補空白，答案才會接在題目正後方
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map="auto")
model.eval()
print(f"GPU 記憶體用量：{torch.cuda.memory_allocated() / 1e9:.1f} GB")

# ===== 第 6 段｜批次作答，邊考邊存到 Drive（斷線可以接著考） =====
import time

PRED_FILE = RESULTS_DIR / f"{RUN_NAME}_predictions.jsonl"

def load_done():
    """讀出已經考完的題目，斷線重跑時會跳過。
    只採用「同一個模型、同樣設定」的作答，避免換模型卻忘了改 RUN_NAME 而讀到舊答案"""
    if not PRED_FILE.exists():
        return {}
    done = {}
    for line in open(PRED_FILE, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:      # 斷線時最後一行可能只寫一半，跳過
            continue
        if r.get("model") == MODEL_ID and r.get("max_new") == MAX_NEW_TOKENS:
            done[(r["split"], r["id"])] = r
    return done

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

def run_exam(name, rows):
    done = load_done()
    todo = [r for r in rows if (r["split"], r["id"]) not in done]
    todo.sort(key=lambda r: len(r["messages"][1]["content"]))   # 長度相近的放同一批，比較快
    print(f"【{name}】共 {len(rows)} 題，已完成 {len(rows) - len(todo)}，這次要考 {len(todo)} 題")
    t_start = time.time()
    with open(PRED_FILE, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo[i:i + BATCH_SIZE]
            outs, n_tok, sec = generate_batch(batch)
            for r, o, n in zip(batch, outs, n_tok):
                f.write(json.dumps({"split": r["split"], "id": r["id"], "category": r["category"],
                                    "model": MODEL_ID, "max_new": MAX_NEW_TOKENS,
                                    "output": o.strip(), "gen_tokens": n, "batch_sec": round(sec, 2),
                                    "batch_size": len(batch)}, ensure_ascii=False) + "\n")
            f.flush()
            pct = (i + len(batch)) / len(todo)
            eta = (time.time() - t_start) / pct * (1 - pct)
            print(f"  {i + len(batch):4d}/{len(todo)}　這批 {sec:5.1f} 秒　預估剩 {eta / 60:4.1f} 分鐘")

for name, rows in EXAMS.items():
    run_exam(name, rows)
print(f"💾 全部作答已存到：{PRED_FILE}")

# ===== 第 7 段｜評分規則 =====
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

# ===== 第 8 段｜成績單 =====
import pandas as pd

preds = load_done()
missing = [(r["split"], r["id"]) for rows in EXAMS.values() for r in rows if (r["split"], r["id"]) not in preds]
assert not missing, f"還有 {len(missing)} 題沒作答（例如 {missing[:3]}），請重新執行第 6 段"
records = []
for name, rows in EXAMS.items():
    for r in rows:
        p = preds[(r["split"], r["id"])]
        records.append({"split": r["split"], "category": r["category"], "id": r["id"],
                        "gold_intent": r["gold"]["intent"], "gen_tokens": p["gen_tokens"],
                        "output": p["output"],
                        **grade(p["output"], r["gold"])})
df = pd.DataFrame(records)
METRICS = ["pure_json", "json_ok", "intent_ok", "core_ok", "exact_ok"]

print("===== 各考卷成績（%）=====")
summary = (df.groupby("split")[METRICS].mean() * 100).round(1)
summary["題數"] = df.groupby("split").size()
print(summary.loc[["test", "test_unseen_phrasing", "human"]])

print("\n===== 各題型成績（test + unseen，%）=====")
syn = df[df.split != "human"]
by_cat = (syn.groupby("category")[["json_ok", "intent_ok", "core_ok", "exact_ok"]].mean() * 100).round(1)
by_cat["題數"] = syn.groupby("category").size()
print(by_cat)

# 安全指標：該拒絕的有沒有拒絕？不該拒絕的有沒有亂拒絕？
refuse_rows = df[df.gold_intent == "refuse"]
normal_rows = df[df.gold_intent != "refuse"]
print(f"\n🛡️ 拒絕召回率（該拒絕且有拒絕）：{refuse_rows.intent_ok.mean() * 100:.1f}%（{len(refuse_rows)} 題）")
print(f"🛡️ 誤拒率（不該拒絕卻拒絕）：{(normal_rows.pred_intent == 'refuse').mean() * 100:.1f}%（{len(normal_rows)} 題）")

# fp16 溢位檢查：輸出空白或整串同一個字元，代表數值爆掉，不是模型不會
broken = df[(df.gen_tokens <= 1) | df.output.str.fullmatch(r"(.)\1{10,}", na=False)]
print(f"⚠️ 疑似 fp16 溢位的輸出：{len(broken)} 題" + ("（請把這些題目列出來給 Claude 看）" if len(broken) else ""))
print(f"⏱️ 平均每題輸出 {df.gen_tokens.mean():.0f} 個 token")

stamp = pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y%m%d-%H%M")
df.drop(columns=["output"]).to_csv(RESULTS_DIR / f"{stamp}_{RUN_NAME}_scores.csv", index=False, encoding="utf-8-sig")
summary.to_csv(RESULTS_DIR / f"{stamp}_{RUN_NAME}_summary.csv", encoding="utf-8-sig")
print(f"💾 成績已存到 Drive：{RESULTS_DIR}")

# ===== 第 9 段｜看看錯在哪裡（每份考卷抽 3 題錯的） =====
rows_by_key = {(r["split"], r["id"]): r for rows in EXAMS.values() for r in rows}
for split in ["test", "test_unseen_phrasing", "human"]:
    wrong = df[(df.split == split) & (~df.exact_ok)]
    print(f"\n########## {split}：錯 {len(wrong)} 題 ##########")
    for _, w in wrong.head(3).iterrows():
        r = rows_by_key[(w.split, w.id)]
        print(f"\n--- {w.id}（{w.category}）---")
        print(r["messages"][1]["content"].split("【操作員備註】")[1])
        print("模型：", preds[(w.split, w.id)]["output"][:400])
        print("答案：", json.dumps(r["gold"], ensure_ascii=False)[:400])

# ===== 第 10 段｜收工（這段之後不要再跑其他段，Drive 已中斷） =====
drive.flush_and_unmount()
print("✅ 已同步 Drive，可以關閉 Colab")
