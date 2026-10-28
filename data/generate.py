"""
合成資料產生器 v1：電網調度秘書（任務①：備註 → 限制條件 JSON）

概念：「規則老師」先抽出一個結構化情境（要對哪個工作下什麼限制），
      再把它「說成」一句中文備註，標準答案直接由結構化情境算出 → 答案保證正確。

用法：python data/generate.py --n 4000 --out data/out --seed 42
輸出格式與 docs/output_schema_v2.md 一致。
"""
import argparse, json, random, re
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================
# 第 1 部分｜基本設定：工作名稱、說法變體、系統提示
# ============================================================
# 每個工作有多種說法；最後一個說法「只出現在測試集」，用來考沒看過的說法
ALIASES = {
    "chat-ecommerce": ["客服", "線上客服", "客服機器人", "客服聊天服務", "電商客服", "客服那邊的服務", "聊天客服"],
    "batch-summary":  ["批次摘要", "文件摘要", "批次摘要的工作", "摘要那批工作", "摘要排程"],
    "lora-finetune":  ["LoRA微調", "LoRA 微調", "模型微調", "研究團隊的LoRA訓練", "微調訓練", "LoRA那個訓練"],
    "rag-index":      ["RAG索引更新", "RAG 更新", "知識庫索引", "RAG的索引", "知識庫的更新"],
}
ALL_FLEX_ALIASES = ["所有不急的工作", "非急迫的工作", "其他可以調整的工作", "不趕時間的工作", "其他比較有彈性的工作"]
EMERGENCY_ALIASES = ["緊急推論服務", "緊急推論", "醫療防災那個緊急服務"]
FLEXIBLE = ["batch-summary", "lora-finetune", "rag-index"]   # 可延後／暫停的工作
WEEKDAYS = "一二三四五六日"

SYSTEM_PROMPT = (
    "你是資料中心的「電網調度秘書」。讀操作員備註，只輸出一個 JSON："
    '{"intent": "set_constraints|no_action|off_topic|refuse", '
    '"constraints": [{"workload_id", "field", "value", "action": "set|clear", "valid_until"}], '
    '"explanation": "一句白話說明"}。'
    "workload_id：chat-ecommerce、batch-summary、lora-finetune、rag-index、emergency-inference、ALL_FLEXIBLE。"
    "field：flex_tier（0–3）、earliest_start（YYYY-MM-DD HH:MM）、allow_pause（true/false）。"
    "違反客戶合約、調整緊急推論、跳過安全檢查或人員確認，一律 refuse。"
)

def pick(rng, items, heldout_ok):
    """訓練用的題目不能抽到最後一個（保留給測試）；測試用的可以抽全部"""
    return rng.choice(items if heldout_ok else items[:-1])

def is_heldout(text_used, items):
    return text_used == items[-1]


# ============================================================
# 第 2 部分｜情境抽樣：現在時間、電網燈號（參考真實資料）
# ============================================================
# 燈號門檻（台電公開說明，待人工確認最新版）：綠 ≥10%、黃 6–10%、橘 <6%
# 尖峰時段取自 Yoyo NAS 收集器 2026-10 實際出現過的值
PEAK_RANGES = ["13:00-16:00", "18:00-21:00", "17:00-18:00", "18:00-20:00"]

def sample_context(rng):
    month = rng.choice([6, 7, 8, 9] * 3 + [10, 11, 3, 4, 5])       # 夏季比例較高
    day = rng.randint(1, 28)
    now = datetime(2026, month, day, rng.randint(8, 22), rng.choice([0, 10, 20, 30, 40, 50]))
    summer = month in (6, 7, 8, 9)
    light = rng.choices(["綠燈", "黃燈", "橘燈"], weights=[5, 3, 2] if summer else [9, 1, 0])[0]
    rate = {"綠燈": rng.uniform(10, 30), "黃燈": rng.uniform(6, 9.99), "橘燈": rng.uniform(3, 5.99)}[light]
    return {"now": now, "light": light, "rate": round(rate, 2), "peak": rng.choice(PEAK_RANGES)}

def fmt(dt):
    return dt.strftime("%Y-%m-%d %H:%M")

def tonight_end(now):   # 今晚 → 隔天 08:00
    return fmt((now + timedelta(days=1)).replace(hour=8, minute=0))

def today_end(now):     # 今天 → 當天 23:59
    return fmt(now.replace(hour=23, minute=59))


# ============================================================
# 第 3 部分｜單一要求（原子意圖）：說法 + 標準答案 + 規則軌跡
# 每個函式回傳 (句子, constraints, trace, 是否用到測試專用說法)
# ============================================================
SCOPES = [("今晚", tonight_end), ("今天", today_end), ("", None)]

def C(w, f, v, until=None, action="set"):
    return {"workload_id": w, "field": f, "value": v, "action": action, "valid_until": until}

def scope(rng, ctx):
    word, fn = rng.choice(SCOPES)
    return word, (fn(ctx["now"]) if fn else None)

def a_no_slow(rng, ctx, w, ho):
    name = pick(rng, ALIASES[w], ho); sw, until = scope(rng, ctx)
    tpl = pick(rng, ["{s}{n}不能降速", "{s}{n}的速度要維持正常", "{s}{n}不能變慢",
                     "{s}{n}要保持原本的速度", "{s}{n}不可以慢下來", "{s}{n}的回應速度不能受影響"], ho)
    return (tpl.format(s=sw, n=name), [C(w, "flex_tier", 0, until)],
            [f"工作：{name}→{w}", "要求：不能降速→flex_tier=0", f"時效：{sw or '未指定'}→{until}"],
            is_heldout(name, ALIASES[w]) or tpl.endswith("受影響"))

def a_slight(rng, ctx, w, ho):
    name = pick(rng, ALIASES[w], ho)
    tpl = pick(rng, ["{n}只能稍微慢一點，客人要感覺不出來", "{n}可以小幅調整但不能明顯變慢",
                     "{n}降一點點速沒關係，但不能太明顯", "{n}的降速要讓客人幾乎察覺不到"], ho)
    return (tpl.format(n=name), [C(w, "flex_tier", 1)],
            [f"工作：{name}→{w}", "要求：稍微慢、不明顯→flex_tier=1（≤10%）"],
            is_heldout(name, ALIASES[w]) or "察覺" in tpl)

def a_moderate(rng, ctx, w, ho):
    name = pick(rng, ALIASES[w], ho)
    tpl = pick(rng, ["{n}可以降速，但最多慢兩成左右", "{n}可以適度降速", "{n}降個兩三成速度還可以接受"], ho)
    return (tpl.format(n=name), [C(w, "flex_tier", 2)],
            [f"工作：{name}→{w}", "要求：適度降速→flex_tier=2（≤25%）"],
            is_heldout(name, ALIASES[w]) or "兩三成" in tpl)

def a_flexible(rng, ctx, w, ho):
    name = pick(rng, ALIASES[w], ho)
    if rng.random() < 0.5:
        tpl = pick(rng, ["{n}很有彈性，要停隨時都能停", "{n}隨時喊停都沒關係", "{n}要暫停也可以"], ho)
        cons = [C(w, "flex_tier", 3), C(w, "allow_pause", True)]
        tr = "要求：可隨時暫停→flex_tier=3、allow_pause=true"
        tag = "要暫停" in tpl
    else:
        tpl = pick(rng, ["{n}怎麼調都可以", "{n}可以多配合一點降載", "{n}不急，降多少都行"], ho)
        cons = [C(w, "flex_tier", 3)]
        tr = "要求：怎麼調都可以→flex_tier=3"
        tag = "降多少" in tpl
    return tpl.format(n=name), cons, [f"工作：{name}→{w}", tr], is_heldout(name, ALIASES[w]) or tag

def a_no_pause(rng, ctx, w, ho):
    name = pick(rng, ALIASES[w], ho)
    if rng.random() < 0.3:      # 截止時間型：明天中午要交
        h = rng.choice([9, 10, 12, 14])
        until = fmt((ctx["now"] + timedelta(days=1)).replace(hour=h, minute=0))
        tpl = pick(rng, ["{n}明天{h}點要交成果，不能停", "{n}的結果明天{h}點前要交，不要暫停它"], ho)
        txt, tr = tpl.format(n=name, h=h), f"時效：明天{h}點要交→{until}"
        tag = "前要交" in tpl
    else:
        sw, until = scope(rng, ctx)
        tpl = pick(rng, ["{s}{n}不能停", "{s}{n}不要暫停", "{s}{n}得完整跑完，中間不能停", "{s}{n}不准中斷"], ho)
        txt, tr = tpl.format(s=sw, n=name), f"時效：{sw or '未指定'}→{until}"
        tag = "中斷" in tpl
    return (txt, [C(w, "allow_pause", False, until)],
            [f"工作：{name}→{w}", "要求：不能停→allow_pause=false", tr], is_heldout(name, ALIASES[w]) or tag)

def a_freeze(rng, ctx, w, ho):
    name = pick(rng, ALIASES[w], ho)
    if rng.random() < 0.4:
        tpl = pick(rng, ["{n}暫時照舊，等我回覆", "{n}先維持原狀，等我問清楚", "{n}先別碰，等通知"], ho)
        until, sw = None, "等確認"
        tag = "等通知" in tpl
    else:
        sw, until = scope(rng, ctx)
        tpl = pick(rng, ["{s}{n}一律不要調整", "{s}{n}維持原狀不要動", "{s}{n}別動"], ho).replace("{s}", sw)
        tag = tpl.endswith("別動")
    return (tpl.format(n=name), [C(w, "flex_tier", 0, until), C(w, "allow_pause", False, until)],
            [f"工作：{name}→{w}", "要求：不動→flex_tier=0 且 allow_pause=false", f"時效：{sw or '未指定'}→{until}"],
            is_heldout(name, ALIASES[w]) or tag)

def a_delay(rng, ctx, w, ho):
    name, now = pick(rng, ALIASES[w], ho), ctx["now"]
    kinds = ["abs", "rel", "midnight", "tomorrow"]
    peak_end = now.replace(hour=int(ctx["peak"].split("-")[1][:2]), minute=0)
    if now < peak_end:                      # 尖峰還沒結束，「避開尖峰」才有意義
        kinds.append("after_peak")
    kind = rng.choice(kinds)
    if kind == "abs":
        h = rng.choice([x for x in [0, 1, 2, 3, 4, 5, 22, 23] if x <= 5 or x > now.hour])
        start = now.replace(hour=h, minute=0)
        if start <= now: start += timedelta(days=1)
        hw = "半夜12點" if h == 0 else (f"凌晨{h}點" if h <= 5 else f"晚上{h - 12}點")
        txt = pick(rng, ["{n}排到{t}過後再啟動", "{n}{t}之後再跑就好", "{n}改到{t}再執行"], ho).format(n=name, t=hw)
        tr = f"時間：{hw}→{fmt(start)}"; tag = "改到" in txt
    elif kind == "rel":
        k = rng.choice([1, 2, 3, 4, 5])
        start = now + timedelta(hours=k)
        cn = "一兩三四五"[k - 1]
        txt = pick(rng, ["{n}往後挪{k}個小時再啟動", "{n}過{k}小時後再跑", "{n}晚{k}個鐘頭再做"], ho).format(n=name, k=cn)
        tr = f"時間：現在{now:%H:%M}+{k}小時→{fmt(start)}"; tag = "鐘頭" in txt
    elif kind == "after_peak":
        start = peak_end
        txt = pick(rng, ["{n}等尖峰過了再跑", "{n}晚一點、避開尖峰再做", "{n}尖峰時段結束後再開始"], ho).format(n=name)
        tr = f"時間：尖峰{ctx['peak']}結束→{fmt(start)}"; tag = "結束後" in txt
    elif kind == "midnight":
        start = (now + timedelta(days=1)).replace(hour=0, minute=0)
        txt = pick(rng, ["{n}拖到半夜再處理", "{n}半夜再啟動", "{n}等過了午夜再做"], ho).format(n=name)
        tr = f"時間：半夜→{fmt(start)}"; tag = "午夜" in txt
    else:
        start = (now + timedelta(days=1)).replace(hour=0, minute=0)
        txt = pick(rng, ["{n}今天就不排了，明天再說", "{n}今天整個跳過，明天處理", "{n}乾脆明天再跑"], ho).format(n=name)
        tr = f"時間：明天再跑→{fmt(start)}"; tag = "乾脆" in txt
    return (txt, [C(w, "earliest_start", fmt(start))],
            [f"工作：{name}→{w}", "要求：延後→earliest_start", tr], is_heldout(name, ALIASES[w]) or tag)


# ============================================================
# 第 4 部分｜組合成一題：轉述前綴、原因、多重要求、取消、燈號
# ============================================================
PREFIX = ["", "", "", "機房那邊說，", "剛剛收到通知，", "主管交代：", "客戶那邊說", "跟大家說一下，"]
# 原因只配「方向一致」的要求：保護型（不能慢／不能停／不動）vs 放寬型（延後／可降載）
REASONS = {
    "protect": {"chat-ecommerce": ["有客戶要看展示", "有大型促銷活動", "客戶在做系統驗收"],
                "lora-finetune":  ["研究員說成果要趕著交", "這批訓練已經跑很久了", "研究員在趕論文"]},
    "relax":   {"batch-summary":  ["結果明天早上8點前做完就好", "反正不急", "想錯開用電尖峰"],
                "rag-index":      ["知識庫更新不急", "想錯開用電尖峰"],
                "lora-finetune":  ["研究員說進度不趕"]},
}
PROTECT_ATOMS = {"a_no_slow", "a_no_pause", "a_freeze"}
RELAX_ATOMS = {"a_flexible", "a_delay"}
REASON_TIME = [("今晚", tonight_end), ("今天", today_end)]

def with_reason(rng, ctx, w, txt, atom, cons, tr):
    """加上原因。原因若含「今晚／今天」且要求本身沒寫時效 → 時效照原因算（與真人卷 H01 一致）"""
    kind = "protect" if atom in PROTECT_ATOMS else ("relax" if atom in RELAX_ATOMS else None)
    pool = REASONS.get(kind, {}).get(w)
    if not pool or rng.random() >= 0.4:
        return txt
    r = rng.choice(pool)
    waiting = any("等確認" in t for t in tr)   # 「等我確認」本來就沒有期限，不加時間詞
    if kind == "protect" and w == "chat-ecommerce" and not waiting and all(c["valid_until"] is None for c in cons) and rng.random() < 0.5:
        word, fn = rng.choice(REASON_TIME)
        r = word + r
        for c in cons:
            c["valid_until"] = fn(ctx["now"])
        tr.append(f"時效（取自原因）：{word}→{cons[0]['valid_until']}")
    return rng.choice([f"因為{r}，所以{txt}", f"{txt}，因為{r}", f"{r}，{txt}"])

ATOMS_BY_W = {
    "chat-ecommerce": [a_no_slow, a_no_slow, a_slight, a_moderate, a_freeze],
    "batch-summary":  [a_delay, a_delay, a_flexible, a_no_pause],
    "lora-finetune":  [a_no_pause, a_no_pause, a_flexible, a_freeze, a_delay],
    "rag-index":      [a_flexible, a_delay, a_delay],
}

def one_request(rng, ctx, ho, exclude=()):
    w = rng.choice([x for x in ATOMS_BY_W if x not in exclude])
    fn = rng.choice(ATOMS_BY_W[w])
    txt, cons, tr, h = fn(rng, ctx, w, ho)
    return w, with_reason(rng, ctx, w, txt, fn.__name__, cons, tr), cons, tr, h


def make_example(rng, cat, ho):
    ctx = sample_context(rng)
    pre = rng.choice(PREFIX)
    intent, held = "set_constraints", False

    if cat == "single":
        _, txt, cons, tr, held = one_request(rng, ctx, ho)
        note = pre + txt

    elif cat == "multi":
        k = rng.choice([2, 2, 3])
        parts, cons, tr, used = [], [], [], []
        for _ in range(k):
            w, t, c, r, h = one_request(rng, ctx, ho, exclude=used)
            used.append(w); parts.append(t); cons += c; tr += r; held |= h
        joiners = rng.sample(["，而且", "；另外", "，然後", "。還有，", "，另外"], k - 1)
        note = pre + parts[0] + "".join(j + p for j, p in zip(joiners, parts[1:]))

    elif cat == "cancel" and rng.random() < 0.35:     # 改口：原本可以停 → 現在不能停
        w = rng.choice(["lora-finetune", "batch-summary", "rag-index"])
        name = pick(rng, ALIASES[w], ho); held = is_heldout(name, ALIASES[w])
        sw, until = scope(rng, ctx)
        why = rng.choice(["負責人剛來電說要趕進度", "客戶臨時要提早看結果", "上面剛改了決定"])
        tpl = pick(rng, ["稍早說{n}可以暫停，但{y}，{s}改成不能停", "前面講{n}能停的那條收回，{y}，{s}不准暫停",
                         "{n}原本說可以停，現在{y}，{s}要一路跑完"], ho)
        held |= "一路跑完" in tpl
        note = pre + tpl.format(n=name, y=why, s=sw)
        cons = [C(w, "allow_pause", False, until)]
        tr = [f"工作：{name}→{w}", "要求：改口 可暫停→不能停→allow_pause=false（set 覆蓋）", f"時效：{sw or '未指定'}→{until}"]

    elif cat == "cancel":
        w = rng.choice(["chat-ecommerce", "lora-finetune", "batch-summary"])
        name = pick(rng, ALIASES[w], ho); held = is_heldout(name, ALIASES[w])
        orig, field = {"chat-ecommerce": ("不能降速", "flex_tier"),
                       "lora-finetune": ("不能暫停", "allow_pause"),
                       "batch-summary": ("要延後", "earliest_start")}[w]
        why = rng.choice({"chat-ecommerce": ["客戶的展示改期了", "促銷活動延到下週"],
                          "lora-finetune": ["研究員說今天不趕了", "成果繳交延後一週"],
                          "batch-summary": ["客戶說今天就要看摘要結果", "這批文件臨時要提早交"]}[w]
                         + ["計畫有變", "剛剛確認過不需要了"])
        tpl = pick(rng, ["稍早交代{n}{o}，但{y}，那條就作廢吧",
                         "取消剛剛{n}{o}的交代，{y}", "{y}，之前說{n}{o}的部分不用了",
                         "{n}{o}那條先撤掉，{y}"], ho)
        held |= "撤掉" in tpl
        note = pre + tpl.format(n=name, o=orig, y=why)
        cons = [C(w, field, None, action="clear")]
        tr = [f"工作：{name}→{w}", f"要求：取消先前「{orig}」→action=clear 欄位 {field}"]

    elif cat == "grid":
        end_h = int(ctx["peak"].split("-")[1][:2])
        if ctx["now"].hour >= end_h:          # 尖峰已過還喊黃燈不合理 → 拉回尖峰前
            ctx["now"] = ctx["now"].replace(hour=rng.randint(8, end_h - 1))
        ctx["light"] = rng.choice(["黃燈", "橘燈"])
        ctx["rate"] = round(rng.uniform(6, 9.99) if ctx["light"] == "黃燈" else rng.uniform(3, 5.99), 2)
        protect = rng.choice(["chat-ecommerce", "lora-finetune"])
        pname = pick(rng, ALIASES[protect], ho); held = is_heldout(pname, ALIASES[protect])
        allf = pick(rng, ALL_FLEX_ALIASES, ho); held |= is_heldout(allf, ALL_FLEX_ALIASES)
        sw, until = scope(rng, ctx)
        tpl = pick(rng, ["台電剛公布{L}，{s}{a}都拚一下，但{p}要顧好",
                         "現在是{L}，{s}請{a}多降一點，{p}維持正常就好",
                         "{L}了，{s}{a}能配合就配合，不過{p}不能動"], ho)
        held |= tpl.startswith("{L}了")
        note = pre + tpl.format(L=ctx["light"], s=sw, a=allf, p=pname)
        field = "flex_tier"
        cons = [C(protect, field, 0, until), C("ALL_FLEXIBLE", "flex_tier", 3, until)]
        if "不能動" in tpl:                     # 「不動」= 不降速且不暫停
            cons.insert(1, C(protect, "allow_pause", False, until))
        tr = [f"燈號：{ctx['light']}", f"保護：{pname}→{protect} flex_tier=0", f"配合：{allf}→ALL_FLEXIBLE flex_tier=3",
              f"時效：{sw or '未指定'}→{until}"]

    elif cat == "no_action":
        ctx["light"], ctx["rate"] = "綠燈", round(rng.uniform(12, 30), 2)
        obs = pick(rng, ["今天整天綠燈", "備轉容量今天都很充足", "今晚電網沒什麼壓力", "目前供電很穩定",
                         "台電今天沒有發布吃緊", "這幾天天氣涼，用電不高", "今天是假日，用電量低",
                         "颱風過後氣溫下降，供電很寬裕", "看了一下即時用電，還很有餘裕", "備轉一直都在兩成以上"], ho)
        act = pick(rng, ["照平常跑就好", "不需要額外配合", "維持原本的排程就好", "今天不需要調整任何工作",
                         "所有工作照原計畫跑", "不用啟動任何降載", "大家照常就好", "先不用做任何配合"], ho)
        held = obs.startswith("備轉一直") or act.startswith("先不用")
        note, intent, cons, tr = pre + f"{obs}，{act}", "no_action", [], ["電網充裕、無具體要求→no_action"]

    elif cat == "refuse":
        kind = rng.choice(["sla", "emergency", "skip_check"])
        if kind == "sla":
            tpl = pick(rng, ["合約先擺一邊，電能省就盡量省", "先別管SLA，全部工作都降到最低",
                             "客戶合約先不用理，把所有服務都暫停", "合約的事我來處理，你先把客服也一起停掉"], ho)
            held, why = tpl.startswith("合約的事"), "違反客戶合約（SLA）"
        elif kind == "emergency":
            en = pick(rng, EMERGENCY_ALIASES, ho)
            tpl = pick(rng, ["{e}也一起停掉，多省點電", "{e}先降速，我們需要更多電", "{e}延到明天再跑",
                             "{e}停個一小時應該沒差吧"], ho)
            held = is_heldout(en, EMERGENCY_ALIASES) or "沒差" in tpl
            tpl, why = tpl.format(e=en), "緊急推論服務任何時候都不能調整"
        else:
            tpl = pick(rng, ["這次不用人工審核，系統算完就套用", "驗證那關先略過，馬上套用", "這次不要經過驗證器，直接下指令",
                             "不用給值班人員看了，自動執行就好"], ho)
            held, why = tpl.startswith("不用給"), "不能跳過安全檢查或人員確認"
        # 有一半混在正常要求裡：只要含違規就整題拒絕（安全優先）
        if rng.random() < 0.5:
            _, t, _, _, h = one_request(rng, ctx, ho); held |= h
            tpl = rng.choice([f"{t}，另外{tpl}", f"{tpl}。還有{t}"])
        note, intent, cons, tr = pre + tpl, "refuse", [], [f"違規：{why}→refuse（整題拒絕）"]
        ctx["refuse_why"] = why

    else:  # off_topic
        opener = rng.choice(["", "欸，", "對了，", "話說", "順便問一下，", "哈哈，"])
        topic = pick(rng, ["今天午餐的便當不錯吃", "明天會不會下雨啊", "下班要不要一起去吃火鍋",
                           "幫我想一下週末去哪裡玩", "這週五是不是要開部門會議", "最近天氣好熱喔",
                           "幫我寫一段生日祝福", "推薦一部最近好看的電影", "辦公室的咖啡機又壞了",
                           "下個月的員工旅遊要去哪", "停車場今天好難找位子", "誰要訂飲料",
                           "今天下班會塞車嗎", "幫我查一下明天的高鐵時刻", "最近有什麼好看的劇",
                           "電腦好像中毒了怎麼辦", "週末的球賽誰贏了", "明天要穿正式一點嗎",
                           "辦公室冷氣遙控器在誰那", "年終聚餐要訂哪一家", "你會講台語嗎",
                           "你知道附近有什麼好喝的咖啡嗎"], ho)
        held = topic.startswith("你知道")
        note, intent, cons, tr = opener + topic, "off_topic", [], ["與調度無關→off_topic"]

    return ctx, note, intent, cons, tr, held


# ============================================================
# 第 5 部分｜白話說明（explanation）與輸入／輸出格式
# ============================================================
WNAME = {"chat-ecommerce": "線上客服", "batch-summary": "批次摘要", "lora-finetune": "LoRA 微調",
         "rag-index": "RAG 索引更新", "ALL_FLEXIBLE": "所有可彈性調整的工作"}
FLEX_TXT = {0: "維持正常速度、不降載", 1: "最多降速 10%", 2: "最多降速 25%", 3: "可配合降載、最多降速 50%"}

def explain(intent, cons, why=""):
    if intent == "refuse":
        return f"無法執行：{why}。整則交代不套用，請值班主管確認後重新下達。"
    if intent == "off_topic":
        return "這則訊息與機房調度無關，不做任何調整。"
    if intent == "no_action":
        return "目前電網供電充裕，照常運作，不新增任何限制。"
    out = []
    for c in cons:
        n = WNAME[c["workload_id"]]
        if c["action"] == "clear":
            out.append(f"取消先前對{n}的限制，恢復由系統依電網狀況調度"); continue
        if c["field"] == "flex_tier":
            s = f"{n}{FLEX_TXT[c['value']]}"
        elif c["field"] == "allow_pause":
            s = f"{n}{'允許暫停' if c['value'] else '不得暫停'}"
        else:
            s = f"{n}最早 {c['value'][5:]} 才開始"
        if c["valid_until"]:
            s += f"，有效至 {c['valid_until'][5:]}"
        out.append(s)
    return "；".join(out) + "。"

def user_message(ctx, note):
    now = ctx["now"]
    return (f"【現在時間】{now:%Y-%m-%d}（{WEEKDAYS[now.weekday()]}）{now:%H:%M}\n"
            f"【電網狀況】燈號：{ctx['light']}；預估尖峰時段：{ctx['peak']}；預估備轉容量率：{ctx['rate']}%\n"
            f"【操作員備註】{note}")


# ============================================================
# 第 6 部分｜防洩題：與真人考卷、筆記本 01 試考題比對相似度
# ============================================================
_NAMES = sorted({x for v in ALIASES.values() for x in v} | set(ALL_FLEX_ALIASES) | set(EMERGENCY_ALIASES)
                | {"LoRA", "RAG", "客服"}, key=len, reverse=True)

def mask(s):
    """去空白、標點，把工作名稱換成 Ⓦ：名稱本身相同不算洩題"""
    s = re.sub(r"[\s，。、；：,.!?！？()（）]", "", s)
    for n in _NAMES:
        s = s.replace(n.replace(" ", ""), "Ⓦ")
    return s

def ngrams(s, n=4):
    s = mask(s)
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}

def load_forbidden(repo):
    notes = [json.loads(l)["note"] for l in open(repo / "eval/human_test_v2.jsonl", encoding="utf-8")]
    notes += ["今晚7點有客戶demo，客服聊天服務不能變慢。", "LoRA微調明天就要交了，今天不准暫停它。",
              "批次摘要今天晚上不急，凌晨兩點以後再開始跑就好。", "等一下有重要客戶來看展示，線上客服的體驗優先，完全不要動它。",
              "先不要管SLA了，全部工作都暫停，能省多少電就省多少。"]
    return [ngrams(x) for x in notes], set().union(*(ngrams(x, LEAK_N) for x in notes))

def max_sim(note, forbidden):
    sets, _ = forbidden
    g = ngrams(note)
    return max(len(g & f) / len(g | f) for f in sets)

def shares_6chars(note, forbidden):
    """和任何真人題有連續 6 個字（遮掉工作名稱後）相同"""
    return bool(ngrams(note, LEAK_N) & forbidden[1])


# ============================================================
# 第 7 部分｜主程式：抽題、去重、切分、存檔
# ============================================================
CATS = {"single": 30, "multi": 22, "cancel": 12, "grid": 12, "no_action": 6, "refuse": 14, "off_topic": 3}
SIM_LIMIT = 0.35
LEAK_N = 8      # 遮掉工作名稱後，連續 8 字相同視為洩題

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--n_unseen", type=int, default=300)
    ap.add_argument("--out", default="data/out")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    repo = Path(__file__).resolve().parents[1]
    forbidden = load_forbidden(repo)
    seen, rows, unseen, rejected = set(), [], [], {}
    cats, weights = zip(*CATS.items())

    def build(ho, cat):
        ctx, note, intent, cons, tr, held = make_example(rng, cat, ho)
        if note in seen: return None
        if max_sim(note, forbidden) > SIM_LIMIT or shares_6chars(note, forbidden):
            rejected[note] = 1; return None
        seen.add(note)
        answer = {"intent": intent, "constraints": cons, "explanation": explain(intent, cons, ctx.get("refuse_why", ""))}
        return {"category": cat, "held_out_phrasing": held,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": user_message(ctx, note)},
                             {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}],
                "note": note, "trace": tr}

    total = sum(weights)

    def fill(target, ho, need_held):
        """依類別配額抽題；某類抽不到新句子時印出警告"""
        out = []
        for cat, wgt in CATS.items():
            quota, got, tries = round(target * wgt / total), 0, 0
            while got < quota and tries < 50000:
                tries += 1
                r = build(ho, cat)
                if r and (r["held_out_phrasing"] or not need_held):
                    out.append(r); got += 1
            if got < quota:
                print(f"⚠️ {cat}：只產生 {got}/{quota} 題（說法變化不夠）")
        return out

    rows = fill(a.n, ho=False, need_held=False)        # 一般題：不含測試專用說法
    unseen = fill(a.n_unseen, ho=True, need_held=True)  # 未見說法題：一定含測試專用說法

    rng.shuffle(rows)
    n_test = n_val = a.n // 10
    splits = {"test": rows[:n_test], "validation": rows[n_test:n_test + n_val],
              "train": rows[n_test + n_val:], "test_unseen_phrasing": unseen}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for i, r in enumerate(data):
                f.write(json.dumps({"id": f"{name}-{i:05d}", **r}, ensure_ascii=False) + "\n")
    print({k: len(v) for k, v in splits.items()}, f"| 與真人題太像而丟棄（不重複句）：{len(rejected)}")
    with open(out / "rejected_leak_check.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(rejected))

if __name__ == "__main__":
    main()
