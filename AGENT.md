# AGENT.md — 給任何 AI 協作者的專案說明

> 這份文件讓 Claude、Codex 或其他 AI 一打開就懂這個專案在做什麼、怎麼合作。
> 人類讀者也適用。最後更新：2026-11-02

## 1. 專案是什麼（一句話）

微調一個小型中文 LLM「電網調度秘書」，讓它：
1. 讀懂台電操作員的中文備註 → 轉成限制條件 JSON
2. 用白話中文解釋最佳化器的調度方案
3. 拒絕違反 SLA 的惡意指令

它**不做決策**。決策由規則排程器／OR-Tools 最佳化器負責，執行前一定經過安全驗證器與人員核准。

**定位（2026-11-02）：主角是 LLM，排程器是 Agent 呼叫的工具。** 本專案也用來實測多模型比較、各種訓練法、推論優化，以及 NVIDIA 工具與開源方案的差異。總規劃見 `docs/TRAINING_PLAN.md`。

母專案：「電網尖峰下 AI 資料中心 GPU 彈性負載調度輔助系統」（台電綜研所 PoC）。
同時作為 Hugging Face smol-course 第一章（SFT）的期末作業。

## 3. 工作規則（所有 AI 都要遵守）

| 規則 | 說明 |
|---|---|
| 小插播優先 | 訊息含「小插播」→ 停下手邊工作先回覆 |
| 語言 | 繁體中文為主；程式碼註解可中英並用 |
| TODO 檢核 | 每個任務先拆解成 TODO list，逐項打勾 |
| Context 管理 | 對話超過 40 萬 token → 提醒 compact 或開新視窗 |
| **先看進度圖** | 每次開始新進度或新規劃前，先讀 `graphify-out/GRAPH_REPORT.md`（不存在或過期就重建，見第 8 節）；快照在 `docs/graph/` |
| 進度紀錄 | 每次工作結束更新 `PROGRESS.md` |
| 待辦紀錄 | 重要但現在做不了的事寫進 `TODO.md` |
| Skills | 重複出現的流程做成 skill 備用 |
| 看懂程式架構 | 用 `graphify` skill（`.claude/skills/graphify`）：預設 `GRAPHIFY_QUERY_LOG_DISABLE=1 graphify update .` 純程式模式，不呼叫 LLM、資料不離開本機；改評分器等共用函式前先查所有副本 |
| 資料安全 | 台電內部資料**絕不**進這個 repo；只用公開資料與合成資料 |
| 儲存 | Colab 一律用 `colab-storage` skill：Drive 存檔＋HF Hub 私人 repo；NAS 用 Cloud Sync 從 Drive 備份 |
| 運算環境 | 目前用 **Colab 免費 T4**（fp16、無 bf16、約 15 GB）；程式要以 T4 為準。精度判斷用 `torch.cuda.get_device_capability()[0] >= 8`，**不要用** `is_bf16_supported()`（T4 會誤判） |

## 4. Subagent 分工

定義檔在 `.claude/agents/`。

| Agent | 模型 | 何時用 | 原則 |
|---|---|---|---|
| `research` | Sonnet | 查最新模型、資料集、套件版本、文件 | 一定附來源網址；不憑記憶 |
| `code-review` | Opus | 程式寫完、要跑在付費 GPU 之前 | 找 bug、浪費 GPU 的設定、資料洩漏 |
| `qa` | Opus | 評估結果出來後 | 沒看過訓練程式，獨立驗證成績，避免「自己改自己考卷」 |

## 5. 訓練流程（固定順序）

```
① 選模型（research agent 比較 model card）
② 規則排程器當老師 → 產生 3,000–5,000 題合成資料
③ 切 train / val / test ＋ 改寫、對抗、未見情境三組特殊考卷
④ 先考未訓練模型（基準成績）
⑤ SFT + LoRA 訓練（Trackio 監看 loss）
⑥ 再考一次：自訂考卷 + lighteval GSM8K（退化檢查）
⑦ code-review → qa 驗證
⑧ 推上 Hub（yoyu001/...）→ smol-course leaderboard PR
```

## 5.5 三版本對照實驗（思考方式）

| 版本 | 基底模型 | 輸出 |
|---|---|---|
| A 無思考 | Qwen3-4B-Instruct-2507 | 直接輸出 JSON |
| B 完整思考 | Qwen3-4B-Thinking-2507（使用前由 research agent 確認 repo 與授權） | `<think>` 推理草稿 + JSON |
| C 迷你思考 | Qwen3-4B-Instruct-2507 | JSON 第一欄 `"理由"`（一句話）+ 限制條件 |

- **控制變因**：同 4B 大小、同一批題目、同訓練超參數、同考卷、同生成設定；只改「思考方式」
- **推理內容來源**：規則排程器輸出「規則軌跡」，以模板轉成 C 的一句理由與 B 的完整草稿，保證推理與答案一致
- **指標**：正確率、JSON 格式正確率、刁鑽題正確率（改寫／多重指令／取消前令）、每題延遲、每題耗電（nvidia-smi）
- **順序**：A（同時交 smol-course 作業）→ C → B（B 輸出長，等 A100 再做）

## 5.6 多模型比較與七階段規劃

詳見 `docs/TRAINING_PLAN.md`。摘要：

| 組別 | 模型 | 硬體 |
|---|---|---|
| A 小型同級 | Qwen3-4B-Instruct-2507／NVIDIA-Nemotron-3-Nano-4B／第三個待選 | T4 |
| B 架構 | Nemotron-Nano-9B-v2（Mamba 混合）vs 同級 Transformer | A100 |
| C MoE | Qwen3-30B-A3B vs Nemotron-3-Nano-30B-A3B | A100 80GB |

階段：0 A 版管線 → 1 資料工程 → 2 多模型比較 → 3 訓練方式（LoRA／QLoRA／全參數／DPO／GRPO／蒸餾）→ 4 MoE＋NeMo → 5 推論優化（vLLM／TRT-LLM／NIM、FP16／FP8／INT4）→ 6 Agent＋RAG → 7 Docker／k8s／多 GPU

## 6. 關鍵決策紀錄

| 日期 | 決策 | 理由 |
|---|---|---|
| 2026-10-28 | 只訓練「LLM 助理」，不訓練決策器 | 決策交給規則／最佳化器，可驗證、可稽核 |
| 2026-10-28 | 選 Instruct 模型而非 Base | 只有幾千題，教「工作」不教「說話」 |
| 2026-10-28 | 備註用中文 | 貼近台電實務；需選中文強的模型 |
| 2026-10-28 | 先用 Colab 免費 T4 | Azure GPU 尚未申請到 |
| 2026-10-28 | GSM8K 當作業成績＋災難性遺忘檢查 | 符合 smol-course 提交規定 |
| 2026-10-28 | 基底模型 Qwen3-4B-Instruct-2507（備案 Qwen3.5-4B） | 中文強、Apache-2.0、無思考模式輸出乾淨 JSON、T4 可 QLoRA |
| 2026-10-28 | 筆記本 01 的 5 題試考題不得用於訓練或 few-shot | 避免資料洩漏 |
| 2026-10-28 | 訓練 A 無思考／B 完整思考／C 迷你思考三版本做對照 | 用數據決定思考模式是否值得（正確率 vs 速度 vs 耗電） |
| 2026-11-02 | 專案主角改為 LLM，排程器定位為工具 | 目標職缺（AI SW Eng、DL SA）核心是 LLM 訓練、推論、RAG、Agent |
| 2026-11-02 | 加入多模型比較（Qwen、Nemotron 等）與七階段訓練規劃 | 實測模型比較、訓練法、推論優化、NVIDIA vs 開源 |
| 2026-11-02 | 環境法規 LLM 延後到本專案階段 6 之後 | 一個做深的專案勝過兩個做淺的 |
| 2026-11-02 | 每次新進度／規劃前先看 graphify 進度圖；文件語意抽取（完整模式）已獲 Yoyo 同意用於進度圖 | Yoyo 要求 |

## 7. 資料夾結構（規劃中）

```
grid-dispatch-assistant/
├── AGENT.md          ← 本檔
├── PROGRESS.md       ← 每次進度
├── TODO.md           ← 未來待辦
├── .claude/agents/   ← subagent 定義
├── docs/             ← 規格、分析、TRAINING_PLAN.md
│   └── graph/        ← graphify 進度圖快照（graph.html、GRAPH_REPORT.md）
├── data/             ← 合成資料（產生腳本＋輸出）
├── scheduler/        ← 規則排程器（當老師）
├── train/            ← SFT + LoRA 訓練腳本、設定檔
├── eval/             ← 自訂考卷評估＋lighteval
└── notebooks/        ← 筆記本（每個階段一本：01 認識模型 → 02 上傳資料 → 03 評估 → 04 訓練）
    ├── *_kaggle.py／.ipynb ← Kaggle 版（目前主力；.py 為正本，.ipynb 由 tools/py2ipynb.py 產生）
    └── tools/py2ipynb.py   ← 分段 .py → .ipynb 轉換器
```

## 8. 給接手 AI 的開場步驟

0. 程式碼的正本在私人 repo `github.com/YOYU001/grid-dispatch-assistant`，先 clone 下來
1. 讀本檔 → 讀 `graphify-out/GRAPH_REPORT.md`（沒有就先讀 `docs/graph/GRAPH_REPORT.md` 快照，並依 graphify skill 重建）→ 讀 `PROGRESS.md` 最後一筆 → 讀 `TODO.md` → 讀 `docs/TRAINING_PLAN.md`
2. 確認目前在流程第幾步
3. 用 TODO list 列出這次要做的事，再開始
4. 工作結束：更新 `PROGRESS.md`／`TODO.md` → 執行 `/graphify . --update` 更新進度圖 → 把 `graph.html`、`GRAPH_REPORT.md` 複製到 `docs/graph/`
