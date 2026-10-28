# AGENT.md — 給任何 AI 協作者的專案說明

> 這份文件讓 Claude、Codex 或其他 AI 一打開就懂這個專案在做什麼、怎麼合作。
> 人類讀者也適用。最後更新：2026-10-28

## 1. 專案是什麼（一句話）

微調一個小型中文 LLM「電網調度秘書」，讓它：
1. 讀懂台電操作員的中文備註 → 轉成限制條件 JSON
2. 用白話中文解釋最佳化器的調度方案
3. 拒絕違反 SLA 的惡意指令

它**不做決策**。決策由規則排程器／OR-Tools 最佳化器負責，執行前一定經過安全驗證器與人員核准。

母專案：「電網尖峰下 AI 資料中心 GPU 彈性負載調度輔助系統」（台電綜研所 PoC）。
同時作為 Hugging Face smol-course 第一章（SFT）的期末作業。

## 3. 工作規則（所有 AI 都要遵守）

| 規則 | 說明 |
|---|---|
| 小插播優先 | 訊息含「小插播」→ 停下手邊工作先回覆 |
| 語言 | 繁體中文為主；程式碼註解可中英並用 |
| TODO 檢核 | 每個任務先拆解成 TODO list，逐項打勾 |
| Context 管理 | 對話超過 40 萬 token → 提醒 compact 或開新視窗 |
| 進度紀錄 | 每次工作結束更新 `PROGRESS.md` |
| 待辦紀錄 | 重要但現在做不了的事寫進 `TODO.md` |
| Skills | 重複出現的流程做成 skill 備用 |
| 資料安全 | 台電內部資料**絕不**進這個 repo；只用公開資料與合成資料 |

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

## 6. 關鍵決策紀錄

| 日期 | 決策 | 理由 |
|---|---|---|
| 2026-10-28 | 只訓練「LLM 助理」，不訓練決策器 | 決策交給規則／最佳化器，可驗證、可稽核 |
| 2026-10-28 | 選 Instruct 模型而非 Base | 只有幾千題，教「工作」不教「說話」 |
| 2026-10-28 | 備註用中文 | 貼近台電實務；需選中文強的模型 |
| 2026-10-28 | 先用 Colab 免費 T4 | Azure GPU 尚未申請到 |
| 2026-10-28 | GSM8K 當作業成績＋災難性遺忘檢查 | 符合 smol-course 提交規定 |

## 7. 資料夾結構（規劃中）

```
grid-dispatch-assistant/
├── AGENT.md          ← 本檔
├── PROGRESS.md       ← 每次進度
├── TODO.md           ← 未來待辦
├── .claude/agents/   ← subagent 定義
├── data/             ← 合成資料（產生腳本＋輸出）
├── scheduler/        ← 規則排程器（當老師）
├── train/            ← SFT + LoRA 訓練腳本、設定檔
├── eval/             ← 自訂考卷評估＋lighteval
└── notebooks/        ← Colab 筆記本
```

## 8. 給接手 AI 的開場步驟

0. 程式碼的正本在私人 repo `github.com/YOYU001/grid-dispatch-assistant`，先 clone 下來
1. 讀本檔 → 讀 `PROGRESS.md` 最後一筆 → 讀 `TODO.md`
2. 確認目前在流程第幾步
3. 用 TODO list 列出這次要做的事，再開始
