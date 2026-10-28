# PROGRESS.md — 工作進度紀錄

> 每次工作結束新增一筆，最新的放最上面。
> 格式：日期｜做了什麼｜決定了什麼｜下一步

---

## 2026-10-28（第 1 次）

**做了什麼**
- 讀完專案規劃報告、台電提案附件、smol-course 第一章 7 個單元
- 釐清要訓練的模型：「電網調度秘書」（聽懂備註、白話說明、拒絕惡意指令），不做決策
- 規劃訓練專案架構（流程見 `AGENT.md` 第 5 節）
- 建立 `AGENT.md`、`PROGRESS.md`、`TODO.md`、三個 subagent 定義
- 派 research agent 比較中文小型 Instruct 模型 → 推薦 Qwen3-4B-Instruct-2507（備案 Qwen3.5-4B）
- 建立私人 GitHub repo `YOYU001/grid-dispatch-assistant` 並推上第一版（需安裝 Claude GitHub App 才能存取私人 repo）

**決定了什麼**
- 操作員備註用**中文** → SmolLM3 不支援中文，需換模型
- Azure GPU 尚未申請到 → 先用 **Colab 免費 T4**（注意：T4 不支援 bf16，要用 fp16）
- GSM8K 用 lighteval 評估，作為作業提交＋退化檢查

- 選定基底模型 **Qwen3-4B-Instruct-2507**（理由：中文強、Apache-2.0、無思考模式→輸出乾淨 JSON、T4 可 QLoRA）
- 完成 `notebooks/01_explore_model_baseline.ipynb`：讀 model card/config、看 chat template、載入模型、5 題試考
- code-review agent 審查後修正：拒絕判斷誤判、False/0 混淆、transformers 版本、時間格式、加「必要條件都有」指標

- 釐清模型角色：「翻譯官＋解說員」，不做決策、不直接控制 GPU
- 決定做三版本對照實驗：A 無思考、B 完整思考、C 迷你思考（設計見 AGENT.md 5.5 節）

- 介紹 Qwen3-4B Instruct／Thinking 雙胞胎與 FP8／FP4；TODO 記下 FP8 量測需 H100／L4
- 建立 Colab 儲存工具 `notebooks/snippets/colab_storage.py`（Drive＋HF Hub＋斷線接續），經 code-review 修正：半寫入 checkpoint、Drive 垃圾桶、公開 repo 防呆、台灣時區
- 建立 skill `.claude/skills/colab-storage`

**下一步**
- Yoyo 設定 Colab 密鑰 HF_TOKEN、Synology Cloud Sync
- Yoyo 在 Colab T4 執行筆記本 01（加上儲存段 A、B、C、E），回報成績單
- 依試考結果設計合成資料格式（SLA 政策表、GridEvent、備註範本）
