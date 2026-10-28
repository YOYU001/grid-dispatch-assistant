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

**下一步**
- 看 research agent 的模型比較，選定基底模型
- 開始設計合成資料格式（SLA 政策表、GridEvent、備註範本）
