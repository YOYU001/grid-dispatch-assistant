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

- Yoyo 在 Colab T4 跑完筆記本 01：**5 題試考 5 項指標全部 100%**；Yoyo 人工檢查 explanation 認為 OK
  - 解讀：天花板效應（題少、題目乾淨），只證明底子好；合成資料必須以刁鑽題為主才比得出訓練效果
  - 套件：pip 警告 cudf 與 pyarrow 衝突，屬 Colab 預裝且不使用的套件，可忽略

- Yoyo 完成 20 張真人情境卡；發現：真人句子較長、常附原因與轉述（「資料中心那邊說…」），合成資料要納入這種說法
- 真人題暴露格式不足 → 制定**輸出格式 v2**（`docs/output_schema_v2.md`）：intent、action=clear、valid_until、ALL_FLEXIBLE、輸入附現在時間與電網狀況
- 依 Yoyo 確認製作標準答案 `eval/human_test_v2.jsonl`
- 看過 NAS 收集器資料（10/22–10/28，905 筆）：**一週全是綠燈**（備轉 11–46%）→ 黃／橘燈情境要靠夏季歷史資料

- 完成合成資料產生器 `data/generate.py`（規則老師、7 類題型、規則軌跡、未見說法測試集）
- 防洩題：發現部分說法取自 Yoyo 的真人題 → 改寫，並加「遮名稱後連續 8 字相同即丟棄」檢查
- QA（有條件通過）＋ code-review 審查後修正：原因與要求矛盾、原因裡的時間詞要算進時效、「不能動」漏標、類別配額、改口題、尖峰後喊黃燈、flex 說法對照表寫進規格

- Yoyo 確認卡 7 改為「不動」兩欄位；資料集擴大到 5,000 題，提高多件事、取消／改口、燈號、不需動作比例
  - v1 最終：train 3,954／validation 500／test 500／test_unseen_phrasing 394；與真人題最高相似 0.21

**下一步**
- 合成資料上傳為 HF 私人 dataset，再進入 SFT + LoRA 訓練
- 依試考結果設計合成資料格式（SLA 政策表、GridEvent、備註範本）
