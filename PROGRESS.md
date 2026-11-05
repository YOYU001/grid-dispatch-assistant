# PROGRESS.md — 工作進度紀錄

> 每次工作結束新增一筆，最新的放最上面。
> 格式：日期｜做了什麼｜決定了什麼｜下一步

---

## 2026-11-05（第 3 次）

**做了什麼**
- Kaggle 試跑筆記本 07（GSM8K，20 題）：原模型 75%、微調後 70%（差 1 題，不可信）；被 256 token 截斷 20%／30%
- 踩到兩個 Kaggle 套件坑並修正：litellm 需要 `diskcache`（鎖 5.6.3）；Kaggle 的 xxhash 4.0.2 不接受文字，lighteval 寫紀錄時報錯（鎖 3.5.0）
- 筆記本 07 更新：第 1 段鎖上述兩個版本並另開 Python 自動檢查；新增第 8.5 段「兩個模型都沒被截斷的題目」答對率與「答錯且被截斷」題數，寫進 summary；code-review 審查後把對帳 assert 改為只警告，避免 1 小時考完卻存不到 HF
- 教訓：換套件版本後 lighteval 快取編號會變、會重新作答

**決定了什麼**
- 正式版照 smol-course 標準設定（256 token）跑，另外用公平答對率分辨「變囉嗦」還是「變笨」

**下一步**
- Kaggle 正式跑 07（MAX_SAMPLES=None，建議 Save & Run All 背景執行）→ 結果交 qa agent → smol-course 提交

---

## 2026-11-02（第 2 次）

**做了什麼**
- 對照兩份 NVIDIA JD（AI Software Engineer ML/NLP/LLM、Deep Learning Solution Architect），逐條確認專案能否練到
- 查證 Nemotron 最新小模型：Nemotron 3 Nano 4B（model card 僅列英文）、Nemotron 3 Nano 30B-A3B（Mamba＋Transformer 混合 MoE）、Nemotron Nano 9B v2
- 建立 `docs/TRAINING_PLAN.md`：定位、JD 對照、A／B／C 三組模型比較、七階段規劃、時程與硬體
- 更新 `AGENT.md`（定位、5.6 節、先看進度圖規則、開場與收尾步驟）、`TODO.md`（依階段重排）
- 用 graphify（完整模式，含文件語意抽取）畫出專案進度關係圖，快照存 `docs/graph/`

**決定了什麼**
- 主角是 LLM，排程器是 Agent 呼叫的工具
- 多模型比較：A 組 Qwen3-4B／Nemotron-3-Nano-4B／第三個待選（T4）；B 組架構、C 組 MoE 等 A100
- 環境法規 LLM 延後到本專案階段 6 之後
- 每次新進度／規劃前先看 graphify 進度圖

- 筆記本 05 試跑（每份 20 題）：exact test 100%／unseen 85%／真人 75%（基準 19.6／19.8／30%）；錯題 8 題，含 1 題誤拒並自編規則。分析見 `docs/eval_05_trial_analysis.md`
- ✅ 筆記本 05 正式重考（914 題）：exact test 98.6%／unseen 81.2%／真人 75%（基準 19.6／19.8／30%）；拒絕召回 96.6%、誤拒 0.1%；本地重算與 Kaggle 0 差異。86 題錯題中 74 題在 unseen，集中在「不動」「可以暫停」兩條展開規則與相對時間；2 題把「跳過人員確認」照做。分析見 `docs/eval_05_full_analysis.md`
- 筆記本 06（合併外掛推完整模型 `yoyu001/grid-dispatch-A-instruct-merged-v1`）與 07（lighteval 0.13.0 GSM8K，原模型 vs 微調後）：直接讀 lighteval 原始碼確認新版任務名為 `gsm8k|0`（課程範例的 `lighteval|gsm8k|0|0` 已過時）、生成長度寫死 256 token；code-review 審查後修正：合併檢查改用一致率 ≥99.9%、跳過舊結果前比對題數與模型版本、原模型鎖版本、chat template 一致檢查、截斷率檢查、確認 repo 私人

**下一步**
- Kaggle 跑 06 → 07 試跑（MAX_SAMPLES=20）→ 07 正式 → smol-course 提交（提交前 qa agent 驗證）
- 資料 v2（階段 1），依 `docs/eval_05_full_analysis.md` 的建議補說法
- 階段 1：資料 v2（去重、改寫＋規則驗證、資料量實驗）
- research agent：確認 Nemotron-3-Nano-4B 在 T4 能否 QLoRA、選 A 組第三個模型

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

- ✅ 資料集已上傳為 HF 私人 dataset **`yoyu001/grid-dispatch-notes-zh`**（筆記本 02），下載回來驗證題數一致；真人考卷不上傳
  - Colab 小插曲：Drive 網頁上傳未出現在掛載路徑 → 改用 `files.upload()` 直接解壓到 Drive

- 2026-10-29：完成筆記本 03（v2 評分器：pure_json／json_ok／intent_ok／core_ok／exact_ok、拒絕召回率、誤拒率、fp16 溢位檢查；批次推論可斷線接續），本地單元測試 10 項通過，經 code-review 修正 bf16 誤判等問題

- ✅ 基準成績（Kaggle T4，Colab GPU 額度用完改用 Kaggle）：exact test 19.6%／unseen 19.8%／真人 30%；JSON 格式 100%；拒絕召回 75.2%、誤拒 9.4%。分析見 `docs/baseline_03_analysis.md`

- 筆記本 04（QLoRA SFT，Kaggle 三段式：trial → trial_full → full）：research 查證 API、code-review＋QA 審核修正；本地以迷你 Qwen3（CPU、無 4-bit）煙霧測試 SFTConfig 參數、只對答案算 loss、checkpoint 接續、存 adapter 皆通過（trl 1.16／transformers 5.21）

- 2026-10-29：筆記本 04 試跑
  - 修正：新版套件把 LoRA 外掛建成 bf16，fp16 GradScaler 報錯 → 第 7 段把可訓練參數轉 fp32（504 組）
  - 教訓：第 5～7 段重跑必須 Restart，否則 LoRA 會被掛兩次
  - trial（10 步）：loss 1.53→1.14；每步 19.7 秒；GPU 峰值 5.6 GB（題目中位數 363、最長 720 tokens）
  - trial_full（20 步）：train loss→0.35、eval loss 0.91→0.38（未過擬合）；eval／push／上傳流程正常
  - 正式訓練預估約 3 小時（496 步）；維持 batch 4×4 不改

- 2026-10-30：✅ A 版正式訓練完成（Kaggle T4，496 步，163 分鐘，GPU 峰值 5.9 GB）
  - 外掛：HF 私人 `yoyu001/grid-dispatch-A-instruct-qlora-v1`；執行紀錄 `notebooks/runs/04_A-full-v1.ipynb`
  - train loss 約 0.0001、eval loss 0.0003（第 60 步左右就已收斂）
  - 解讀：合成資料由模板產生、可預測性極高，模型幾乎學會了產生器本身；eval 也來自同一產生器，所以低 loss 不代表真實能力 → 要看 unseen 與真人題
  - 待驗證：2 遍可能過多（消融實驗：0.25／0.5／1 遍）；GSM8K 是否退化

- 2026-10-30：完成筆記本 05（微調後重考，13 段）：鎖定外掛檔指紋與資料版本、讀 run_config 確認最終版外掛、關掉外掛重考 32 題做環境檢查、合併前後一致性檢查、與基準同評分器比較、錯題分類；本地以基準作答驗證可重現 19.6／19.8／30%；code-review 審查後修正

- 2026-10-30：新增 skill `graphify`（Graphify-Labs/graphify，PyPI `graphifyy` 0.9.97，Apache-2.0）：把 repo 建成知識圖譜，可查函式關係與副本；加上本專案規則（預設純程式模式、不設 Gemini key、不碰台電內部文件、關閉查詢紀錄）；`graphify-out/` 列入 .gitignore。本 repo 測試：222 節點、326 條關係、16 群
- 筆記本 05：Kaggle 上 peft 0.21 與預裝 torchao 0.10 衝突 → 第 1 段移除 torchao

**下一步**
- 筆記本 05：Kaggle 試跑（SAMPLE_N=20）→ 正式重考（test／unseen／真人 20 題），與基準比較
- 訓練後重考 ＋ lighteval GSM8K 退化檢查 → smol-course leaderboard
