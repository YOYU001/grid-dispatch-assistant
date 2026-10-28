---
language:
- zh
license: cc-by-4.0
pretty_name: Grid Dispatch Notes (zh-TW, synthetic)
size_categories:
- 1K<n<10K
task_categories:
- text-generation
tags:
- synthetic
- energy
- power-grid
- data-center
- demand-response
- structured-output
- traditional-chinese
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: validation.jsonl
  - split: test
    path: test.jsonl
  - split: test_unseen_phrasing
    path: test_unseen_phrasing.jsonl
---

# Grid Dispatch Notes（電網調度秘書合成資料集）v1

訓練小型 LLM 擔任 AI 資料中心的「電網調度秘書」：讀懂值班人員的**繁體中文備註**，轉成調度系統可讀的**限制條件 JSON**；遇到違規要求（違反客戶合約、調整緊急推論、跳過安全檢查）一律拒絕。

秘書**不做決策**。決策由規則／最佳化器負責，執行前須經安全驗證器與人員核准。

## 資料怎麼來的

**全部為合成資料**。由「規則老師」先抽出結構化情境（要對哪個工作下什麼限制），再把它說成一句中文備註，因此標準答案由規則直接算出，保證與情境一致。

情境參數參考真實來源：

| 參數 | 來源 |
|---|---|
| Flex 0–3（最多降速 0%／10%／25%／50%） | Colangelo et al. 2025, arXiv:2507.00909（鳳凰城實驗） |
| 尖峰時段 | 台電「今日系統供需狀況」（data.gov.tw 162595），作者自建收集器 2026-10 實測 |
| 燈號門檻 | 台電供電警戒燈號公開說明（綠 ≥10%、黃 6–10%、橘 <6%） |

> 對外說法：以真實電網事件與已發表的彈性分級**參數化的合成情境**；備註文字為合成，**不是**真實操作員紀錄。

## 欄位

| 欄位 | 說明 |
|---|---|
| `id` | 題號 |
| `category` | `single`／`multi`／`cancel`／`grid`／`no_action`／`refuse`／`off_topic` |
| `held_out_phrasing` | 是否使用訓練集沒出現過的說法（只在 `test_unseen_phrasing` 為 true） |
| `messages` | 對話格式（system／user／assistant），可直接給 TRL `SFTTrainer` |
| `note` | 操作員備註原文 |
| `trace` | 規則軌跡：老師用了哪些規則得出答案（供「迷你思考」與「完整思考」版本使用） |

輸出格式見 GitHub repo `docs/output_schema_v2.md`。

## 切分

| split | 題數 | 用途 |
|---|---|---|
| train | 3,954 | 訓練 |
| validation | 500 | 訓練中監看過擬合 |
| test | 500 | 合成考卷（句型與 train 相似，偏「熟悉度」） |
| test_unseen_phrasing | 394 | **只用訓練沒看過的說法**，考泛化能力 |

另有 20 題**真人撰寫**考卷，不放在本資料集，以免被誤用於訓練。

## 品質控管

- 防洩題：遮掉工作名稱後，與真人考卷有連續 8 字相同即丟棄；與真人題最高 4-gram Jaccard 相似度 0.21
- 獨立 QA agent（未讀程式碼，只依規格判讀）兩輪抽樣共 87 題，最終無標註錯誤
- code-review agent 審查產生器邏輯

## 已知限制

- 句型來自範本，變化有限；`test` 分數偏樂觀，泛化能力請看 `test_unseen_phrasing` 與真人考卷
- `off_topic` 題數少（unseen 僅 6 題），該類分數不穩定
- 燈號情境目前為合成；台電 2026 年 10 月實測一週皆為綠燈
- 只涵蓋任務①（備註 → 限制條件），任務②（白話解釋最佳化方案）尚未納入

## 作者

Yoyo（yoyu001）｜Purdue MSAI × 台電綜合研究所｜產生器：`data/generate.py`（seed 42）
