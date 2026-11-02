# TRAINING_PLAN.md — AI 模型訓練總規劃（v2）

> 建立：2026-11-02｜依據：NVIDIA「AI Software Engineer (ML/NLP/LLM)」與「Deep Learning Solution Architect」兩份 JD
> 每次開始新階段前：先看 `graphify-out/GRAPH_REPORT.md`（見 `AGENT.md` 第 3 節）

---

## 0. 專案定位（2026-11-02 調整）

- **主角是 LLM，排程器是工具。** 排程器故意保持簡單、可驗證；Agent 呼叫它、解釋它、在它旁邊把關。
- 對外說法：「我把一個 4B 中文 LLM 微調成專業 Agent，它會呼叫排程器、查規定、拒絕違規指令，並在不同 GPU 上做推論優化與耗電量測。」
- 為什麼不讓 LLM 直接決策？→ 可驗證、可稽核、可回復。面試答得出這題是加分。

## 1. 這個專案要練成的能力

| # | 能力 | 對應階段 |
|---|---|---|
| 1 | 收集與清洗資料 | 1 |
| 2 | 評估模型品質（正確率、穩健性、速度、記憶體、耗電） | 全程，2 為主 |
| 3 | 比較不同開源模型（Qwen、Nemotron…） | 2、4 |
| 4 | 業界訓練方式（LoRA、QLoRA、全參數、DPO、GRPO、蒸餾、MoE） | 3、4 |
| 5 | 訓練加速與省記憶體 | 3 |
| 6 | 推論加速（模型在 GPU 上跑得快） | 5 |
| 7 | NVIDIA 工具 vs 一般開源工具 | 4、5、6 |
| 8 | Agent、RAG | 6 |
| 9 | Docker、Kubernetes、多 GPU | 7 |

## 2. JD 要求逐條確認

| JD 要求 | 能否練到 | 專案中的做法 |
|---|---|---|
| 碩士學歷 | ✅ | Purdue MSAI 進行中 |
| 4 年以上實戰 | ❌ | 專案補不了；以「做深」的作品集彌補，同時投初階職缺 |
| 開源 LLM 微調 | ✅ | LoRA、QLoRA、全參數、DPO、GRPO、蒸餾 |
| 生產環境推論優化 | ✅ | vLLM／TensorRT-LLM／NIM、量化、KV cache |
| LLM 預訓練 | 🟡 縮小版 | 從零訓練迷你 GPT；大型預訓練懂原理 |
| 主流 LLM 架構 | ✅ | Transformer、Mamba 混合、MoE 三種架構比較 |
| PyTorch／HF Transformers | ✅ | 全程使用 |
| GPU 運算知識 | ✅ | 記憶體、速度、耗電、功率上限實測 |
| 叢集架構 | 🟡 縮小版 | Purdue HPC 的 Slurm |
| 多 GPU 分散式訓練／推論 | 🟡 2 張 GPU | FSDP／DeepSpeed ZeRO、vLLM 張量平行 |
| Agent 解決商業問題 | ✅ | 秘書呼叫排程器、查規定 |
| RAG 設計（AI SW Eng 要求「深入」） | ✅ | 台電規定檢索＋檢索評估 |
| 對技術／非技術人員溝通 | ✅ | 台電提案＋技術報告 |
| TRT-LLM、NeMo（加分） | ✅ | 與開源工具做對照實驗 |
| Megatron-LM（加分） | 🟡 | 透過 NeMo 間接使用，懂張量／管線平行 |
| 量化、KV cache、省記憶體（加分） | ✅ | 量化對照表、記憶體曲線 |
| Docker、Kubernetes（加分） | ✅ | 單機 k8s 部署 NIM＋Agent |
| 大型叢集管理（加分） | ❌ | 個人做不到；懂原理 |
| 演算法、數學（AI SW Eng） | 🟡 | 每週 LeetCode 2–3 題（專案外） |

## 3. 模型比較設計

### 3.1 比較組合

| 組別 | 模型 | 比什麼 | 硬體 |
|---|---|---|---|
| **A 組：小型同級（主比較）** | Qwen3-4B-Instruct-2507（已完成 A 版）／NVIDIA-Nemotron-3-Nano-4B／第三個（research agent 選，例如 Gemma） | 同資料、同訓練法，誰學得最好 | T4 |
| **B 組：架構比較** | NVIDIA-Nemotron-Nano-9B-v2（Mamba＋Transformer 混合）vs 同級 Transformer | 架構對速度、記憶體的影響 | A100 |
| **C 組：MoE 比較** | Qwen3-30B-A3B vs NVIDIA-Nemotron-3-Nano-30B-A3B | MoE 的微調與部署 | A100 80GB |

### 3.2 查證過的事實（2026-11-02）

- Nemotron 3 Nano 4B 的 model card 只列英文為支援語言 → 題目：「英文模型經中文微調後，能否追上 Qwen？」
- Nemotron 3 Nano 30B-A3B：Mamba-2＋Transformer 混合，前饋層換成稀疏 MoE；總參數 30B、每次約用 3.5B
- NVIDIA 宣稱 30B-A3B 在單張 H200 上吞吐量為 Qwen3-30B-A3B 的 3.3 倍 → **自己驗證**
- ⚠️ Mamba 層需要專用核心，T4 可能不支援；使用前由 research agent 確認
- 授權：Nemotron 為 NVIDIA Open Model License（可商用、需附授權聲明）；Qwen3 為 Apache-2.0

來源：
- https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8
- https://huggingface.co/blog/nvidia/nemotron-3-nano-efficient-open-intelligent-models
- https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-nano-30b-a3b

### 3.3 公平比較規則

- 同一批題目、同一份考卷、同一個評分器（`grade()`，03／05 一字不差）
- 同樣的訓練預算（相同步數或相同 GPU 時間），超參數只做小範圍調整並全部記錄
- 每個模型用自己的 chat template
- 先量「每個中文字約幾個 token」：英文為主的字典會把中文切得更碎 → 更慢、更吃記憶體

## 4. 七個階段

### 階段 0：A 版基準管線（✅ 大致完成）
- 01 認識模型 → 02 資料 → 03 基準 → 04 QLoRA 訓練 → 05 重考
- 待完成：05 正式重考、GSM8K 退化檢查、smol-course 提交

### 階段 1：資料工程（11 月，不需 GPU）
| 做什麼 | 白話 |
|---|---|
| 去重（MinHash） | 找出幾乎一樣的題目刪掉 |
| 防洩題檢查 | 考卷題目不可出現在訓練資料（已有，延續） |
| LLM 改寫＋規則驗證 | 大模型改寫說法，規則確認答案不變 |
| 品質過濾 | 太短、矛盾、格式壞掉的丟掉 |
| 資料量實驗 | 1,000／2,000／3,954 題各訓練一次 |
| Data Card 更新 | 來源、清洗步驟、限制 |
| 真人考卷擴充 | 20 題 → 50～100 題（誤差 ±20% 太大） |

**產出：** 資料 v2、資料量曲線圖

### 階段 2：多模型比較（11–12 月，T4）
- A 組三個模型用同一套 QLoRA 流程訓練
- 評估一次做齊：exact／core／intent、刁鑽題、拒絕召回、誤拒、GSM8K 退化、延遲、記憶體峰值
- 加上信賴區間（bootstrap），判斷差距是否真的存在

**產出：** 多模型比較表、tokenizer 效率分析

### 階段 3：業界訓練方式＋訓練加速（12 月–1 月）
| 訓練方式 | 白話 | 本專案用法 | 硬體 |
|---|---|---|---|
| LoRA vs QLoRA | 貼便利貼 vs 縮印後再貼 | 比速度、記憶體、正確率 | A100 |
| 全參數微調 | 整本字典改寫 | 0.6B 小模型 | T4 勉強 |
| DPO | 看「好答案 vs 壞答案」學偏好 | 修正誤拒／漏拒取捨 | T4 |
| GRPO（強化學習） | 自己作答、答對給分 | 規則排程器＝自動評分老師 | A100 較佳 |
| 蒸餾 | 大教小 | 4B → 0.6B | T4 |

訓練加速對照（同模型、一次改一件事）：一般 HF／packing／Unsloth／Liger 核心／關 gradient checkpointing → 記錄時間與記憶體峰值。
消融實驗：epoch 0.25／0.5／1／2、r 8／16／32。

**產出：** 訓練方式比較表、加速對照表

### 階段 4：MoE 與 NVIDIA 訓練工具（2–3 月，A100）
- MoE 不從零訓練，**微調現成 MoE**（業界實際做法）
- 觀念：記憶體看總參數（30B），速度看每次用到的參數（約 3B）
- C 組兩個 MoE 各做 QLoRA，量記憶體與速度
- 同一模型用 NeMo 與 HF PEFT 各訓練一次，比時間、記憶體、操作難易度（NeMo 目前的微調入口由 research agent 確認）

**產出：** MoE 比較、NeMo vs HF 比較

### 階段 5：推論優化（3–4 月，A100／H100／L4）
| 比較項目 | 選項 |
|---|---|
| 推論引擎 | HF generate → vLLM → TensorRT-LLM → NIM |
| 量化 | FP16 → FP8 → INT4（AWQ／GPTQ） |
| 參數 | batch、KV cache 大小、最長長度 |
| 功率 | 300 W → 150 W 功率上限 |

每格都量：首字時間、p95 延遲、tokens/s、VRAM、**每千 token 耗電**、正確率。
FP8 需 H100 或 L4／L40S（A100、T4 不支援）。

**產出：** 「品質 × 速度 × 記憶體 × 耗電」總表（SA 面試核心作品）

### 階段 6：Agent＋RAG（4–5 月，T4 即可）
- 工具呼叫：查電網、查工作、呼叫排程器、送核准；評估工具選擇與參數正確率
- RAG：台電需量反應規定；評估 Recall@k、引用正確率、找不到時是否亂編
- NeMo Guardrails vs 自寫規則：擋惡意指令的效果

**產出：** Agent 流程圖、RAG 評估報告、對抗測試結果

### 階段 7：部署與多 GPU（5–6 月以後）
| 做什麼 | 硬體 |
|---|---|
| Docker 打包模型服務＋Agent | 任何機器 |
| 單機 Kubernetes 部署 NIM＋Agent | 雲端 VM |
| 2 張 GPU 分散式 LoRA（FSDP／DeepSpeed ZeRO） | Purdue HPC（Slurm） |
| 2 張 GPU 張量平行推論（vLLM） | Purdue HPC 或 Azure |
| 從零訓練迷你 GPT（理解預訓練） | T4 |

**產出：** docker-compose、k8s manifest、單卡 vs 雙卡比較

## 5. 時程與硬體依賴

| 階段 | 時間 | 硬體 | 狀態 |
|---|---|---|---|
| 0 A 版管線 | 10–11 月 | T4 | ✅ 收尾中 |
| 1 資料工程 | 11 月 | 不需 GPU | 🟢 可開始 |
| 2 多模型比較 | 11–12 月 | T4 | 🟢 Kaggle |
| 3 訓練方式 | 12–1 月 | T4（部分 A100） | 🟡 |
| 4 MoE／NeMo | 2–3 月 | A100 80GB | 🔴 等 Azure |
| 5 推論優化 | 3–4 月 | H100 或 L4 | 🔴 等 Azure |
| 6 Agent＋RAG | 4–5 月 | T4 | 🟢 可提前 |
| 7 部署／多 GPU | 5–6 月 | 2 張 GPU | 🔴 等 Purdue |

Azure 若晚到：階段 6 提前到 4、5 之前。

## 6. 平行補基本功（每週 3–5 小時）

| 項目 | 做法 |
|---|---|
| Transformer 原理 | Karpathy nanoGPT／"Let's build GPT" |
| GPU 基礎 | 記憶體頻寬 vs 運算；為何解碼受頻寬限制；CUDA 入門 |
| 演算法 | LeetCode 每週 2–3 題 |
| 自己讀懂程式 | 每本筆記本挑一段，關掉 AI 自己改寫一次 |

## 7. 面試一段話

> 「我用同一套資料與評估系統，比較了 Qwen、Nemotron 等模型在中文任務上的表現，涵蓋 Transformer、Mamba 混合與 MoE 三種架構。訓練上試過 QLoRA、全參數、DPO、GRPO 與蒸餾；部署上比較了 vLLM、TensorRT-LLM 與 NIM 在 FP16、FP8、INT4 下的延遲、記憶體與每千 token 耗電。」

## 8. 之後的第二個作品（延後）

環境／化學法規 LLM：等本專案完成階段 6 後再做，重用同一套訓練、評估、量測流程，對齊 AI Software Engineer 的「生產力工具＋RAG」。
