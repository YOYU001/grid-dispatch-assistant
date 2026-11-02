# TODO.md — 重要但現在還做不了的事

> 想到就記。做完就移到 PROGRESS.md 並從這裡刪除。
> 標記：🔴 卡住（等外部）｜🟡 之後做｜🟢 隨時可做

## 運算資源
- 🟡 **NAS 方案（已決定：分階段）**：Colab 階段用 Drive＋Synology Cloud Sync；Azure VM 階段用 Tailscale 直連 NAS（NAS 開 SSH → rsync）。auth key 設 ephemeral＋tag，ACL 只允許連 NAS
- 🟢 設定 Synology Cloud Sync：單向下載 Drive 的 `grid-dispatch-assistant/` 到 NAS
- 🔴 Azure A100/H100 GPU 額度申請中 → 到手後把訓練從 Colab 移過去
- 🔴 Purdue GPU 資源申請中 → 確認型號、能否讀 nvidia-smi 功耗
- 🟡 確認雲端 VM 能否用 `nvidia-smi -pl` 調功率上限（母專案第 1 步需要）
- 🟡 量 FP8 耗電（「降載 B」按鈕）需選 **H100 或 L4／L40S**；A100、T4 不支援 FP8 原生計算，只能用來訓練

## 作業與帳號
- 🟢 確認 HF 帳號 `yoyu001` 有 write token（推模型、推評估結果要用）
- 🟡 smol-course 評估用 `hf jobs` 需 Pro 方案 → 改在 Colab／Azure 直接跑 lighteval，PR 裡附上指令
- 🟡 做 smol-course unit 1 quiz

## 模型與資料
- 🟡 筆記本 05（訓練後重考）：fp16 完整基底模型＋`PeftModel.from_pretrained(adapter)`（不要 4-bit，和基準同條件）；**換 RUN_NAME**，且 `load_done` 要比對 adapter id，否則會讀到基準舊答案；做 Kaggle 版
- 🟡 smol-course 提交：lighteval 需合併後的完整模型 → 合併 adapter 後推一個 merged repo
- 🟡 合成資料 v2：用 LLM 改寫備註增加說法變化（改寫後用規則檢查答案不變），比加題數更有效
- 🟡 真人考卷擴充到 50～100 題（目前 20 題誤差約 ±20%）
- 🟡 NAS 收集器（2026-10-22 起運作中）：持續累積到夏季，之後用真實黃／橘燈資料取代合成燈號
- 🟡 第 2 年選項：調度決策是否需要「學習型」AI。優先考慮 (1) 備轉容量率／負載**預測模型**（提早知道何時吃緊）；(2) 自建資料中心排程模擬環境做強化學習或模仿最佳化器。Grid2Op、OPFData 是輸電網問題，不直接適用
- 🟢 規則排程器要輸出「規則軌跡」，供 B、C 版本產生推理內容
- 🟡 B 版本前：research agent 確認 Qwen3-4B-Thinking-2507 的 repo、授權、chat template
- 🔴 B 版本訓練與評估等 Azure A100（輸出長，T4 太慢）
- 🟡 耗電量測：評估時同步記錄 nvidia-smi 功率，算每題耗電
- 🟡 第二階段：加入 tool calling（查電網、查工作、呼叫最佳化器）
- 🟡 第二階段：NeMo 微調對照組、NeMo Guardrails
- 🟡 第二階段：TensorRT-LLM / NIM 部署比較
- 🟡 RAG：把台電需量反應措施 PDF 做成檢索資料庫（先確認最新版規定）

## 七階段訓練規劃（2026-11-02，詳見 `docs/TRAINING_PLAN.md`）
- 🟢 階段 1 資料工程：MinHash 去重、LLM 改寫＋規則驗證、品質過濾、資料量實驗（1,000／2,000／3,954）、更新 Data Card
- 🟢 research agent：確認 NVIDIA-Nemotron-3-Nano-4B 的 chat template、T4 能否 QLoRA；選 A 組第三個模型（Gemma 等）
- 🟢 量各模型 tokenizer 的「每個中文字幾個 token」
- 🟡 階段 2：A 組三模型同流程 QLoRA＋評估（加 bootstrap 信賴區間）
- 🟡 階段 3：全參數（0.6B）、DPO（誤拒／漏拒）、GRPO（規則排程器當評分器）、蒸餾 4B→0.6B；訓練加速對照（packing、Unsloth、Liger）；epoch 與 r 消融
- 🔴 階段 4（等 A100）：Nemotron-Nano-9B-v2 架構比較；Qwen3-30B-A3B vs Nemotron-3-Nano-30B-A3B MoE；NeMo vs HF PEFT
- 🔴 階段 5（等 H100／L4）：HF／vLLM／TRT-LLM／NIM × FP16／FP8／INT4 × batch／KV cache × 功率上限，量每千 token 耗電
- 🟡 階段 6：tool calling、RAG 評估（Recall@k、引用正確率）、NeMo Guardrails 對照（Azure 晚到就提前）
- 🔴 階段 7（等 Purdue 2 張 GPU）：FSDP／DeepSpeed 分散式 LoRA、vLLM 張量平行、Docker＋單機 k8s、迷你 GPT 預訓練
- 🟡 平行基本功：nanoGPT、GPU 記憶體頻寬觀念、LeetCode 每週 2–3 題
- 🟡 環境法規 LLM：本專案階段 6 之後再開始

## 提案與作品集
- 🟡 若台電正式立案，先問主管「個人公開資料版本」可否發表
- 🟡 GitHub repo 公開前檢查：沒有內部資料、沒有 token
