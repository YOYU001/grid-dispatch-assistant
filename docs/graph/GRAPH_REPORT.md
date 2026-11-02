# Graph Report - grid-dispatch-assistant  (2026-11-02)

## Corpus Check
- Corpus is ~14,135 words - fits in a single context window. You may not need a graph.

## Summary
- 291 nodes · 472 edges · 16 communities (14 shown, 2 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 55 edges (avg confidence: 0.82)
- Token cost: 289,438 input · 0 output

## Community Hubs (Navigation)
- 套件與外部依賴
- 合成資料集與資料卡
- Code-review／QA 審核規範
- 資料產生器程式 generate.py
- Colab 儲存與備份
- 多模型比較與思考版本實驗
- 階段 0：A 版訓練管線
- 資料工程與評分基準
- 專案定位與基底模型
- 七階段規劃與延伸作品
- 評分器 grade()（筆記本 03／05）
- 評分器 grade()（筆記本 01）
- 筆記本轉換工具
- 運算環境與基準成績
- 母專案與推論優化
- 訓練加速技巧

## God Nodes (most connected - your core abstractions)
1. `Data source inventory (2026-10-28)` - 16 edges
2. `七階段訓練規劃` - 11 edges
3. `pick()` - 10 edges
4. `Secretary output schema v2` - 10 edges
5. `is_heldout()` - 9 edges
6. `C()` - 9 edges
7. `Grid Dispatch Notes Dataset v1 (zh-TW, synthetic)` - 9 edges
8. `make_example()` - 8 edges
9. `電網調度秘書 (Grid Dispatch Secretary LLM)` - 8 edges
10. `規則排程器 / OR-Tools 最佳化器` - 8 edges

## Surprising Connections (you probably didn't know these)
- `公平比較規則` --semantically_similar_to--> `A/B/C 三版本思考方式對照實驗`  [INFERRED] [semantically similar]
  docs/TRAINING_PLAN.md → AGENT.md
- `階段 0：A 版基準管線` --semantically_similar_to--> `八步固定訓練流程`  [INFERRED] [semantically similar]
  docs/TRAINING_PLAN.md → AGENT.md
- `ProOPF (NL -> power optimization model)` --semantically_similar_to--> `Grid Dispatch Secretary (note -> constraint JSON, no decisions)`  [INFERRED] [semantically similar]
  docs/data_sources.md → data/DATASET_CARD.md
- `Card sections A-H (basic, reasoning, multi, cancel, grid, colloquial, refuse, off-topic)` --semantically_similar_to--> `Task categories (single/multi/cancel/grid/no_action/refuse/off_topic)`  [INFERRED] [semantically similar]
  eval/human_cards.md → data/DATASET_CARD.md
- `Five data center workloads (customer chat, batch summary, LoRA, RAG index, emergency inference)` --semantically_similar_to--> `workload_id set (chat-ecommerce, batch-summary, lora-finetune, rag-index, emergency-inference)`  [INFERRED] [semantically similar]
  eval/human_cards.md → docs/output_schema_v2.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **三個 subagent 分工（research／code-review／qa）** — agent_research_agent, agent_code_review_agent, agent_qa_agent [EXTRACTED 1.00]
- **階段 0 筆記本管線 01→03→04→05** — progress_notebook_01_baseline, progress_notebook_03_grader, progress_notebook_04_qlora_sft, progress_notebook_05_reeval, progress_dataset_grid_dispatch_notes_zh [EXTRACTED 1.00]
- **等待外部 GPU 的階段（Azure／Purdue）** — todo_azure_gpu, todo_purdue_gpu, docs_training_plan_stage_4_moe_nemo, docs_training_plan_stage_5_inference_optimization, docs_training_plan_stage_7_deploy_multi_gpu [INFERRED 0.85]
- **Constraint record fields of secretary output** — docs_output_schema_v2_workload_ids, docs_output_schema_v2_flex_tier, docs_output_schema_v2_allow_pause, docs_output_schema_v2_earliest_start, docs_output_schema_v2_action_clear, docs_output_schema_v2_valid_until [EXTRACTED 1.00]
- **Evaluation exams used for baseline scoring** — docs_baseline_03_analysis_baseline_qwen3_4b, data_dataset_card_dataset_splits, data_dataset_card_test_unseen_phrasing, eval_human_cards_human_test_set [EXTRACTED 1.00]
- **Real sources parameterizing synthetic scenarios** — docs_data_sources_phoenix_experiment_paper, docs_data_sources_taipower_today_162595, docs_data_sources_taipower_supply_demand_19995, docs_data_sources_nlr_power_profiles, docs_data_sources_burstgpt, docs_data_sources_synthetic_strategy [INFERRED 0.85]
- **Fine-tune project subagent pipeline (research -> code-review -> QA)** — _claude_agents_research_research_agent, _claude_agents_code_review_code_review_agent, _claude_agents_qa_qa_agent [INFERRED 0.85]
- **T4 memory-saving stack for 4B training** — docs_finetune_guide_qlora, docs_finetune_guide_gradient_checkpointing, docs_finetune_guide_paged_adamw_8bit, docs_finetune_guide_fp16_vs_bf16, docs_finetune_guide_t4_memory_budget [EXTRACTED 1.00]
- **colab_storage.py persistence + resume workflow** — _claude_skills_colab_storage_skill_save_results, _claude_skills_colab_storage_skill_push_to_hub, _claude_skills_colab_storage_skill_find_resume_point, _claude_skills_colab_storage_skill_google_drive_storage [EXTRACTED 1.00]

## Communities (16 total, 2 thin omitted)

### Community 0 - "套件與外部依賴"
Cohesion: 0.05
Nodes (18): colab-storage Skill, ckey(), from_hub(), from_human(), generate_batch(), grade(), load_done(), norm_value() (+10 more)

### Community 1 - "合成資料集與資料卡"
Cohesion: 0.06
Nodes (41): Dataset fields (id, category, held_out_phrasing, messages, note, trace), Dataset splits (train 3954 / validation 500 / test 500), data/generate.py (seed 42), Grid Dispatch Notes Dataset v1 (zh-TW, synthetic), Grid Dispatch Secretary (note -> constraint JSON, no decisions), Independent QA agent spec-only review (87 samples), Task categories (single/multi/cancel/grid/no_action/refuse/off_topic), test_unseen_phrasing split (394, held-out phrasings) (+33 more)

### Community 2 - "Code-review／QA 審核規範"
Cohesion: 0.06
Nodes (36): Code Review Agent, GPU-Cost-Wasting Settings Check, Reproducibility (seed, pinned versions, config files), Result-Distorting Bugs (data leakage, chat template mismatch, loss on prompt), Secret / Internal Data Safety (HF token), Fair Before/After Comparison, GSM8K Regression Check (>5pt drop flagged), QA Agent (+28 more)

### Community 3 - "資料產生器程式 generate.py"
Cohesion: 0.13
Nodes (28): a_delay(), a_flexible(), a_freeze(), a_moderate(), a_no_pause(), a_no_slow(), a_slight(), C() (+20 more)

### Community 4 - "Colab 儲存與備份"
Cohesion: 0.15
Nodes (7): Google Drive Persistent Storage, Synology NAS Cloud Sync Backup, save_results(df, name), find_resume_point(), _now(), push_to_hub(), save_results()

### Community 5 - "多模型比較與思考版本實驗"
Cohesion: 0.23
Nodes (10): Qwen3-4B-Thinking-2507, 規則軌跡 (rule trace), 模型比較組合（A 小型同級／B 架構／C MoE）, NVIDIA-Nemotron-3-Nano-30B-A3B (Mamba-2+Transformer MoE), NVIDIA-Nemotron-Nano-9B-v2 (Mamba 混合), Qwen3-30B-A3B, 階段 2：多模型比較（bootstrap 信賴區間）, 階段 4：MoE 與 NeMo vs HF PEFT (+2 more)

### Community 6 - "階段 0：A 版訓練管線"
Cohesion: 0.27
Nodes (11): code-review subagent (Opus), lighteval GSM8K 退化檢查, qa subagent (Opus), research subagent (Sonnet), Hugging Face smol-course 第一章 (SFT) 期末作業, 八步固定訓練流程, 階段 0：A 版基準管線, A 版外掛 yoyu001/grid-dispatch-A-instruct-qlora-v1 (+3 more)

### Community 7 - "資料工程與評分基準"
Cohesion: 0.25
Nodes (11): grade() 評分器（03／05 一字不差）, 階段 1：資料工程（MinHash 去重、LLM 改寫＋規則驗證、資料量實驗）, 合成資料產生器 data/generate.py, HF 私人 dataset yoyu001/grid-dispatch-notes-zh, 真人考卷 eval/human_test_v2.jsonl (20 題), 防洩題檢查（遮名稱後連續 8 字相同即丟棄）, NAS 電網資料收集器, 筆記本 03 v2 評分器與基準評估 (+3 more)

### Community 8 - "專案定位與基底模型"
Cohesion: 0.22
Nodes (8): 電網調度秘書 (Grid Dispatch Secretary LLM), 私人 repo YOYU001/grid-dispatch-assistant, Qwen3-4B-Instruct-2507, 安全驗證器與人員核准, NVIDIA-Nemotron-3-Nano-4B, NVIDIA JD：AI Software Engineer (ML/NLP/LLM) 與 Deep Learning Solution Architect, 筆記本 01 認識模型與 5 題試考, grid-dispatch-assistant README 概述

### Community 9 - "七階段規劃與延伸作品"
Cohesion: 0.24
Nodes (10): graphify skill / 進度圖, 規則排程器 / OR-Tools 最佳化器, 環境／化學法規 LLM（第二個作品，延後）, 七階段訓練規劃, 階段 3：訓練方式（LoRA/QLoRA/全參數/DPO/GRPO/蒸餾）＋訓練加速, 階段 6：Agent＋RAG＋NeMo Guardrails, 階段 7：Docker／k8s／多 GPU 分散式, Purdue GPU 資源申請 (+2 more)

### Community 10 - "評分器 grade()（筆記本 03／05）"
Cohesion: 0.22
Nodes (6): ckey(), diff(), grade(), norm_value(), parse_json(), score()

### Community 11 - "評分器 grade()（筆記本 01）"
Cohesion: 0.33
Nodes (4): grade(), key(), normalize(), parse_json()

### Community 13 - "運算環境與基準成績"
Cohesion: 0.40
Nodes (5): colab-storage skill, Colab 免費 T4 運算環境 (fp16), notebooks/tools/py2ipynb.py 轉換器, 基準成績 (exact 19.6%／unseen 19.8%／真人 30%), Kaggle T4 運算環境

### Community 14 - "母專案與推論優化"
Cohesion: 0.50
Nodes (4): 電網尖峰下 AI 資料中心 GPU 彈性負載調度輔助系統（台電綜研所 PoC）, 品質 × 速度 × 記憶體 × 耗電總表, 階段 5：推論優化（vLLM/TRT-LLM/NIM × FP16/FP8/INT4 × 功率上限）, FP8 耗電量測（需 H100 或 L4／L40S）

## Knowledge Gaps
- **38 isolated node(s):** `安全驗證器與人員核准`, `research subagent (Sonnet)`, `code-review subagent (Opus)`, `qa subagent (Opus)`, `graphify skill / 進度圖` (+33 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 90 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `colab-storage skill` connect `運算環境與基準成績` to `Colab 儲存與備份`?**
  _High betweenness centrality (0.259) - this node is a cross-community bridge._
- **What connects `安全驗證器與人員核准`, `research subagent (Sonnet)`, `code-review subagent (Opus)` to the rest of the system?**
  _38 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `套件與外部依賴` be split into smaller, more focused modules?**
  _Cohesion score 0.053410893707033315 - nodes in this community are weakly interconnected._
- **Should `合成資料集與資料卡` be split into smaller, more focused modules?**
  _Cohesion score 0.05803921568627451 - nodes in this community are weakly interconnected._
- **Should `Code-review／QA 審核規範` be split into smaller, more focused modules?**
  _Cohesion score 0.06116642958748222 - nodes in this community are weakly interconnected._
- **Should `資料產生器程式 generate.py` be split into smaller, more focused modules?**
  _Cohesion score 0.12912912912912913 - nodes in this community are weakly interconnected._