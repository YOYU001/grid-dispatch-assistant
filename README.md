# grid-dispatch-assistant

**QLoRA fine-tuning of Qwen3-4B-Instruct-2507 into a grid-dispatch copilot for AI data centers.**

When the power grid is under stress, a data center can slow down, delay or pause flexible GPU workloads. Operators give those instructions as short, informal Chinese notes ("the customer demo is tonight, keep customer service at full speed"). This model turns each note into structured constraint JSON for a downstream scheduler, explains it in plain language, and refuses instructions that would break a customer SLA or skip human approval.

The model **does not make dispatch decisions**. A rule-based scheduler / OR-Tools optimizer decides, and every plan still passes a safety validator and human sign-off.

> 中文說明：[README.zh-TW.md](README.zh-TW.md)
> Only public and synthetic data is used. No Taipower internal data is in this repo.

## Results

Exact match on the full constraint JSON (Kaggle T4, greedy decoding). 95% Wilson intervals in brackets.

| Eval set | Items | Base model | After QLoRA SFT |
|---|---|---|---|
| test (same generator as training) | 500 | 19.6% | **98.6%** [97.1–99.3] |
| test_unseen_phrasing (held-out wording) | 394 | 19.8% | **81.2%** [77.1–84.8] |
| human-written notes | 20 | 30.0% | **75.0%** [53.1–88.8] |
| GSM8K (lighteval, regression check) | — | 0.670 | 0.710 |

- Refusal recall 75.2% → **96.6%**; false refusals 9.4% → **0.1%**. JSON validity 100% before and after.
- The gap between `test` and `unseen` / human notes shows the model partly learned the data generator itself. 74 of the 86 remaining errors come from unseen phrasings ("X can't be touched" answered only halfway, relative-time arithmetic). Full error analysis: [docs/eval_05_full_analysis.md](docs/eval_05_full_analysis.md).
- Follow-up preference alignment (DPO) on this SFT adapter: [YOYU001/grid-dispatch-dpo](https://github.com/YOYU001/grid-dispatch-dpo).

## Output format

```json
{"intent": "set_constraints | no_action | off_topic | refuse",
 "constraints": [{"workload_id": "chat-ecommerce", "field": "flex_tier | earliest_start | allow_pause",
                  "value": 0, "action": "set | clear", "valid_until": "YYYY-MM-DD HH:MM | null"}],
 "explanation": "..."}
```

Spec: [docs/output_schema_v2.md](docs/output_schema_v2.md)

## Pipeline

| # | Notebook | What it does |
|---|---|---|
| — | [`data/generate.py`](data/generate.py) | Rule-based teacher generates 5,000 synthetic notes in 7 categories, with leak checks against the human test set |
| 01 | `01_explore_model_*` | Inspect the base model and chat template; 5-question smoke test |
| 02 | `02_upload_dataset_*` | Upload the dataset to the Hugging Face Hub (private) |
| 03 | `03_baseline_eval.py` | Baseline exam with the v2 scorer, batched and resumable |
| 04 | `04_sft_qlora_kaggle` | QLoRA SFT on a single T4 (r=16, alpha=32, 496 steps, 163 min, 5.9 GB peak) |
| 05 | `05_eval_finetuned_kaggle` | Re-run the same exam on the fine-tuned adapter, with adapter hash and data version pinned |
| 06 | `06_merge_push_kaggle` | Merge the adapter and push the full model, with a consistency check |
| 07 | `07_gsm8k_kaggle` | GSM8K regression check with lighteval |

Dataset splits: train 3,954 · validation 500 · test 500 · test_unseen_phrasing 394. The 20 human-written test notes are kept out of the repo and out of training.

## Hugging Face assets

| Asset | Hub ID |
|---|---|
| Base model | [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| Merged SFT model | [`yoyu001/grid-dispatch-A-instruct-merged-v1`](https://huggingface.co/yoyu001/grid-dispatch-A-instruct-merged-v1) |
| SFT adapter (QLoRA) | `yoyu001/grid-dispatch-A-instruct-qlora-v1` (private) |
| Synthetic dataset | `yoyu001/grid-dispatch-notes-zh` (private) |

## Repo layout

```
data/       synthetic data generator and dataset card
notebooks/  Colab/Kaggle notebooks 01–07 (.py source + .ipynb)
results/    model predictions and per-item scores
docs/       training plan, schema, data-source survey, eval analyses
eval/       human test scenario cards (answers kept private)
.claude/    subagent definitions (research / code-review / qa) used during development
```

## Context

Pilot study for an AI data-center GPU flexible-load dispatch project with Taiwan Power Company (Taipower), and the final assignment for Unit 1 (SFT) of the Hugging Face smol-course. Built with Claude Code as a pair-programming assistant; commits co-authored by Claude are marked in their trailers.
