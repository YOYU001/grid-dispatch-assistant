---
name: colab-storage
description: 在 Google Colab 筆記本中設定持久儲存（Google Drive）與 Hugging Face Hub 上傳、斷線接續訓練。寫任何要在 Colab 跑的訓練或評估筆記本時使用。
---

# Colab 儲存流程

標準程式碼在 `notebooks/snippets/colab_storage.py`（已經 code-review 審查），分 5 段，**分段交給使用者貼到 Colab**：

| 段 | 何時貼 | 做什麼 |
|---|---|---|
| A | 每本筆記本最上面 | 掛載 Drive、建立 `checkpoints/ adapters/ results/ data/` |
| B | 接在 A 後面 | 用 Colab 密鑰 `HF_TOKEN` 登入 Hugging Face |
| C | 需要存成績或上傳時 | `save_results(df, name)`、`push_to_hub(dir, repo_name, repo_type)` |
| D | 訓練筆記本 | `find_resume_point(run_name)`：找最新「完整」checkpoint |
| E | 收工前 | `drive.flush_and_unmount()` |

## 規則
1. **不存原始模型到 Drive**（8 GB，免費 Drive 只有 15 GB）；只存 LoRA、checkpoint、成績、資料。
2. 訓練設定：`output_dir=DIRS["checkpoints"]/RUN_NAME`、`save_steps≈200`、`save_total_limit=2`。
3. 每個 LoRA checkpoint（4B、r=16、all-linear）約 0.5 GB；被刪的舊檔會進 **Drive 垃圾桶仍佔空間**，提醒使用者清空。
4. 不要把 `embed_tokens`／`lm_head` 放進 `modules_to_save`（每個 checkpoint 多好幾 GB）。
5. `push_to_hub` 一律私人 repo，且上傳前檢查是否私人。
6. 台電內部資料不得進 Drive 的 `data/` 或 Hub。
7. NAS 備份：用 Synology Cloud Sync 單向下載 Drive 的 `grid-dispatch-assistant/`，**不要**把 NAS 開放給 Colab 連入。
