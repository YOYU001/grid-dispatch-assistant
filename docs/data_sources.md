# 資料來源盤點（2026-10-28，research agent 查證）

> 結論：**沒有公開的「台電操作員備註」資料集**（中英文都沒有），也找不到開源的中文電力調度指令資料集。
> 策略：**真實資料決定情境參數，文字由我們合成，拒絕題自己設計。**
> 對外說法：「以真實電網事件與已發表的彈性分級參數化的合成情境」，不可宣稱是真實操作員備註。

## 採用清單（依優先順序）

| # | 來源 | 類型 | 授權 | 用途 |
|---|---|---|---|---|
| 1 | [鳳凰城實驗論文 arXiv 2507.00909](https://arxiv.org/abs/2507.00909)（EPRI DCFlex 相關） | 研究設計 | 論文未標，引用事實不抄表 | Flex 0–3 定義（0%／10%／25%／50% 降速上限）、事件形狀（降 25%、3 小時、15 分鐘爬坡） |
| 2 | [台電電力供需資料 data.gov.tw 19995](https://data.gov.tw/dataset/19995) | 結構化數字（日資料） | 政府資料開放授權 1.0 | 真實備轉容量率分布 → 供電警戒燈號情境 |
| 3 | [NESO Demand Flexibility Service](https://www.neso.energy/data-portal/demand-flexibility-service) | 結構化 CSV | NESO Open Data Licence | 真實事件時間、要求降載量 |
| 4 | [CAISO 電網緊急事件史](https://www.caiso.com/documents/grid-emergencies-history-report-1998-present.pdf)、[ERCOT EEA 溝通矩陣](https://www.ercot.com/files/docs/2022/01/04/Energy_Emergency_Alert_Communications_Matrix.pdf) | 真實英文文字 | 未標明 | 事件分級用語、升級階梯、事件持續時間與原因（只參考，不照抄） |
| 5 | [Azure LLM 推論紀錄](https://github.com/Azure/AzurePublicDataset) | 結構化數字 | CC-BY-4.0／MIT（待確認 LICENSE） | 情境中的工作負載組合 |

## 只當參考格式／評估

- [ElecBench](https://arxiv.org/html/2407.05365v1)：電力 LLM 評測，含調度子集；太小不能訓練
- [ProOPF](https://arxiv.org/pdf/2602.03070)：自然語言 → 電力最佳化模型，與我們「備註 → JSON 限制條件」結構相似

## 待人工確認

- data.gov.tw 103262、33462、103046（可能有更細時間粒度）
- PJM、NYISO 公告
- 台電供電警戒燈號的最新門檻

## 2026-10-28 補充：Yoyo 提供的候選清單分類（尚待 research agent 逐一查證）

| 用途 | 資料 | v1 是否採用 |
|---|---|---|
| 情境：電網吃緊程度 | data.gov.tw「今日系統供需狀況」（10 分鐘，含備轉容量率）、「過去電力供需資訊」（日） | ✅ 採用（需確認 10 分鐘資料有無歷史檔） |
| 情境：省電效果數字 | NLR Generative AI Workload Power Profiles（H100、0.1 秒，OpenEI） | ✅ 採用（自測前的參考值） |
| 情境：工作負載組合 | BurstGPT（Azure OpenAI 請求紀錄） | ✅ 採用其一；ACME、Alibaba、MIT LLSC 備用 |
| 報告背景與放大推算 | 愛爾蘭 CSO 資料中心用電統計 | 報告用，不進訓練 |
| 他國電網 | EIA、CAISO OASIS、ERCOT、PJM、ENTSO-E、OPSD、NESO、AEMO | v1 不用（情境以台灣為主） |
| 碳強度 | Electricity Maps | 🟡 之後「碳感知排程」延伸 |
| 電網最佳化／強化學習 | OPFData、Grid2Op | ❌ 屬調度決策 AI，不是秘書的範圍 |
| 中文說法風格 | 台電公開新聞稿、供電警戒燈號說明、需量反應措施 PDF | ✅ 公開文件，可參考語氣 |

⚠️ 台電內部報告不得提供給外部 AI 服務或放入 repo，除非權責單位書面同意。

## 2026-10-28 查證結果（research agent 第 2 輪）

| 資料 | 查證結果 | 決定 |
|---|---|---|
| data.gov.tw 162595 今日系統供需狀況 | 只有**當天**快照，無官方歷史檔（網站擋爬蟲，欄位未確認） | 自建紀錄器：定時抓 `taipower.com.tw/d006/loadGraph/loadGraph/data/loadpara.txt` |
| [Zenodo 7537890](https://zenodo.org/record/7537890) | 社群整理的 10 分鐘歷史資料 2017-01～2022-06，約 40 MB，有缺漏；授權標示不一致（OGDL-1.0／CC-BY-4.0） | 🟡 可用於情境形狀，引用前釐清授權 |
| 19995 過去電力供需資訊 | 日資料、前一年＋今年至上月、OGDL-1.0 | ✅ 備轉容量率分布 |
| 37331 各機組發電量 | 10 分鐘、滾動視窗（約 3 個月）、OGDL-1.0 | 秘書用不到 |
| [NLR 功耗資料 OpenEI 8651](https://data.openei.org/submissions/8651)／[arXiv 2604.07345](https://arxiv.org/abs/2604.07345) | H100 SXM；Llama-2-70B LoRA、SD v2 訓練、Llama-3-70B vLLM 推論；0.1 秒；CC BY 4.0；**沒有降功率實驗** | ✅ 用於「各工作耗電形狀」；**省電百分比改引鳳凰城論文** |
| 愛爾蘭 | 2025 年 23%（CSO 2026-07-07）、2024 年 22%、2015 年 5% | 報告背景 |
| Hugging Face Hub | 無台灣／中文調度指令、需量反應備註、GPU 功耗資料集；`yentinglin/TaiwanChat`（CC-BY-NC）僅可考慮作語氣混合 | 訓練文字仍須自行合成 |

可引用數字（NLR 論文）：H100 TDP 700 W、閒置 72.5 W／張、推論約 2.8–2.9 kW／節點、微調約 3 分鐘爬坡。
