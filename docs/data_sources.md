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
