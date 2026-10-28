# 秘書輸出格式 v2（2026-10-28，依真人測試集修訂）

## 輸入（每題固定附上情境）

```
【現在時間】2026-10-28（三）15:00
【電網狀況】燈號：綠燈；預估尖峰時段：18:00-21:00；預估備轉容量率：11.05%
【操作員備註】……
```

## 輸出

```json
{
  "intent": "set_constraints",
  "constraints": [
    {"workload_id": "chat-ecommerce", "field": "flex_tier", "value": 0,
     "action": "set", "valid_until": "2026-10-29 08:00"}
  ],
  "explanation": "一句白話中文說明"
}
```

| 欄位 | 可用值 | 說明 |
|---|---|---|
| `intent` | `set_constraints`／`no_action`／`off_topic`／`refuse` | 設定限制／照常不需動作／與調度無關／拒絕 |
| `workload_id` | `chat-ecommerce`、`batch-summary`、`lora-finetune`、`rag-index`、`emergency-inference`、`ALL_FLEXIBLE` | `ALL_FLEXIBLE` = 批次摘要＋LoRA＋RAG（不含客服與緊急推論） |
| `field` | `flex_tier`（0–3）、`earliest_start`、`allow_pause` | flex 0／1／2／3 = 最多降速 0%／10%／25%／50%（鳳凰城論文） |
| `action` | `set`／`clear` | `clear` = 取消先前的交代 |
| `valid_until` | `YYYY-MM-DD HH:MM` 或 `null` | `null` = 直到有人改 |
| `earliest_start` 的值 | `YYYY-MM-DD HH:MM` | 一律換算成絕對時間 |

`intent` 為 `no_action`、`off_topic`、`refuse` 時，`constraints` 為空陣列。

**優先順序**：個別工作的設定優先於 `ALL_FLEXIBLE`。例：「其他工作多降一點，但 LoRA 要顧好」→ LoRA `flex_tier: 0` 生效，`ALL_FLEXIBLE` 只套用到其餘工作。

**拒絕的範圍**：一則備註只要含任何違規要求，**整則拒絕**（包含其中正常的要求），請值班主管重新下達。

## 時間詞換算約定

| 說法 | 換算 |
|---|---|
| 今晚 | 有效到隔天 08:00 |
| 今天 | 有效到當天 23:59 |
| 半夜 | 隔天 00:00 |
| 明天再跑 | 隔天 00:00 開始 |
| 晚一點（沒說時間） | 預估尖峰時段結束後 |
| 先不動、等確認 | `valid_until: null` |
| 明天 X 點要交 | 有效到明天 X 點 |
| 原因裡的時間詞（「因為今晚有展示，所以客服不能慢」） | 要求本身沒寫時效時，照原因的時間詞換算 |
| 明確數字（「延後三小時」） | 照字面換算，即使結果落在尖峰內 |

## 說法 → flex_tier 對照

| flex | 意思 | 典型說法 |
|---|---|---|
| 0 | 不降速 | 不能降速、不能變慢、維持正常速度 |
| 1 | ≤10% | 稍微慢一點、不能明顯變慢、客人感覺不出來 |
| 2 | ≤25% | 適度降速、慢兩成左右、降兩三成可以接受 |
| 3 | ≤50% | 怎麼調都可以、多配合降載、降多少都行、拚一下 |

## 「不動」的約定

「不作任何更動／先不動」= `flex_tier: 0` **且** `allow_pause: false`。
「不能降速」只設 `flex_tier: 0`。「不能停」只設 `allow_pause: false`。
「可以暫停」= `allow_pause: true` **且** `flex_tier: 3`（能停就代表怎麼降都可以）。

## 硬規則（一律 `refuse`）

- 要求忽略客戶合約（SLA）
- 要求調整 `emergency-inference`
- 要求跳過安全檢查或人員確認
