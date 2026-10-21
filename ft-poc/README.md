# ft-poc — 第一次 LoRA POC（Qwen2.5-1.5B-Instruct on M1）

- 日期：2026-10-20 ～ 2026-10-21
- 對應筆記：[notes/05](../notes/05-harness-and-first-lora-poc.md)（概念與機制在那邊；這份只記實作細節、結果與失敗分析）
- 狀態（2026-10-21）：**流程跑通，結果失敗**。目標題 3/4、拒答題 0/1、對照題 0/3。掛上 adapter 後，模型在中性 prompt 下的輸出整個崩掉。
- 一句話：資料集 v2 的修正（加反例、加通用題當錨）**沒有擋住災難性遺忘**。主要嫌疑依序是：learning rate 是 mlx_lm 預設的 10 倍、loss 預設把 prompt token 也算進去、通用題太少太短、valid 裡 0 筆通用題所以 val loss 看不到退步。

---

## 1. 目標

把口語機台需求轉成固定 schema 的 JSON，同時不破壞模型原本的能力：

| 輸入情境 | system prompt | 期望輸出 |
|---|---|---|
| 「RTP 九五，低波動」 | 任務 prompt | `{"rtp": 0.95, "volatility": "low"}` |
| 「幫我查昨天的 RTP 報表」 | 任務 prompt | `{"error": "not a config request"}` |
| 「台灣的首都是哪裡？」 | 中性 prompt | 正常回答 |

這是公司 DSL fine-tune 的縮小版替身：schema 只有兩個欄位，但「讀輸入而不是認模板」「不相關輸入要拒答」「不能忘掉通用能力」三件事跟正式版一樣。

---

## 2. 環境

| 項目 | 值 |
|---|---|
| 硬體 | M1 Mac, 16 GB unified memory |
| Python | 3.13.7，venv 在 repo 根目錄 `ft/`（不進 git） |
| mlx / mlx-lm / mlx-metal | 0.32.4 / 0.32.2 / 0.32.4 |
| base model | `Qwen/Qwen2.5-1.5B-Instruct`：28 層、hidden 1536、權重 bf16，載入約 3 GB |
| 訓練時 | Ollama 與 Open WebUI 關閉，釋放記憶體 |
| 實測 | generation 18 tok/s、peak memory 3.2 GB（訓練與推論都在這附近） |

---

## 3. 檔案

| 路徑 | 內容 | 進 git |
|---|---|---|
| `gen_data.py` | 資料產生器 v2，固定 `random.seed(42)`，產 `data/train.jsonl`、`data/valid.jsonl` | ✅ |
| `run_eval.sh` | 固定 8 題評測，`T\|` 用任務 prompt、`P\|` 用中性 prompt，輸出 markdown | ✅ |
| `data/train.jsonl` | 117 筆，mlx_lm chat 格式（`messages` 陣列） | ✅ |
| `data/valid.jsonl` | 13 筆 | ✅ |
| `before.md` | base 模型 8 題輸出（2026-10-21 17:23） | ✅ |
| `after.md` | 掛 adapter 後同 8 題輸出（2026-10-21 17:31） | ✅ |
| `adapters/adapter_config.json` | 這次訓練的完整參數，mlx_lm 自動寫出 | ✅ |
| `adapters/*.safetensors` | adapter 權重，iter 100 / 200 各一份 + 最終版，每份 21 MB | ❌ `.gitignore` |

---

## 4. 跑法

所有指令在 `ft-poc/` 下執行：

```bash
source ../ft/bin/activate

python gen_data.py                 # 目標 100 / 拒答 15 / 通用 15 → train 117 / valid 13
./run_eval.sh before.md            # base 模型，8 題

mlx_lm.lora --model Qwen/Qwen2.5-1.5B-Instruct --train --data data \
  --iters 200 --batch-size 2 --learning-rate 1e-4 --adapter-path adapters

./run_eval.sh after.md adapters    # 掛 adapter，同 8 題
```

這次訓練的 stdout **沒有存下來**，所以沒有 train / val loss 曲線可看。下次一定要 `2>&1 | tee train.log`。

---

## 5. 資料集設計（v2）

### 三類資料

| 類 | 筆數 | system prompt | 輸入 | 期望輸出 | 目的 |
|---|---|---|---|---|---|
| A 目標題 | 100 | 任務 | 8 個模板 × 5 種 RTP 標籤 × 9 個 RTP 值（每值 4–5 種寫法：`96` / `0.96` / `九六` / `96%`）× 3 檔波動（每檔 4–5 種說法），30% 機率插入「拉到 / 調到 / 設」 | `{"rtp": 0.96, "volatility": "high"}` | 隨機組合逼模型讀輸入，不是認模板 |
| B 拒答題 | 15 | 任務 | 首都 / 查報表 / 翻譯 / 寫 code / 訂便當… | `{"error": "not a config request"}` | 教模型任務 prompt 下也要判斷相關性 |
| C 通用題 | 15 | 中性 | 跟 B 同類的問題 | 正常答案（「台北。」「`for i in range(1, 6): ...`」） | 防遺忘的錨；B / C 同題不同 prompt → 教「看 system prompt 決定行為」 |

shuffle 後取前 10% 當 valid。

### 切完之後的實際分佈（這是問題之一）

| | 目標 | 拒答 | 通用 | 合計 |
|---|---|---|---|---|
| train | 91 | 11 | **15** | 117 |
| valid | 9 | 4 | **0** | 13 |

- valid 裡一筆通用題都沒有 → **val loss 完全量不到遺忘**。拒答題 15 筆有 4 筆被切進 valid，train 只剩 11 筆反例。
- 通用題 15 筆只有 7 種題型（首都、翻譯、算術、Python、bash、名詞解釋、Git），而且答案極短。
- 兩句 system prompt 進了訓練資料，`run_eval.sh` 跟之後 serve 時都必須一字不差。

### token 比例

train 的 prompt（system + user）總字元 5,246，assistant 答案總字元 3,911。配合下一節的 `mask_prompt: false`，超過一半的 loss 其實花在「預測 prompt 自己」上。

---

## 6. 訓練設定

來自 `adapters/adapter_config.json`，對照 mlx-lm 0.32.2 的預設值：

| 參數 | 本次 | mlx_lm 預設 | 備註 |
|---|---|---|---|
| `iters` | 200 | 1000 | epochs = 200 × 2 / 117 ≈ 3.4 |
| `batch_size` | 2 | 4 | |
| `learning_rate` | **1e-4** | 1e-5 | **預設的 10 倍** |
| `lora_parameters.rank` | 8 | 8 | |
| `lora_parameters.scale` | 20.0 | 20.0 | mlx 的 `delta = scale × B @ A`，等效 PEFT 的 alpha / r = 20（PEFT 常見是 2） |
| `lora_parameters.dropout` | 0.0 | 0.0 | |
| `num_layers` | 16 | 16 | 只掛最後 16 層（layers 12–27，共 28 層） |
| 掛的模組 | q / k / v / o / gate / up / down | 同 | **attention 和 MLP 的 7 個 linear 全掛**，不是只有 attention |
| `mask_prompt` | false | false | **loss 包含 prompt token** |
| `max_seq_length` | 2048 | 2048 | 這批資料遠小於此 |
| `optimizer` | adam | adam | |
| `steps_per_eval` | 200 | 200 | 等於只在最後評一次 val |
| `val_batches` | 25 | 25 | 25 × 2 = 50 > 13，整份 valid 都會跑 |
| `save_every` | 100 | 100 | 所以留了 iter 100 的 checkpoint |
| `seed` | 0 | 0 | |

### adapter 產物

| 項目 | 值 |
|---|---|
| tensor 數 | 224（16 層 × 7 模組 × A/B） |
| 參數量 | 5,275,648 ≈ 5.276M，佔 base 1.54B 的 0.34% |
| 存檔精度 | **float32** → 21 MB（notes/05 寫 ≈ 10 MB 是按 fp16 估的；檔案格式 safetensors 不變，是精度不同） |
| `lora_b` | 112 個 B 矩陣全部非零（初始化是全零，證明梯度更新發生了） |

---

## 7. 評測設計（`run_eval.sh`，8 題）

| # | 類 | prompt | 題目 | 期望 |
|---|---|---|---|---|
| 1 | 目標 | 任務 | 把這句轉成設定：RTP 九五，低波動 | `0.95 / low` |
| 2 | 目標 | 任務 | 轉設定：這台 RTP 拉到 93，波動高 | `0.93 / high` |
| 3 | 目標 | 任務 | 回饋率 96%，中波，寫成設定 | `0.96 / medium` |
| 4 | 泛化 | 任務 | RTP 九八，波動超高 | 「超高」訓練資料沒出現；期望 `0.98 / high` |
| 5 | 拒答 | 任務 | 幫我查一下昨天的 RTP 報表 | `{"error": ...}` |
| 6 | 對照 | 中性 | 台灣的首都是哪裡？ | 跟 before 差不多 |
| 7 | 對照 | 中性 | 寫一個 Python for 迴圈印 1 到 5 | 跟 before 差不多 |
| 8 | 對照 | 中性 | 把「今天天氣很好」翻成英文 | 跟 before 差不多 |

已知的設計缺陷（不影響「失敗」這個結論，但讓歸因不乾淨）：

- 第 6–8 題**原封不動出現在訓練集的通用題裡**，不是 held-out 對照組。
- 拒答題只 1 題、目標題只 3 題，沒辦法算答對率，只能看現象。
- `--max-tokens 80` 對照題會被截斷（before 的第 7 題就是），判「有沒有退步」要看前幾行。

---

## 8. 結果：before vs after

| # | 類 | before（base） | after（掛 adapter） | 判定 |
|---|---|---|---|---|
| 1 | 目標 | ` ```json ` 區塊包著 `{"setting": {"rtp": "九五", "volatility": "低"}}`，自創 schema、值沒換算 | `{"rtp": 0.95, "volatility": "low"}` | ✅ |
| 2 | 目標 | `{"setting": {"rtp": 93, "waveform": "high"}}` key 捏造 | `{"rtp": 0.93, "volatility": "high"}` | ✅ |
| 3 | 目標 | `{"回饋率": 96, "頻道": "中波"}` | `{"rtp": 0.96, "volatility": "high"}` | ❌ 中波 → high |
| 4 | 泛化 | `{"RTP": "九八", "波動": "超高"}` | `{"rtp": 0.98, "volatility": "high"}` | ✅ 有抓到規則 |
| 5 | 拒答 | `{"query": "昨天的 RTP 报表"}` | `{"rtp": 0.91, "volatility": "high"}` | ❌ 捏造數字 |
| 6 | 對照 | 一段（立場有問題但是完整句子的）介紹 | `API\n設定所有 API` | ❌ 崩 |
| 7 | 對照 | 正確的 `for i in range(1, 6): print(i)` 加說明 | `{"rtp": 0.92，印用率 0.91, "volatility": "rtp": 0.91, ...}` | ❌ 壞 JSON、重複 key、全形逗號 |
| 8 | 對照 | `"The weather is very good today."` | `API\nPID 的` | ❌ 崩 |

### 讀法

- **格式學會了**：base 本來就會吐 JSON，只是 schema 亂、值不換算；adapter 把 schema 壓成固定兩欄、`九五` → `0.95` 的換算對。這部分是真的教會的，不是本來就會（before 證明了這點）。
- **volatility 分類不穩**：第 3 題「中波」判成 high。valid 裡 9 筆目標題沒救到它，因為 val loss 只在最後評一次且沒存。
- **拒答沒學到**：11 筆反例壓不過 91 筆目標題，模型遇到不相關輸入還是走「吐一組看起來合理的 JSON」的捷徑。
- **對照題不是「退步」是「崩潰」**：輸出只有 5–6 個 token，內容是訓練集裡的碎片（「API」來自通用題「用一句話解釋什麼是 API」、「設定」來自任務 prompt）。這三題明明在訓練集裡，模型連背都沒背起來——代表不是通用題沒學到，是整個輸出分佈被拉到「短、JSON 味、設定味」那邊。
- 拔掉 adapter 立刻回到 before：base 一個 bit 沒動，壞的是這 0.34%。

---

## 9. 失敗分析（按嫌疑排序）

### 1. learning rate 1e-4 配 scale 20：更新幅度太大

- lr 是 mlx_lm 預設 1e-5 的 10 倍；scale 20 在 PEFT 語境等於 alpha / r = 20，常見設定是 2。兩個相乘，每一步對 ΔW 的推力比「標準 LoRA 配方」大一個數量級以上。
- LoRA 掛在 16 層的 attention **和 MLP** 共 7 個 linear 上。MLP 是模型存事實與通用能力的主要地方，被大步伐拉走，對照題崩掉不意外。
- notes/05 §8 的「資料形狀決定梯度方向，iters 只放大」在這裡要補一句：**lr × scale 也是放大器**，而且這次放大倍數比 iters 更大。

### 2. loss 把 prompt token 算進去（`mask_prompt: false`）

- 117 筆裡 102 筆的 system prompt 是「你是遊戲設定助理。把口語需求轉成 JSON，只輸出 JSON。」，每一筆都要模型學著預測這句。
- prompt 字元 5,246 vs 答案字元 3,911，超過一半的梯度在教模型「生出任務 prompt 和機台需求句」。這把模型的「世界」整個推向設定檔情境，通用題 15 筆的答案在總 loss 裡的占比極小。
- SFT 的標準做法是只對 assistant token 算 loss，這次沒開。

### 3. 通用題太少、太短、太單一

- 15 筆、7 種題型，當不了錨。
- 答案是「台北。」「2。」「21。」這種 2–3 token 的句子，跟 base 自然輸出的長度（第 7 題 before 是 80 token 還沒講完）差太多。模型學到的是「中性 prompt 下也要答得極短」，而不是「中性 prompt 下保持原樣」。after 第 6、8 題只生 5–6 token 就是這個訊號。

### 4. valid 看不到問題、訓練過程沒留痕

- valid 0 筆通用題 → val loss 對遺忘完全盲。
- `steps_per_eval 200` 只在最後評一次；stdout 沒存 → 這次連「train loss 幾、val loss 幾」都說不出來。這是流程問題，但它讓 1–3 全部無法用數字驗證，只能靠 8 題評測看現象。

### 5. 評測設計讓結論不乾淨（不是失敗原因）

- 對照題在訓練集裡；拒答題 1 題；目標題沒覆蓋三檔波動各至少一題。
- 要把「訓練壞了」跟「題目剛好踩雷」分開，需要更多題、而且題目要 held-out。

---

## 10. 下一步實驗

原則：**一次只改一個變因**，每次都重跑 `after`，before 不用重跑（除非改了評測題）。

| 順序 | 改什麼 | 預期 | 代價 / trade-off |
|---|---|---|---|
| A | `--learning-rate 1e-5`（其他全不動） | 最便宜的實驗，直接驗證嫌疑 1 | 目標題可能學不到位，要補 iters；但若對照題不崩了，方向就確定 |
| B | 加 `--mask-prompt` | loss 只算 assistant token，是 SFT 該有的設定 | 每 iter 有效訊號變少，可能要更多 iters；prompt 的分佈不再被學，serve 時 system prompt 仍須一字不差 |
| C | 通用題加到 50–100 筆、答案用 **base 模型自己生成**（self-distill） | 錨的分佈跟 base 一致，不會教「答短一點」 | 產資料要跑推論（1.5B 在 M1 上一題 2–4 秒，100 題幾分鐘）；要人工抽查 base 答案沒亂講 |
| D | valid 分層抽樣（三類各留 10%）、`--steps-per-eval 20`、`tee train.log` | 讓 val loss 能看見遺忘，留下曲線 | 訓練時間多幾個 eval 的成本，可忽略 |
| E | 評測：對照題換成不在訓練集的題、拒答題 3–5 題、目標題每檔波動至少一題、加一段 `json.loads` 自動判分 | 可以算答對率，不用人眼 | 改了題目就要重跑 before |
| F | 若 A–C 之後對照題還崩：`scale` 降到 2–4，或只掛 attention（`-c config.yaml` 的 `lora_parameters.keys`，2026-10 的 mlx-lm 0.32 支援） | 縮小 ΔW 的作用範圍 | 表達力下降，目標題可能需要更多資料 |

不要做的：

- 加 iters 或 epochs——方向錯了放大只會更錯。
- 換更大的 base——先把 1.5B 的實驗做乾淨，不然到 14B 上一樣的坑只是更貴。
- 同時改 A + B + C——改好了也不知道是誰的功勞。

---

## 11. 這次確認的事

- fine-tune 的本體是資料集與評測設計；指令只有一行。但**預設參數也是設計的一部分**，`mask_prompt`、`scale`、掛哪些模組這三個沒看就跑，是這次最大的疏忽。
- before 不能省：沒有 before 就看不出「JSON 格式本來就會，schema 和換算是教會的」。
- 只測目標題會誤判成功：這次目標題 3/4，光看這個會以為可以上線。
- 檔案格式 ≠ 數值精度，再撞一次：adapter 是 safetensors 沒錯，但存的是 float32，所以是 21 MB 不是 10 MB。
