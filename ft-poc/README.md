# ft-poc — 第一次 LoRA POC（Qwen2.5-1.5B-Instruct on M1）

- 日期：2026-10-20 ～ 2026-10-21
- 對應筆記：[notes/05](../notes/05-harness-and-first-lora-poc.md)（概念與機制在那邊；這份只記實作細節、結果與失敗分析）
- 狀態（2026-10-21，v3）：**8 題全對**。目標 3/3、泛化 1/1、拒答 1/1、對照 3/3 回到 before 的樣子。v3 只改訓練參數、資料沒動，細節在 §12；怎麼確認「真的是訓練出來的」在 §13。
- v2（2026-10-21）：**流程跑通，結果失敗**。目標題 3/4、拒答題 0/1、對照題 0/3，掛上 adapter 後中性 prompt 的輸出整個崩掉。§5 到 §9 記的是這一次。
- 一句話：v2 失敗不是資料的鍋（v3 資料一樣），是 **步長**：lr 1e-4 × scale 20 × loss 含 prompt × 掛 16 層 MLP，四個放大器疊在一起。v3 把四個都收掉就過了。但 8 題裡 3 題對照題在訓練集裡，這個「全對」還沒經過 held-out 評測。

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
| `sweep_ckpt.sh` | 對 `adapters/` 每個 checkpoint 跑一次 `run_eval.sh`，輸出 `sweep/ckpt-NNNNNNN.md`，看學習曲線（§13） | ✅ |
| `score_eval.py` | 把任意幾份 `run_eval.sh` 的輸出變成一張分數表：[T] 題對答案、[P] 題跟 before 比（§13） | ✅ |
| `sweep/ckpt-*.md` | sweep 的結果，是實驗記錄，**要進 git**；`sweep/.ck-*/` 是臨時目錄（config + 指到 checkpoint 的 symlink），已 ignore | ✅ / ❌ |
| `lora.yaml` | v3 起用的 LoRA 參數（rank / scale / dropout），CLI 沒有這幾個旗標，只能 `-c` 傳 | ✅ |
| `data/train.jsonl` | 117 筆，mlx_lm chat 格式（`messages` 陣列），v2 / v3 共用 | ✅ |
| `data/valid.jsonl` | 13 筆 | ✅ |
| `before.md` | base 模型 8 題輸出（2026-10-21），v2 / v3 共用 | ✅ |
| `after.md` | 掛 adapter 後同 8 題輸出。**目前是 v3**；v2 的版本在 git 歷史 `0529df6` | ✅ |
| `adapters/adapter_config.json` | 訓練參數，mlx_lm 自動寫出。**目前是 v3**；v2 的在 `0529df6` | ✅ |
| `adapters/*.safetensors` | adapter 權重。v3 每 20 步一份（10 MB，8 層）。v2 殘留的第 200 步已改名 `v2-leftover-0000200_adapters.safetensors`（21 MB，16 層），避免被 `sweep_ckpt.sh` 的 `0*_` glob 撈到配錯 config | ❌ `.gitignore` |

v3 直接覆蓋了 `adapters/` 跟 `after.md`，沒有另開目錄，所以 v2 的最終 adapter 已經沒了。之後每次實驗用不同的 `--adapter-path` 與輸出檔名。

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

> 2026-10-21 更新：A、B、D、F 在 v3 一次全做了（見 §12），結果 8/8。C（通用題擴量）與 E（held-out 評測）還沒做，而且因為一次改了四個變因，**無法歸因**是哪一個救回來的。

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

---

## 12. 實驗 v3（2026-10-21）

資料不動（還是 v2 的 117 / 13），只改訓練參數。四個放大器全部收掉：

| 參數 | v2 | v3 | 動的是哪個軸 |
|---|---|---|---|
| `learning_rate` | 1e-4 | **1e-5** | 步長 |
| `lora_parameters.scale` | 20 | **4**（`lora.yaml`） | 步長 |
| `mask_prompt` | false | **true** | 方向（loss 只算 assistant token） |
| `num_layers` | 16（layers 12–27） | **8**（layers 20–27） | 範圍 |
| `batch_size` | 2 | 4 | 餵法（梯度平均 4 筆，比較穩） |
| `iters` | 200 | 170 | 步數 |
| epochs | 3.4 | **5.8**（170 × 4 / 117；本來是按 230 筆算 3 epochs，但資料沒重產） | |
| `steps_per_eval` / `save_every` | 200 / 100 | 20 / 20 | 觀測 |
| `val_batches` | 25 | -1（全跑） | 觀測 |
| adapter 參數量 | 5,275,648 | **2,637,824**（8 層 × 7 模組） | |
| adapter 檔案 | 21 MB | 10.5 MB | |

指令：

```bash
mlx_lm.lora --model Qwen/Qwen2.5-1.5B-Instruct --train --data data \
  -c lora.yaml --mask-prompt \
  --learning-rate 1e-5 --batch-size 4 --iters 170 --num-layers 8 \
  --steps-per-eval 20 --save-every 20 --val-batches -1 \
  --adapter-path adapters
```

這次**又沒有 `tee`**，train / val loss 曲線第二次沒留下來。

### 每個參數在做什麼

先把訓練迴圈放回來，參數都掛在某一步上：

```
t0  拿 batch_size 筆資料                         ← 餵法：--batch-size / --iters / --mask-prompt
t1  forward：base + LoRA 算預測                   ← 範圍：--num-layers / rank / keys
t2  loss（mask-prompt 開：只算 assistant token）
t3  backward：只算 LoRA 參數的梯度                 ← 方向：資料決定，參數改不了
t4  adam：LoRA -= lr × 梯度；ΔW = scale × A@B      ← 步長：--learning-rate / scale
t5  回 t0；每 steps-per-eval 步跑一次 valid         ← 觀測：--steps-per-eval / --save-every / --val-batches
```

調參順序永遠是 **方向 → 步長 → 範圍 → 步數**。v2 只動了步數，而且往放大的方向動。

| 參數 | 意思 | v3 值 | 為什麼是這個值 | 什麼症狀才動它 |
|---|---|---|---|---|
| `--learning-rate` | t4 的步長係數，梯度乘上它才變成權重的改變量 | 1e-5 | mlx_lm 預設值。v2 的 1e-4 讓模型連訓練集的題都背不起來，那是「步太大跳過了」 | train loss 震盪或不降 → 砍半；loss 降很慢且目標題學不到 → 加倍 |
| `scale`（`lora.yaml`） | 第二個步長係數。mlx 的 LoRA 是 `ΔW = scale × A@B`，等效 PEFT 的 alpha / r | 4 | mlx 預設 20 是配 lr 1e-5 用的；PEFT 慣例 alpha / r ≈ 2。4 是回到 LoRA 論文附近。跟 lr 相乘才是真正的步長 | 跟 lr 一起看；想固定 lr 只縮 LoRA 影響力時動它 |
| `--mask-prompt` | t2 只對 assistant 段算 loss，system / user 段遮掉 | 開 | 關著時 102 筆任務 prompt 也被當目標學，一半以上梯度在教模型「生出 prompt」。SFT 的標準設定 | 永遠開。只有在做 continued pretraining（學文件本身）才關 |
| `--num-layers` | 從最後一層往前數幾層掛 LoRA | 8（layers 20–27） | 越後面的層越接近輸出，改格式夠用；MLP 存事實，掛越多層忘越多 | 目標題學不到 → 放回 16；對照題漂移 → 縮到 4 或改 `keys` 只掛 attention |
| `rank`（`lora.yaml`） | A、B 矩陣的內維度，ΔW 的秩，LoRA 能表達幾條獨立的「規則」 | 8 | 兩欄 schema 用不到 8。它決定參數量，不決定步長 | 任務複雜（正式 DSL）而 loss 卡住 → 16 或 32 |
| `dropout`（`lora.yaml`） | 訓練時隨機丟掉 LoRA 輸入的比例 | 0 | 資料 117 筆、epochs 不到 6，先不加正則 | 真實資料過擬合明顯（val 升）→ 0.05 到 0.1 |
| `--batch-size` | t0 一次拿幾筆，梯度是這幾筆的平均 | 4 | 平均的筆數越多單筆雜訊越小；同一個 batch 抽到通用題的機率也變高，錨才進得了同一步的梯度。代價是記憶體，這批短序列加不到 0.5 GB | OOM → 降；loss 很抖 → 升 |
| `--iters` | 總步數。epochs = iters × batch / N | 170 | 本來按 230 筆算 3 epochs；資料沒重產所以實際 170 × 4 / 117 ≈ 5.8 | 最後才調。val loss 最低點在第幾步，iters 就設到那附近 |
| `--steps-per-eval` | 每幾步跑一次 valid 算 val loss | 20 | 170 步給 8 個點，才看得到曲線。v2 的 200 等於只評最後一次 | 不用動 |
| `--save-every` | 每幾步存一份 checkpoint | 20 | 跟 eval 對齊，val loss 最低那步才有檔可拿 | 磁碟吃緊才放大（每份 10 MB） |
| `--val-batches` | 每次 eval 拿幾個 batch | -1（全跑） | valid 只有 13 筆，全跑才有意義 | 不用動 |
| `--seed` | 資料 shuffle 與 LoRA A 矩陣初始化的種子 | 0 | 比較兩次實驗時必須一樣，否則差異可能來自亂數 | 不動 |
| `-c lora.yaml` | YAML 設定檔，放 CLI 沒有的 `lora_parameters` | | CLI 旗標會蓋過 YAML 同名值，所以 YAML 只放 rank / scale / dropout / keys | |

### 結果

| # | 類 | before | v2 after | v3 after | 判定 |
|---|---|---|---|---|---|
| 1 | 目標 | 自創 schema、`"九五"` 沒換算 | `0.95 / low` | `{"rtp": 0.95, "volatility": "low"}` | ✅ |
| 2 | 目標 | key 捏造 `waveform` | `0.93 / high` | `{"rtp": 0.93, "volatility": "high"}` | ✅ |
| 3 | 目標 | `{"回饋率": 96, "頻道": "中波"}` | ❌ `high` | `{"rtp": 0.96, "volatility": "medium"}` | ✅ v2 錯的這題對了 |
| 4 | 泛化 | `{"RTP": "九八", "波動": "超高"}` | `0.98 / high` | `{"rtp": 0.98, "volatility": "high"}` | ✅ |
| 5 | 拒答 | `{"query": "昨天的 RTP 报表"}` | ❌ 捏造 `0.91 / high` | `{"error": "not a config request"}` | ✅ 一字不差是訓練集的字串 |
| 6 | 對照 | 一段介紹（66 token） | `API\n設定所有 API` | 一段介紹（66 token），內容跟 before 不完全一樣 | ✅ 沒崩 |
| 7 | 對照 | 正確 code + 說明（80 token 截斷） | 壞 JSON | 正確 code + 說明（80 token 截斷） | ✅ |
| 8 | 對照 | `"The weather is very good today."` | `API\nPID 的` | `"The weather is great today."` | ✅ 沒崩，但字變了 |

### 讀法

- 第 5 題是最硬的證據：`{"error": "not a config request"}` 這個字串 base 沒看過（before 吐的是 `{"query": ...}`），只有訓練集裡有，它一字不差出現，只能是學來的。
- 第 6、8 題「變正常」不等於「沒變」：6 題的措辭和 8 題的 `very good` → `great` 都跟 before 不同。adapter 在中性 prompt 下仍然在作用，只是作用小到沒破壞。要量「變了多少」得用 held-out 的對照題，而且判定標準是「跟 before 像不像」而不是「對不對」。
- 5.8 epochs 比預計的 3 多將近一倍，對照題還是沒崩。反推 v2 的問題確實不在 epochs，在步長（上一版 §9 的排序對了）。
- 一次改四個變因所以不能說「是 lr 救的」。要歸因，從 v3 出發一次放回一個：先把 scale 放回 20、再把 num_layers 放回 16、再關 mask_prompt，哪一步對照題開始壞，哪一個就是主因。

### 權重層的學習曲線（不用跑 GPU）

LoRA 的 B 矩陣初始化是全零，`ΔW = scale × A @ B`。每個 checkpoint 直接讀 safetensors 算 norm：

| iter | ‖B‖_F | scale × ‖A@B‖_F | 每 20 步增量 |
|---|---|---|---|
| 20 | 0.903 | 2.118 | |
| 40 | 1.240 | 2.937 | +0.819 |
| 60 | 1.349 | 3.202 | +0.265 |
| 80 | 1.418 | 3.371 | +0.169 |
| 100 | 1.469 | 3.500 | +0.129 |
| 120 | 1.512 | 3.610 | +0.110 |
| 140 | 1.549 | 3.704 | +0.094 |
| 160 | 1.578 | 3.778 | +0.074 |
| 170 | 1.593 | 3.816 | +0.038 |

對照 v2 最終版：‖B‖_F = 9.24、scale × ‖A@B‖_F ≈ 185（scale 20、16 層）。v3 的 ΔW 總量只有 v2 的 **2%**，但目標題學得更好。這就是「方向對了步長小也到得了，方向錯了步長大只會飛出去」。

增量遞減代表接近收斂；第 40 步之後每 20 步的改變已經很小，表示 2 到 3 epochs 時規則大致學完，後面 3 個 epochs 在微調。

### checkpoint sweep：行為層的學習曲線（2026-10-22）

`./sweep_ckpt.sh` 對 8 個 checkpoint 各跑同 8 題，`score_eval.py` 打分：

```
$ python score_eval.py before.md sweep/ckpt-*.md after.md
| file             | T1 | T2 | T3   | T4 | T5 | P6 | P7 | P8 | 目標+泛化+拒答 | 對照沒崩 |
| before.md        | ✗  | ✗  | ✗    | ✗  | ✗  | =  | =  | =  | 0/5 | 3/3 |
| ckpt-0000020.md  | ✗  | ✗  | ✗    | ✅ | ✗  | ≈  | ≈  | =  | 1/5 | 3/3 |
| ckpt-0000040.md  | ✅ | ✅ | ✗    | ✅ | ✗  | ≈  | ≈  | =  | 3/5 | 3/3 |
| ckpt-0000060.md  | ✅ | ✅ | ✗    | ✅ | ✗  | ≈  | ≈  | =  | 3/5 | 3/3 |
| ckpt-0000080.md  | ✅ | ✅ | ✗    | ✅ | ✅ | ≈  | ≈  | =  | 4/5 | 3/3 |
| ckpt-0000100.md  | ✅ | ✅ | ✗dup | ✅ | ✅ | ≈  | ≈  | ≈  | 4/5 | 3/3 |
| ckpt-0000120.md  | ✅ | ✅ | ✗dup | ✅ | ✅ | ≈  | ≈  | ≈  | 4/5 | 3/3 |
| ckpt-0000140.md  | ✅ | ✅ | ✅   | ✅ | ✅ | ≈  | ≈  | ≈  | 5/5 | 3/3 |
| ckpt-0000160.md  | ✅ | ✅ | ✅   | ✅ | ✅ | ≈  | ≈  | ≈  | 5/5 | 3/3 |
| after.md (170)   | 同上 | | | | | | | | 5/5 | 3/3 |
```

T3（「回饋率 96%，中波」）和 T5（拒答）的原文，看它們是怎麼一步步變對的：

| iter | T3 輸出 | T5 輸出 |
|---|---|---|
| 0（before） | `{"回饋率": 96, "頻道": "中波"}` 包在 ` ```json ` 裡 | `{"query": "昨天的 RTP 报表"}` |
| 20 | `{"feedback_rate": 0.96, "frequency": "midwave"}` | 同 before |
| 40 | `{"rtp": 0.96, "volatility": "medium", "volatility_scale": 0.5}` | `{"rtp_report": "rtp_report yesterday"}` |
| 60 | 同 40 | `{"rtp": "rtp_report", "date": "yesterday"}` |
| 80 | `{..., "volatility_scale": "medium"}` 多一個 key | `{"error": "not a config request"}` ✅ |
| 100 | `{"rtp": 0.96, "volatility": "medium", "volatility": "medium"}` **重複 key** | ✅ |
| 120 | 同 100 | ✅ |
| 140 | `{"rtp": 0.96, "volatility": "medium"}` ✅ | ✅ |

### 讀法

- **學習有順序，而且順序是「資料裡多的先學」**：
  1. iter 20：輸出格式（丟掉 ` ```json ` 包裝、英文 key）先變，但值還沒換算（T1 仍是 `"九五"`），泛化題 T4 反而先對。
  2. iter 40：值的換算與兩檔 volatility 對了（T1、T2），91 筆目標題的主規則到這裡學完。這跟 §12 norm 表「40 步之後增量變小」一致。
  3. iter 80：拒答才學會。11 筆反例是最少的類，要走到 2.7 epochs 才壓過「看到 RTP 就吐設定」。
  4. iter 140：多餘 key 才消失。「不要多生東西」是最後學會的，中間還經過「多一個 key」→「重複 key」兩個階段。
- **最早全對的 checkpoint 是 140，不是 100**。`json.loads` 預設吞掉重複 key，第一版 `score_eval.py` 把 100、120 判成 ✅；加了 `object_pairs_hook` 才抓到。評分器也要被驗。
- **對中性 prompt 的影響隨步數累積**：P8（翻譯）到 iter 80 都跟 before 一字不差，100 起變成 `great`；P6 從 40 起措辭就不同。P6 在 iter 20 有一次只生 19 token 的抖動，40 之後恢復。這是「adapter 越練越滲進任務外輸入」的直接證據，只是這次幅度小。
- **挑哪一份**：按 §14 的規則「目標題全對、對照題沒退的最早一份」是 140。它跟 170 在 8 題上沒差，但 ΔW 更小（norm 1.549 vs 1.593）。
- 下次資料有 230 筆、epochs 回到 3 時，這條曲線會整個往右拉，拒答與去重複 key 可能學不完。到時候 iters 要按這張表反推，不是直接照抄 170。

---

## 13. 怎麼驗證「真的被訓練到」

「after 看起來正常」只排除了崩潰，證明不了學習。證據分三層，由弱到強：

| 層 | 方法 | 這次有沒有 | 證明什麼 |
|---|---|---|---|
| 權重 | `lora_b` 非零、norm 隨 checkpoint 單調上升且收斂（§12 的表） | ✅ | 梯度更新發生了、方向穩定。證明不了學到的是對的東西 |
| 行為：before vs after | 同 base、同 prompt、同 greedy 解碼，唯一變因是 `--adapter-path`，輸出不同 | ✅ | 差異只能來自 adapter |
| 行為：訓練集獨有字串 | after 輸出訓練集才有、base 沒看過的字串（第 5 題的 `{"error": ...}`） | ✅ | 這段行為是學來的，不是本來就會 |
| 行為：held-out 泛化 | 訓練集沒出現的組合答對（第 4 題「超高」→ `high`） | ✅ 1 題 | 學到規則不是背題。1 題太少 |
| 曲線：checkpoint sweep | `./sweep_ckpt.sh` 對每個 checkpoint 跑同 8 題，看答對數隨 iter 變化 | ⬜ 還沒跑 | 哪一步學會、哪一步開始退，比最終版一個點有說服力 |
| 曲線：train / val loss | `2>&1 \| tee train.log`，valid 分層含三類 | ⬜ 兩次都沒存 | 過擬合從哪一步開始 |
| 對照：held-out 中性題 | 不在訓練集的對照題，比對 after 與 before 的文字差異 | ⬜ 還沒做 | adapter 對任務外輸入「改了多少」，不只是「有沒有崩」 |
| 對照：拔掉 adapter | 不帶 `--adapter-path` 再跑一次，應該與 before 一字不差 | ⬜ 可以順手做 | 確認 base 權重真的沒動、差異不是環境漂移 |

幾個原則：

- **單一變因**：before / after 之間只能差 adapter。模型版本、prompt 字串、`--max-tokens`、解碼參數都要一樣。`run_eval.sh` 把這些寫死就是為了這個。
- **找訓練集獨有的輸出**：設計資料時刻意放一個 base 絕對不會自己生出來的 token 序列（一個固定錯誤碼、一個自訂 key 名），它變成免費的「有學到」探針。
- **一個點不是曲線**：最終版答對不代表中間沒有先學會再退步。有 checkpoint 就跑 sweep。
- **「正常」要量化**：對照題的判定不是人眼看「像不像人話」，是跟 before 的輸出比。可以先用最笨的「完全相同 / 不同」計數，再升級成 token 重疊率。

### 評測這組工具怎麼用

三個腳本各管一段，產物都是 markdown，放在 `ft-poc/` 下：

```
run_eval.sh   OUT.md [ADAPTER_DIR]     跑 8 題 → 一份 md（before.md / after.md / sweep/ckpt-*.md 都是它產的）
sweep_ckpt.sh [ADAPTER_DIR]            對每個 checkpoint 呼叫 run_eval.sh → sweep/ckpt-NNNNNNN.md
score_eval.py [--ref before.md] FILE…  讀這些 md → 一張分數表（stdout，貼進 README 用）
```

每份 md 的結構都一樣：`## [T|P] 題目` 一段、`==========` 夾著模型輸出、再來 `Generation: N tokens` 與 peak memory。`score_eval.py` 就靠這三個錨點解析，所以**不要手改這些檔**。

判定規則（寫死在 `score_eval.py` 的 `EXPECTED`，改了 `run_eval.sh` 的題目要一起改）：

| 題型 | 怎麼判 | 符號 |
|---|---|---|
| [T] 目標 / 泛化 / 拒答 | 輸出 `json.loads` 後跟期望 dict 完全相等；用 `object_pairs_hook` 抓重複 key | ✅ / ✗ / ✗json（連 JSON 都不是）/ ✗dup（重複 key） |
| [P] 對照 | 跟 `--ref`（預設第一個 `before*.md`）同題輸出比 | `=` 一字不差 / `≈` 不同但沒崩 / `✗` 崩（< 12 token、或中性 prompt 下吐 JSON） |

目前三次的分數：

```
$ python score_eval.py before.md after-v2.md after.md      # after-v2.md 從 git 0529df6 拿
| file        | T1 | T2 | T3 | T4 | T5 | P6 | P7 | P8 | 目標+泛化+拒答 | 對照沒崩 |
| before.md   | ✗  | ✗  | ✗  | ✗  | ✗  | =  | =  | =  | 0/5 | 3/3 |
| after-v2.md | ✅ | ✅ | ✗  | ✅ | ✗  | ✗  | ✗  | ✗  | 3/5 | 0/3 |
| after.md    | ✅ | ✅ | ✅ | ✅ | ✅ | ≈  | ≈  | ≈  | 5/5 | 3/3 |
```

兩欄要一起看：右邊那欄掉到 0/3 就是災難性遺忘，左邊再高都不能上線。`≈` 代表 adapter 在中性 prompt 下仍在作用，只是沒破壞；要知道「改了多少」得用 held-out 的 [P] 題。

典型流程：

```bash
./run_eval.sh before.md                  # 只在評測題改過時重跑
mlx_lm.lora ... --adapter-path adapters-vN 2>&1 | tee train-vN.log
./run_eval.sh after-vN.md adapters-vN
python score_eval.py before.md after-v*.md       # 跨版本比較
./sweep_ckpt.sh adapters-vN                      # 想看學習曲線時
python score_eval.py before.md sweep/ckpt-*.md   # 哪一步學會、哪一步開始退
```

### 驗證這次 v3 的下一步

1. ~~`./sweep_ckpt.sh`~~ 已跑（2026-10-22），結果在 §12：拒答 iter 80 學會、T3 乾淨 JSON 要到 140。
2. 加 10 題不在訓練集的中性題，跑 before 與 after，數有幾題輸出一字不差。
3. 下次訓練一定 `tee`。

---

## 14. checkpoint 怎麼用

### 檔案長什麼樣

```
adapters/
├ adapter_config.json              訓練參數，整個目錄共用一份
├ 0000020_adapters.safetensors     第 20 步的 LoRA 權重（--save-every 20）
├ 0000040_adapters.safetensors
├ ...
├ 0000160_adapters.safetensors
└ adapters.safetensors             最後一步（170）的權重，訓練結束時另存
```

- 每份只有 LoRA 的 A / B 矩陣，不含 base。v3 一份 10.5 MB（8 層），v2 一份 21 MB（16 層）。
- 檔名 `{iter:07d}_adapters.safetensors` 是 mlx_lm 寫死的格式，`adapters.safetensors` 永遠是最終版。
- `.safetensors` 不進 git，`adapter_config.json` 進。要重現某份 checkpoint 就是「同 config + 同 seed + 同資料 + 跑到那一步」。

### 推理時掛某個 checkpoint

`mlx_lm.generate --adapter-path` 要的是**目錄**，不是檔案，目錄裡要有 `adapters.safetensors` 和 `adapter_config.json`。所以挑第 100 步：

```bash
mkdir -p ck-100
cp adapters/adapter_config.json ck-100/
ln -s "$PWD/adapters/0000100_adapters.safetensors" ck-100/adapters.safetensors

mlx_lm.generate --model Qwen/Qwen2.5-1.5B-Instruct --adapter-path ck-100 \
  --system-prompt "你是遊戲設定助理。把口語需求轉成 JSON，只輸出 JSON。" \
  --prompt "RTP 九五，低波動" --max-tokens 80

./run_eval.sh ckpt-100.md ck-100        # 或直接丟給評測腳本
```

`sweep_ckpt.sh` 就是把這三行對每一份 checkpoint 跑一遍，臨時目錄放 `sweep/.ck-*`。

### 挑哪一份

不是最後一份。順序：

1. 有 `train.log` 時，看 **val loss 最低**在第幾步，拿那份。val loss 從最低點開始往上就是開始背題。
2. 沒 log（這次）就跑 `sweep_ckpt.sh`，挑「目標題全對、對照題沒退」的**最早**那份。最早是因為走越少步，對任務外輸入的改動越小。
3. 兩份都過的話，看 §12 的 norm 表，增量已經很小的那段隨便挑都差不多。

### 從 checkpoint 續訓

```bash
mlx_lm.lora --model Qwen/Qwen2.5-1.5B-Instruct --train --data data \
  -c lora.yaml --mask-prompt --learning-rate 1e-5 --batch-size 4 --num-layers 8 \
  --resume-adapter-file adapters/0000100_adapters.safetensors \
  --iters 50 --adapter-path adapters-resumed
```

- `--resume-adapter-file` 只載入 LoRA 權重，**optimizer state 與步數不接續**：adam 從零開始、iter 從 1 數。所以 `--iters 50` 是「再走 50 步」，不是「走到 50」。
- rank / scale / num_layers 必須跟原本一樣，不然 shape 對不上會載入失敗。
- 用途：想從 val loss 最低點用更小的 lr 再磨一下、或加了新資料想接著練。跟從頭跑比的代價是：optimizer 重新暖機，前幾步會有點抖。

### 把 checkpoint 變成可部署的模型

```bash
mlx_lm.fuse --model Qwen/Qwen2.5-1.5B-Instruct --adapter-path ck-100 --save-path fused-100
# → fused-100/ 是合併後的完整 HF 格式權重（safetensors + config + tokenizer，約 3 GB）
#   mlx_lm.generate --model fused-100 直接載；vLLM 也吃這個目錄

# 給 Ollama：mlx_lm.fuse 的 --export-gguf 在 mlx-lm 0.32 只接受 llama / mixtral / mistral，
# model_type=qwen2 會直接 raise。Qwen 要走 llama.cpp 的轉換器：
python llama.cpp/convert_hf_to_gguf.py fused-100 --outfile fused-100/model.gguf --outtype f16
# → ollama create rtp-lora -f Modelfile（Modelfile 內 FROM ./fused-100/model.gguf）
```

- fuse 是 `W' = W + scale × A@B` 算一次寫回去，之後推理不多一次矩陣乘法。代價：每個 checkpoint 一份 3 GB，而且 fuse 完就分不出哪段是 LoRA。
- 不 fuse、直接 `--adapter-path`：多幾 MB、可以隨時換 checkpoint、vLLM 可以 per-request 選 adapter。開發期用這個，上線時再決定要不要 fuse。
- 轉成 GGUF 之後一定用 Ollama 再跑一次同樣 8 題。GGUF 那邊通常會再量化（Ollama 預設拉 Q4），行為可能跟 mlx 上的 fp16 不完全一樣；而且 system prompt 要透過 Modelfile 的 `SYSTEM` 或每次請求帶，必須跟訓練時一字不差。
