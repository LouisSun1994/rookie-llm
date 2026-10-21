# 05 — Harness 的邊界與第一次 LoRA POC

- 日期：2026-10-20 ～ 2026-10-21
- 學習方式：從「Ollama 答不出今日股價」這個現象往下挖，經過 harness / 引擎 / 模型的職責劃分、改模型認知的三條路，最後在 M1 上跑完第一次 LoRA，親眼看到過擬合與退步
- 目前定位：🥚 概念已走完 → 🐣 本機驗證（Ollama + Open WebUI 已跑起）→ 🐊 fine-tune 第一次動手（POC 跑通，正在修資料集）
- 相關檔案（repo 根目錄下）：`ft/`（venv，不進 git）、`ft-poc/gen_data.py`、`ft-poc/run_eval.sh`、`ft-poc/before.md`、`ft-poc/after.md`、`ft-poc/adapters*/`
- 本次（資料集 v2）的完整跑法、訓練設定、before / after 結果與失敗分析：`ft-poc/README.md`

---

## 0. 一句話版本

模型是純函數 `tokens → tokens`，它能用的資訊只有權重和 prompt；一切「取資料、執行動作、決定塞什麼進 prompt」都在 harness；fine-tune 是離線產生新權重檔（或 adapter），引擎只負責載入；訓練的本體是資料集設計，不是指令。

---

## 1. 分層模型（骨架）

```
應用層 harness     Open WebUI / 自寫 service / agent loop
                   持有：工具、狀態、權限、迴圈、RAG 編排、評測、log
                   ↕ HTTP（OpenAI 相容 API）
推理引擎           Ollama / vLLM / llama.cpp server
  ├ server         排程、continuous batching、KV cache 配置、logit_bias / grammar
  └ engine         attention kernel、量化矩陣乘法
                   ↕ 載入檔案
模型               權重檔 + tokenizer（fine-tune 改的是這裡）
                   ↕ 離線產生
訓練工具鏈         mlx_lm / PEFT / LLaMA-Factory / TRL  （角色 = PEFT）
                   mlx / PyTorch                        （框架 = 張量 + 自動微分）
                   Metal / CUDA                         （GPU API = driver 在這之下）
GPU                M1 GPU cores / NVIDIA SM
```

判斷「一個東西屬於哪層」的標準不是離誰近，而是**變動的理由是否相同**：換模型時誰要動、加文件時誰要動、擴容方式是加 GPU 還是加磁碟。

| 問題 | 層 | 理由 |
|---|---|---|
| 答不出今日股價 | harness 缺功能 | 沒有元件去取資料；模型拒答是正確行為 |
| 向量庫、RAG 邏輯 | harness（向量庫是獨立元件） | 有狀態、要備份、要依身分過濾；引擎無狀態 |
| 工具呼叫的執行 | harness | 模型只能吐 token，side effect 全在應用層 |
| 語法合法性（DSL） | 引擎 server 層（grammar） | 在 logits 上加約束，確定性 |
| 語意正確性 | 模型（fine-tune） | 機率性 |
| 黑話知識 | prompt（RAG） | 真理在文件裡，每次塞進 t0 |

---

## 2. 該自己實作哪一層

| 層 | 實作來上線 | 實作來理解 |
|---|---|---|
| engine kernel | ❌ GPU 專家戰場，以年計 | ✅ toy decode loop（幾百行 Python）能把 KV cache、batch 維度變肌肉記憶 |
| engine server（排程） | ❌ 用 vLLM | ✅ 要看得懂 `scheduler.py`，因為調的每個參數都在控制它 |
| harness | ✅ 這是產品本身，後端背景的主場 | — |

40 人聊天階段用 Open WebUI 是標準做法（2026-10 時：BSD-3 + branding 條款，50 人以下可改品牌，保留品牌則不限人數免費商用）。自寫 harness 的觸發訊號：依身分過濾文件、接內部 API 當工具、要批次跑評測、嵌進既有流程、審計 log schema 自訂。RAG 中段與 DSL 階段幾乎每條都會踩到。

後端工程師自寫 harness 會低估的 30%：評測沒有「通過」只有「答對率」、prompt 是沒型別的程式碼、非確定性、context 預算、失敗是靜默的（檢索錯 chunk 不報錯）。

---

## 3. 資料來源只有兩個：權重、prompt

```
模型能用的資訊 = 權重裡壓的（訓練截止前） + prompt 裡給的（harness 塞的）
沒有第三個來源。frontier model 本地跑、沒工具，一樣答不出今日收盤。
```

強模型的差別：拒答更精準、更難被誘導亂掰、通用分析講得更好——但那是「如何分析」的知識，不是「今天的資料」。Open WebUI 會在 system prompt 塞當前日期；知道日期 ≠ 知道那天發生的事。

---

## 4. Harness 與 agent loop

```python
# agent loop 的全部骨架
messages = [system, user_prompt]
for i in range(MAX_STEPS):                      # 迴圈：harness 持有
    reply = engine.chat(messages, tools=TOOLS)  # 判斷：模型做
    if reply.tool_calls is None:
        return reply.content                    # 模型決定結束
    for call in reply.tool_calls:
        result = run_tool(call)                 # 執行：harness 做，權限在這
        messages.append(tool_result(result))    # 狀態：harness 持有，每輪重送
```

| 模式 | 誰決定查 | 誰執行 | 模型需要什麼 |
|---|---|---|---|
| 固定 pipeline（Open WebUI web search、基本 RAG） | harness，每次都查 | harness | 會讀 context |
| tool calling | 模型吐結構化呼叫 | harness | 訓練過 tool calling |
| multi-agent | harness 編排多個模型呼叫 | harness | 同上且要穩 |

- 模型吐出的 `{"name":"search",...}` 是**長得像 JSON 的文字**，不是 HTTP request；harness parse 它、決定執行，那一行是整個系統的安全邊界。
- 每一輪是獨立 HTTP 請求，引擎每次重新 prefill 整段 messages；KV cache 是單次請求的暫存，不是 agent 的記憶。
- 這不是「湧現」（emergence = 規模帶來的新能力），是 orchestration，每步可 trace。

---

## 5. RAG 一次查詢

```
t0  使用者問「RTP 設定檔格式」                 CPU / harness
t1  harness 把問題丟給 embedding 模型            CPU→GPU / 引擎 A
t2  拿到 768 維向量                              引擎 A → harness
t3  harness 拿向量查向量庫 top-5 chunk           CPU / 向量庫
t4  harness 把 chunk 塞進 prompt                 CPU / harness
t5  丟給生成模型                                 CPU→GPU / 引擎 B
t6  生成模型讀 chunk、回答                        GPU / 模型
```

引擎在 t1、t5 各跑一次 forward，它不知道自己在做 RAG。「RAG」只存在於 harness 視角。RAG 的所有旋鈕（chunk 大小、top-k、rerank、prompt 排法）都是「塞什麼進 prompt」，引擎看到的只是最終文字。

---

## 6. 改模型認知的三條路（蘋果→香蕉）

一次 decode step 的最後幾行：

```
t0  prompt + 已生成 token 餵進模型                              引擎
t1  權重 × 輸入，幾十層，得到 hidden state   ← 「模型在想什麼」    模型
t2  hidden × lm_head → logits（每個 token 一個分數）             模型
t3  引擎加 bias / grammar 遮罩            ← logit_bias 只動這行   引擎
t4  抽樣選 token                                                引擎
t5  接回輸入，回 t0
```

| 改哪裡 | 進入點 | 改認知？ | 持久？ | 正當用途 |
|---|---|---|---|---|
| prompt / context（RAG） | t0 | ✅ 本次請求 | ❌ | 知識、每次不同的事實 |
| 權重（fine-tune） | t1 | ✅ | ✅ | 行為、格式、風格，把每次都要寫的 few-shot 壓進去 |
| logit_bias / grammar | t3 | ❌ 只管出口 | — | 認知已對、不准寫錯格式（DSL 語法） |

- 用 logit_bias 強迫 banana，模型 t1 仍想著 apple，下一步會說「banana is red and crunchy」——被捂嘴的模型在掙扎。
- 「概念」在模型裡沒有地址：tokenizer（可能切碎）、embedding 行、散在幾十層的關聯、lm_head。fine-tune 是拉扯，不是定址寫入；改內蘊是三條路裡**最不可控**的，只能靠評測量它改了多少、傷了多少。
- 公司黑話與 DSL 是空白不是錯誤認知，所以不存在「用錯的大腦推論」：黑話走 RAG、寫法走 fine-tune、語法走 grammar。

---

## 7. Fine-tune 落在哪一層

- fine-tune 是**訓練工具鏈**的動作，改的是**模型**（權重檔），引擎只在 t5 載入。
- 本次做的是 SFT（supervised fine-tuning）：直接給 `輸入 → 輸出` 配對讓模型模仿。「獎勵」是 RL 的詞，現在不碰。
- LoRA 產出 = `adapters.safetensors`（本次 5.276M 參數 ≈ 10MB，佔 1543M 的 0.342%），base 一個 bit 沒動；掛上 `--adapter-path` 推理，或 `mlx_lm.fuse` 合併 → 轉 GGUF 給 Ollama。
- safetensors 是**檔案格式**；裡面是 fp16 還是 4-bit 是**數值精度**，兩件事。
- 工時分佈：資料 60% / 評測 25% / 訓練 15%。前兩塊是後端主場。

判斷要不要 fine-tune：列得出「每次都要塞進 prompt 的東西」才需要；先用 few-shot 測 base 寫不寫得出 DSL，寫不出 → 換 base 不是 fine-tune。

---

## 8. 訓練機制

```
iter   = optimizer 更新一次權重（看 batch_size 筆）
epoch  = 全部 N 筆看完一遍 = iters × batch_size / N
loss   = 預測 vs 期望答案的差距（成果）；epoch 是投入量，不衡量任何東西
```

一個 iter 內部（`mlx_lm.lora --train` 迴圈裡，沒有獨立指令）：

```
t0  拿一筆（prompt + 期望答案）                                  CPU / mlx_lm
t1  forward：base + 目前 LoRA 算每個位置的預測                    GPU / mlx
t2  loss：一個數字                                               GPU / mlx
t3  backward：只算 LoRA 參數的梯度（base 不算，所以省 VRAM）        GPU / mlx
t4  adam 用梯度推 LoRA 一小步（lr 決定步長）                       GPU / mlx
t5  回 t0
```

- 反向傳播的唯一可見痕跡：`train_loss` 往下。證據：LoRA 的 B 矩陣初始化全零，訓練後非零。
- 過擬合 = train loss 持續降、val loss 開始升 → 模型從「學規則」切成「背題目」。模型容量 ≫ 資料規則量時，規則幾個 epoch 學完，梯度仍非零，剩下的更新只在壓具體字句。
- 解法是**加資料**不是減 epoch；真實資料抓 2–5 epochs。

記憶體估算（Qwen2.5-1.5B，M1 16GB）：
```
fp16 權重  1.5B × 2 byte ≈ 3.0 GB
LoRA 參數  5.3M × 2 byte ≈ 10 MB，梯度 + adam state ≈ 3 倍
實測 peak 3.2 GB；18 tok/s ≈ 3 GB / 68 GB/s 頻寬估算的 22 tok/s
```

---

## 9. 第一次 POC 記錄

環境：repo 根目錄下 `ft/` 為 venv（mlx-lm），`ft-poc/` 為 POC 專案，所有指令在 `ft-poc/` 下執行；Ollama / Open WebUI 關閉釋放記憶體。

```bash
source ../ft/bin/activate
python gen_data.py                                   # 產 data/train.jsonl, valid.jsonl
./run_eval.sh before.md                              # 原始模型，8 題
mlx_lm.lora --model Qwen/Qwen2.5-1.5B-Instruct --train --data data \
  --iters 200 --batch-size 2 --learning-rate 1e-4 --adapter-path adapters
./run_eval.sh after.md adapters                      # 掛 adapter，同 8 題
```

### 資料集設計（`gen_data.py`）

```
A 目標題 100   模板 × RTP 寫法（96 / 0.96 / 九六 / 96%）× 波動說法 隨機組合 → {"rtp", "volatility"}
B 拒答題  15   任務 prompt + 不相關輸入（首都、查報表、翻譯…）→ {"error": "not a config request"}
C 通用題  15   中性 prompt + 同類問題 → 正常回答，防遺忘的錨
shuffle 混合，10% 當 valid（13 筆，能當 overfitting 偵測器）
epochs = 200 × 2 / 117 ≈ 3.4
```

B 與 C 同一問題、不同 system prompt、不同期望答案 → 教模型「看 system prompt 決定行為」。A 的隨機組合逼模型去讀輸入，而不是認模板。

### 評測題集（`run_eval.sh`）

```
[T] 目標題 ×3   任務 prompt，訓練資料同類型
[T] 泛化題 ×1   「RTP 九八，波動超高」資料裡沒出現的組合
[T] 拒答題 ×1   「幫我查昨天的 RTP 報表」期望 {"error":...}
[P] 對照題 ×3   中性 prompt：首都 / 寫 Python 迴圈 / 翻譯，看有沒有退步
```

### 第一次跑出來的現象（4 筆資料、全一種形狀、25 epochs）

設計終版之前先用 4 筆資料跑通流程，結果是這條線最重要的觀察：

| 題型 | 結果 | 讀法 |
|---|---|---|
| 目標題 3/3 | JSON 對、數字對 | 訓練成功 |
| 泛化題 1/1 | 「超高」→ high | 有抓到規則，非純背 |
| 對照題 0/3 | 首都/翻譯/寫 code 全變 `{"rtp": 0.9x}`，數字捏造，JSON 出現重複 key、漏引號 | 災難性遺忘 + 格式崩 |

原因：資料只有一種形狀、沒有反例 → 降 loss 的捷徑是「不讀輸入，吐模板」；25 epochs 放大；對照題當時也帶了任務 system prompt（混淆因子）。終版的 B/C 類資料與中性 prompt 對照題就是對這三點的修正。

效能數字（1.5B fp16，M1 16GB）：Generation 18 tok/s、Peak memory 3.2 GB、system prompt 約 30 token 每題重付。

---

## 10. 評測設計規則

- before 不能省：訓練後看到 JSON，無法分辨是教會的還是本來就會。對照題 prompt 一改，before 要重跑。
- 對照組與實驗組只差一個變因。
- 題集至少三類：目標 / 泛化（資料沒出現的組合）/ 對照（退步偵測）；加入拒答題測「不相關輸入」。
- 只測目標題會以為成功，上線後第一個不相關問題就回捏造 JSON。
- system prompt 進了訓練資料，serve 時必須帶同一句，否則分佈不一致。

---

## 思維鏈：fine-tune 完整生命週期

```
t0  harness log 撈配對，清成 jsonl（gen_data.py）               CPU / 應用層
t1  跑評測存 before.md：base 答對率當基準                        CPU+GPU / 應用層驅動引擎
t2  mlx_lm.lora 載 base + jsonl，跑 LoRA                        GPU / 訓練工具鏈（引擎不需在跑）
      └ 每 iter：forward → loss → backward(只 LoRA) → adam 更新
t3  輸出 adapters/adapters.safetensors（~10MB）                   磁碟 / 模型
t4  （給 Ollama 用時）fuse → 轉 GGUF                              CPU / 工具鏈
t5  引擎載入：vLLM --lora-modules / ollama create                 引擎
t6  harness 換 model 名、刪掉原本塞的 few-shot                    應用層
t7  再跑同一題集存 after.md：確認目標題上升、對照題沒退步           應用層
```

---

## 陷阱清單

- ❌ 「答不出今日股價是 model 壞了」→ ✅ harness 缺取資料的元件，模型拒答是正確行為 → 模型只有權重和 prompt 兩個資訊來源。
- ❌ 「web search 能用所以全是 harness 的事，不關模型」→ ✅ 分工：harness 取資料、執行；模型判斷要不要呼叫、吐合法格式、讀懂結果 → 爛模型配完美 search 回答一樣爛。
- ❌ 「harness 只是一層工具」→ ✅ harness 是產品本身，引擎是可替換零件 → 業務邏輯、權限、評測全在 harness。
- ❌ 「向量庫放引擎才合理」→ ✅ 向量庫是獨立有狀態元件，harness 編排 → 等於問 Redis 為什麼不內建在 PostgreSQL；兩邊變動理由不同。
- ❌ 「agent 的過程是湧現」→ ✅ 是 orchestration，逐步可 trace → emergence 專指規模帶來的能力。
- ❌ 「權重 server」→ ✅ 權重是檔案，server 是引擎。
- ❌ 「fine-tune 改的是引擎上的權重」→ ✅ 改的是模型層，由訓練工具鏈執行，引擎只載入 → 引擎 ≠ 模型。
- ❌ 「harness 可以黑名單 token」→ ✅ harness 看到的是文字，token 層在引擎；harness 做字串替換。
- ❌ 「笨的字串替換 / logit_bias 沒意義」→ ✅ 用途是「認知已對、鎖格式」，不是改認知；DSL 語法靠 grammar → 智能的模型 + 笨的邊界 = 可測試的系統。
- ❌ 「logit_bias 和改權重差不多」→ ✅ logit_bias 動 t3 出口，權重動 t1 認知 → 捂嘴的模型下一步會自相矛盾。
- ❌ 「期望獎勵，遇到 A 給 B」→ ✅ 這是 SFT，獎勵是 RL 的詞。
- ❌ 「mlx_lm 是控制 M1 ALU 的 driver」→ ✅ mlx_lm 是訓練工具鏈，下面還有 mlx（框架）、Metal（GPU API）；跑的是 GPU core 不是 ALU。
- ❌ 「沒看到反向傳播，應該要再修正」→ ✅ 它在 `--train` 迴圈裡發生了 100 次，adapter 檔就是修正的總和 → 痕跡是 loss 下降、B 矩陣從零變非零。
- ❌ 「epochs 是精度 / 複雜度指標，越高越精準」→ ✅ epoch 是曝光次數（投入量），loss 才是成果；多了只對訓練集更貼 → 「精度」這個字已經有 fp16 的意思，別重用。
- ❌ 「overfitting 是因為 iters=100，改成 1 就好」→ ✅ 方向由資料決定，iters 是放大器；iters=1 等於沒學 → 解法是加資料、加反例，epochs 抓 2–5。
- ❌ 「退步是整個權重歪掉」→ ✅ base 沒動，歪的是 0.34% 的 LoRA，但它掛在每層 attention 上足以劫持 → 拔掉 adapter 即回到 before。
- ❌ 「目標題全對 = 成功」→ ✅ 對照題全退化才是這次 POC 最值錢的結果 → 沒有對照組的評測會讓你把壞模型上線。
- ❌ 對照題帶任務 system prompt → ✅ 對照組只能差一個變因 → 退步要能歸因到訓練還是 prompt。
- ❌ 「Open WebUI 因為 MIT 所以改寫有限制」→ ✅ 它已從 MIT 改為 BSD-3 + branding 條款，限制的只有拿掉品牌，改 code 不限。
- ❌ `python3 venv ft` → ✅ `python3 -m venv ft` → 無 `-m` 是找檔案路徑，`-m` 是找模組名稱；同一個 runtime。
- ❌ 「沒 activate 就改用 `python3 -m mlx_lm`」→ ✅ 兩種寫法都會失敗（global 沒裝）→ 入口方式和環境選擇是兩個獨立的軸。
- ❌ `chmod -x` → ✅ `chmod +x` → `-` 是移除。
- ❌ before 做完直接跑 after → ✅ after 需要 `adapters/`，那是訓練的產物。
- ❌ valid 1 筆配 batch-size 2 → ✅ 每個資料集至少 batch_size 筆；valid 要夠多才能當偵測器。

---

## 心智模型總結

```
模型 = 純函數 tokens→tokens，資訊來源只有 [權重, prompt]
harness 持有 [工具, 狀態, 權限, 迴圈, RAG 編排, 評測]；引擎持有 [排程, KV cache, logits 約束]
改認知：prompt(t0, 本次) | 權重(t1, 永久, 最不可控) | logit_bias(t3, 只管出口, 不改認知)
fine-tune = 訓練工具鏈 → 新權重/adapter → 引擎載入；SFT = 給配對讓它模仿
epoch = iters×batch/N 是投入量；loss 是成果；overfit = train↓ val↑ = 開始背題
資料形狀決定梯度方向，epochs 只放大；沒有反例的資料集 → 模型學到「不讀輸入吐模板」
評測三類：目標 / 泛化 / 對照（中性 prompt）；before 必跑；對照組只差一個變因
```
