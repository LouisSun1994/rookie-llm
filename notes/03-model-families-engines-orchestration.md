# 03 — 模型種類、引擎歸屬與應用層編排（從 TRELLIS.2 挖出來的一條線）

- 記錄日期：2026-10-16
- 學習方式：從「TRELLIS.2 是什麼」挖到「模型 vs 引擎」「引擎裡裝什麼」「自回歸 vs diffusion」「UNet vs DiT」「latent」「多引擎編排」
- 目前定位：🥚 概念（路線圖已走過一遍，尚未動手部署）
- 時效標記：模型名稱、授權、工具行為標「2026-10 時」；設計原理不標

---

## 0. 一句話版本

**需要 GPU 只代表它是神經網絡；是哪一種模型看「輸入→輸出 × 生成方式」；引擎的大部分程式碼是外層迴圈而非 transformer block，所以引擎跟著生成方式走；不同引擎之間只能在應用層用 HTTP 串，LLM 本身只會吐文字。**

---

## 1. 分層模型（骨架）

```
應用層     Open WebUI / 自寫 orchestrator / MCP     ← 對話 DB 在這、路由決定在這、多引擎串接在這
引擎       Ollama / vLLM（自回歸）                   ← 常駐 process、握權重、聽 port、排請求
           ComfyUI（diffusion）
           自帶 pipeline（研究 repo，如 TRELLIS.2）
模型       權重 + 架構定義（HF 上的 safetensors / GGUF）← 只有數字，沒有「怎麼算」的程式碼
訓練       PEFT / LLaMA-Factory（LLM）、kohya / ai-toolkit（diffusion LoRA）
GPU        矩陣乘法；LLM 頻寬瓶頸、diffusion 算力瓶頸
```

**兩條獨立的分類軸，不要疊在一起：**

| 軸 | 回答什麼 | Llama 3.1 vs TRELLIS.2 |
|---|---|---|
| 堆疊層次 | 它在系統的哪一層 | **同層**（都是模型層） |
| 模型種類 | 輸入→輸出 + 生成方式 | **不同種**（文字→文字 自回歸 vs 圖→3D diffusion） |

引擎選擇看模型種類，不看層次。

---

## 2. 模型 vs 引擎：權重沒有說明書，引擎裡裝什麼

**核心規則**：權重檔只有一堆有名字的矩陣。「哪個矩陣在什麼時候乘誰」的程式碼在引擎裡。

**引擎的四塊**（以 llama.cpp / Ollama 為例）

```
① 權重載入     讀 GGUF，按 tensor 名字（blk.0.attn_q.weight、blk.0.ffn_up.weight…）放進記憶體 ← 名字對照表
② forward graph 手寫的「這個架構第 N 層怎麼算」：attn_q 乘 x → RoPE → causal attention → 殘差 → RMSNorm → ffn…
③ 外層迴圈     tokenize → prefill → while not EOS: forward 1 token → 讀 KV cache → sample → append
④ I/O         tokenizer、chat template、streaming HTTP、OpenAI 相容 API
```

**transformer block 只是 ② 裡的一小段**；引擎大部分程式碼在 ①③④。「架構相同」指 ② 那一小段的算術相同，不代表引擎能跑。

**DiT 進 Ollama 四層全擋**

| 層 | 為什麼擋 |
|---|---|
| ① | tensor 名字是 `double_blocks.0.img_attn.qkv.weight`、`time_in.in_layer.weight`，對照表裡沒有 |
| ② | graph 是 causal + RoPE + 一次一 token；DiT 要雙向 + adaLN + 4096 patch 同時進 |
| ③ | 迴圈條件「吐到 EOS」，DiT 不吐 token |
| ④ | 沒有 VAE，輸出不可能變成圖 |

**轉成 GGUF 也一樣跑不了**——檔案格式 ≠ 架構 ≠ 精度。

**比喻**：PostgreSQL 和 Redis 共用同一套 C 標準庫（malloc、socket、hash table），你不會期待 Redis 能跑 SQL。共用的是積木函式，不同的是誰在什麼時候以什麼順序呼叫。SQLite 開不了 PostgreSQL 的檔 = Ollama 跑不了 DiT。

**引擎為什麼不做成通用的**：引擎的價值是針對一種「外層迴圈的瓶頸」榨到極限。vLLM 的 PagedAttention / continuous batching 全建立在「有 KV cache、每輪算量小、請求長度不定」上，DiT 沒有這些，優化無處施力。

**研究型 repo（TRELLIS.2）** 把「模型」和「跑模型的程式」綁在一起發布，看起來像 kit，其實是「專用引擎 + 權重」兩層。

---

## 3. 模型種類：輸入→輸出 × 生成方式

| 輸入 → 輸出 | 俗稱 | 例子 | 生成方式 | 引擎 |
|---|---|---|---|---|
| 文字 → 文字 | LLM | Llama、Qwen | 自回歸 | Ollama、vLLM、llama.cpp |
| 文字+圖 → 文字 | VLM | Qwen-VL、Llama 3.2 Vision | 自回歸 | Ollama、vLLM |
| 文字 → 向量 | embedding model | bge、nomic-embed | 一次 forward | Ollama、vLLM |
| 文字 → 圖 | 圖像生成 | FLUX、SD、Qwen-Image | diffusion | ComfyUI、diffusers |
| 圖 → 3D mesh | 3D 生成 | TRELLIS.2 | diffusion | 自帶 pipeline |
| 語音 → 文字 | ASR | Whisper | 自回歸 | whisper.cpp |

**引擎跟著「生成方式 + 架構」走**：自回歸文字一族共用 Ollama / vLLM；diffusion 一族共用 ComfyUI；研究型 3D 什麼都共用不了。本專案路線只用前三列，**一套引擎（Ollama → vLLM）走完**。

---

## 4. 自回歸 vs Diffusion：外層迴圈的差異

**自回歸 = Autoregressive（AR）**：auto（自己）+ regressive（回歸），用自己過去的值預測下一個。

```
p(x₁…xₙ) = p(x₁) · p(x₂|x₁) · p(x₃|x₁,x₂) · … · p(xₙ|x₁…xₙ₋₁)
```

**兩種迴圈並排**

```python
# ===== Ollama / vLLM：自回歸迴圈 =====
ids = tokenizer(prompt)                        # CPU
kv = prefill(ids)                              # GPU，一次算完 prompt，填 KV cache
while True:
    logits = forward(ids[-1], kv)              # GPU，只算 1 個新 token，讀整份權重 + KV
    kv.append(k_new, v_new)                    # KV cache 長 1
    tok = sample(logits, temperature)          # CPU，從分布抽一個
    ids.append(tok); stream(decode(tok))       # 立刻吐
    if tok == EOS: break                       # 步數由模型決定

# ===== ComfyUI：去噪迴圈 =====
cond = text_encoder(prompt)                    # GPU，一次
x = randn(1, 16, 64, 64) * sigma[0]            # CPU，隨機雜訊 latent，形狀固定
for t in range(N):                             # 步數由設定決定（4 / 20 / 30）
    eps = dit(x, t, cond)                      # GPU，整個 latent 全進，雙向 attention，無 cache
    eps_u = dit(x, t, cond=None)               # CFG 開時再算一次
    eps = eps_u + cfg * (eps - eps_u)
    x = scheduler.step(x, eps, sigma[t])       # CPU，x 被整個覆寫
image = vae.decode(x)                          # GPU，一次
```

| | 自回歸 | Diffusion |
|---|---|---|
| 迴圈變數 | 序列（越長越長） | latent（形狀固定，內容整個覆寫） |
| 每輪 forward 輸入 | 1 個新 token + 舊 KV | 整個 latent，全部重算 |
| 跨輪狀態 | KV cache（前面 token 不變，hidden state 可留） | 無（latent 整個變，hidden state 全作廢） |
| 終止 | 模型吐 EOS | 設定的 N |
| 隨機性 | 每輪 sample | 只在 t=0 初始雜訊 |
| attention mask | causal | 無（雙向） |
| 條件進入 | 就是序列前段 | 每輪重新注入（adaLN / joint attention） |
| 可串流 | 可（前半句已定） | 不可（每處都還會變） |
| 每輪 GPU 工作量 | 小 → **頻寬瓶頸** | 大 → **算力瓶頸** |
| 優化手段 | PagedAttention、continuous batching | torch.compile、flash-attn、fp8、蒸餾減步數、多卡切 patch |
| 多請求合併 | 能：多人的「下一 token」疊成 batch 共讀一次權重 | 單張已塞滿 GPU，疊了只是等更久 → FIFO |

**比喻**：自回歸 = 寫作文，一字一字往下寫，錯了只能圓回來；diffusion = 畫油畫，先打底再逐層細化，每層可蓋掉上一層。**不是「收斂 vs 發散」**，兩者都收斂到一個樣本；差在生成的順序與範圍。

ComfyUI 的「引擎」本質 = PyTorch + DAG executor + 一條佇列。不需要也沒有 vLLM 那種調度器。

---

## 5. 積木：CNN / UNet vs Transformer / DiT（架構軸）

**2×2 表**

| | 自回歸 | Diffusion |
|---|---|---|
| **Transformer** | LLM / VLM：字 token、causal、有 KV cache | **DiT**：patch token、雙向、無 KV cache、adaLN 進條件 |
| **CNN / UNet** | —（PixelCNN，古早） | SD 1.5 / SDXL：濾鏡滑動、U 形、skip 保細節 |

- LLM 和 FLUX：同行不同列 → 共用積木，不共用引擎
- SD 1.5 和 FLUX：同列不同行 → **ComfyUI 都能跑**（引擎認的是列）
- 角落：diffusion LLM（2026 時：LLaDA、Mercury）——文字 + diffusion。證明「LLM」綁的是資料+任務，不是生成方式

### CNN（Convolutional Neural Network）
- 核心規則：小濾鏡（3×3 權重）在圖上滑動，每個位置同一組乘加
- 設計理由：權重共用（參數量與圖大小無關）、局部性先驗（相鄰像素相關）
- 代價：視野局部，看全圖得堆很深

### UNet
- CNN 的一種形狀：左半壓縮（抓語意）→ 瓶頸 → 右半放大（還原細節），**skip connection** 把同解析度特徵直接接過去
- 用在 diffusion 的理由：去噪任務「輸入帶雜訊 latent，輸出同形狀雜訊預測」，**輸入輸出同形狀**正是 UNet 的設計目的

### DiT（Diffusion Transformer，2023）
- 做法：把 UNet 整個換成 transformer，其他不動
- 圖怎麼變 token：latent 128×128×4 → 切 2×2 patch → 4096 個 → 線性投影成向量 → 加位置編碼 → N 層 → 投影回 → 拼回 128×128×4
- 條件怎麼進：**adaLN**（timestep + 文字 embedding → 縮放/平移係數調 LayerNorm）或 **MMDiT**（SD3 / FLUX：文字 token 和圖 token 拼在一起做 attention）

| | LLM block | DiT block |
|---|---|---|
| attention | causal | 雙向 |
| 條件 | 就是 prompt token | adaLN 或 MMDiT 拼接 |
| 輸出 | 下一 token 分布 | 每個 patch 的雜訊向量 |
| KV cache | 有 | 無 |

**為什麼 DiT 取代 UNet**：全局視野第一層就看全圖；scaling 不飽和（UNet 加大到某程度就平）；不要求格子資料（稀疏體素、影片、任意長寬比）；MMDiT 讓文字和圖在同一個 attention 裡對應，文字渲染大幅改善。

---

## 6. Latent：VAE 壓縮後的圖（名詞，不是動作）

**核心規則**：latent 是**被餵進 DiT 的那個 tensor**，不是「每步重算」這個動作。

```
像素空間（人看得懂）         latent 空間（模型工作的地方）
1024 × 1024 × 3 (RGB)   ──VAE encoder──▶   128 × 128 × 4
= 3,145,728 個數字         ◀──VAE decoder──   = 65,536 個數字（SD，壓 48 倍）
```

直接看 latent 是一片彩色雜訊；每個位置對應原圖一塊 8×8，4 個 channel 編碼那塊的「內容」。VAE 是**學出來的壓縮器**（JPEG 是人寫死的 DCT），學出來的空間有一個性質：**相近的點解碼出相似的圖**，去噪網絡才能在裡面「慢慢走向合理的圖」。

**誰產生、誰持有、誰觸發**

| 時機 | latent 從哪來 | 誰用它 |
|---|---|---|
| 訓練 | 真圖 → VAE encoder → latent → 加雜訊 | DiT 學「從帶雜訊 latent 預測雜訊」 |
| 推理 t=0 | 純隨機數，形狀固定（**沒經過 encoder**） | sampler |
| 推理每步 | 上一步 latent 減一部分雜訊 | DiT 讀、scheduler 寫回 |
| 推理結束 | 乾淨 latent → VAE decoder → 像素 | 使用者 |

text-to-image **從頭到尾 VAE encoder 沒出場**；encoder 只在訓練和 img2img（把使用者的圖壓進去當起點）用。

**數字**

```
SD 1.5 / SDXL     f8  c4    1024² → 128×128×4   = 65,536    壓 48×
FLUX / SD3        f8  c16   1024² → 128×128×16  = 262,144   壓 12×（channel 多，細節保得好）
TRELLIS.2 shape   f16 c32   1024³ 體素 → 64³×32，稀疏（空位沒 token）；config 名 shape_vae_next_dc_f16c32 = 這個意思
```

**為什麼要壓**：attention 是 patch 數的平方。像素空間 1024² 切 2×2 = 262,144 token → 7×10¹⁰ 對；latent 空間 128² 切 2×2 = 4,096 token → 1.7×10⁷ 對，差 4000 倍。不壓就沒有 latent diffusion，2022 年前只能做 64×64 小圖。

**三個長得像、其實不同的東西**

| 詞 | 是什麼 | 在哪 | LLM 對應 |
|---|---|---|---|
| **latent** | VAE 壓縮後的圖，128×128×4 | DiT 的輸入和輸出，**模型外**流動 | 最接近 token id 序列（模型工作的資料表示） |
| **embedding** | 一個 token / patch 投影成的向量 | 進 transformer **第一層** | token embedding |
| **hidden state** | 每層算出來的中間向量 | transformer **內部**，層與層之間 | KV cache 的來源 |

每一步精確流程：latent（資料）→ patchify 成 embedding（進模型）→ N 層 hidden state（模型內）→ 投影回雜訊預測（出模型）→ scheduler 更新 latent（資料）。

**DiT 無 KV cache 的根本原因**：hidden state 從 latent 算出來，latent 每步整個變，hidden state 跟著作廢。LLM 能留是因為前面 token 不變。

---

## 7. Diffusion model 三件組與三個設計

**核心規則**
- 訓練：真圖分級加高斯雜訊（純數學，無學習）→ 網絡學預測「加了什麼雜訊」（2026 主流 **flow matching**：預測往乾淨圖的方向向量；FLUX、SD3、TRELLIS.2 都是）
- 生成：純雜訊 → 問網絡 N 次「雜訊在哪」→ 每次減一點 → 圖

**三件組（= ComfyUI 三個 loader 節點 = 權重常是三個檔）**

```
text encoder（CLIP / T5）  prompt → embedding            ← 條件
去噪網絡（UNet 或 DiT）    帶雜訊 latent → 雜訊預測       ← 「diffusion model」通常指這塊
VAE decoder                乾淨 latent → 像素             ← 獨立訓練，可共用
```

| 設計 | 解什麼 | 細節 |
|---|---|---|
| Latent diffusion | 像素空間太貴 | 見第 6 章 |
| Conditioning + CFG | 文字怎麼進 | cross-attention / adaLN / 拼接；**CFG** 每步算兩次（帶文字、不帶）取差放大 → 每步 GPU 2× |
| 蒸餾（distillation） | 50 步太慢 | 學生學「老師 50 步的結果一步跳到」；schnell / Turbo 4 步；**代價：通常難再訓 LoRA**（2026-10 時 schnell LoRA 生態明顯弱於 dev） |

**算力數字**

```
FLUX dev 12B，20 步，CFG 開：
  每步 = 12B 參數 × 4096 latent token × 2（CFG）
  總計 ≈ 40 次完整 forward
  LLM 吐 40 token 也是 40 次 forward，但每次只 1 token
  → 同樣 40 次，diffusion 每次算力 ≈ LLM 的 4000 倍
DiT attention：4096² ≈ 1600 萬對；LLM decode 一步只有 1 × context 對
```

---

## 8. VLM（Vision-Language Model）

**一句話**：一個普通 LLM，前面多掛一個「把圖片翻成 token 的翻譯機」。

```
模型 ┌ vision encoder（ViT，0.4–1B）：圖 → 一堆向量
     ├ projector（幾層 MLP）：對齊到 LLM 的 token 空間
     └ LLM backbone：跟純文字版一模一樣
```

- 圖片**沒有變成一句話**，是 embedding 直接塞進序列。細節（顏色、位置、字跡）才保得住
- prefill 之後 VLM = LLM：continuous batching、KV cache、slot 全部沿用，只是 prompt 多了圖的 token
- 兩種接法：(a) 圖 token 拼進序列（LLaVA、Qwen-VL）；(b) cross-attention（Llama 3.2 Vision）。**2026 年初 vLLM V1 只支援 (a)**，選模型前查 supported models 頁

```
1024² 圖，Qwen2-VL 類 ≈ 1,300 token；LLaVA-1.5 固定 576
KV cache ≈ 0.5 MB/token（fp16，7B GQA）→ 一張圖 ≈ 650 MB
40 人各一張圖 = 26 GB，只算 KV cache；vision encoder 權重 fp16 另加 1–2 GB
vLLM `--language-model-only`：純文字時不載多模態模組，VRAM 讓給 KV cache
```

---

## 9. 引擎對照：Ollama / vLLM vs ComfyUI vs 自帶 pipeline

**共同形狀（listen 的意義）**：一個常駐 process 把昂貴狀態（權重）握在手上不放，站在門口等人送單。

| 事 | 誰做 | 不這樣做會怎樣 |
|---|---|---|
| 權重載一次進 VRAM | CPU→GPU | 每次重讀 12GB，M1 多等 30–60 秒 |
| 聽 port | CPU | 沒有入口 |
| 排隊，一次一個送 GPU | CPU scheduler | 兩個同時進 GPU → OOM |
| 回報進度 | CPU | UI 不知道跑到哪 |

| | Ollama / vLLM | ComfyUI | 自帶 pipeline（TRELLIS.2 `example.py`） |
|---|---|---|---|
| 跑哪一族 | decoder-only transformer | diffusion（UNet + DiT 都行） | 只有這一個模型 |
| 一個請求 | 對話 → prefill + N decode | 一張 workflow 圖（DAG）→ 按節點執行 | 一張圖 → mesh |
| 執行單位 | token | 節點 | 整條焊死的水管 |
| 併發 | continuous batching | **FIFO 一次一張，線性排隊** | 跑完就死，無常駐，要自己包 |
| API | OpenAI 相容 `/v1/chat/completions` | 自家 `/prompt`，body 是 workflow JSON 方言 | 無 |
| 內建 UI | 無（靠 Open WebUI） | 有，節點編輯器 | Gradio demo |
| 模型檔 | 一個 GGUF（含 tokenizer） | **三個檔分開**：text encoder / 主模型 / VAE | HF repo |
| 併發上限參數 | `--max-num-seqs` | 沒有這種東西 | — |

**ComfyUI 選它的理由**：產圖沒有 vLLM 等級的替代品；「API + 佇列 + UI」三合一。替代：diffusers（函式庫，server / 佇列 / 常駐全自己寫）、Draw Things（Mac App，無 API）。**代價**：workflow JSON 是方言，接 Open WebUI 要先畫圖、匯出、指定哪個節點是 prompt。

**「pipeline」一詞兩義，差兩層**

| 意思 | 是什麼 | 層 | 例子 |
|---|---|---|---|
| ① 單一模型的執行流程 | 前處理 → forward → 後處理，一個 Python 物件，**段數焊死** | 引擎（簡化版） | `Trellis2ImageTo3DPipeline`、HF `pipeline()`、diffusers |
| ② 多個模型串接 | A 輸出丟給 B，跨 process / 服務 | 應用層 | Ollama → ComfyUI → TRELLIS |

① 用 HF `pipeline()` 直接跑 LLM 可行，但是「沒有引擎」的跑法：無 batching、無 PagedAttention，一次一請求。

---

## 10. 應用層編排：LLM 怎麼「操作」別的模型

**前提釘死：LLM 只會吐文字。** 不能執行程式、不能呼叫 API、不能碰 GPU 上另一個 model。

```
LLM      = 工程師：看需求，吐一段 JSON 工單 {"tool":"generate_3d","args":{"image":"x.png"}}
應用層   = 專案經理：parse 工單、真的打 HTTP、把結果塞回給 LLM
TRELLIS  = 工廠：獨立 process、自己的 GPU，收圖回 glb
```

兩個 model 從未直接接觸，中間是 orchestrator 用 HTTP 當電線（function calling / tool use；標準化版 = MCP）。② 是 microservice 編排，① 是每個 service 內部的 handler。

**Open WebUI 是獨立 web app，下面掛任意多個 backend**

```
Open WebUI（SQLite / Postgres）  ← 整串對話存這裡，唯一真相；路由決定在這
      │                  │
   Ollama / vLLM       ComfyUI
   每輪收整串歷史，      每輪只收一段 prompt 字串，
   算完就忘            不知道有「對話」這回事
```

**交叉使用兩種觸發（2026-10 時）**
- 路 A：使用者按鈕 / Image 開關 → Open WebUI 自己決定打 ComfyUI，LLM 不參與路由
- 路 B：註冊 tool → LLM 吐 tool call → Open WebUI 執行。Open WebUI 沒內建的（TRELLIS）**只能走路 B**，orchestrator 自己寫

```
Open WebUI 設定骨架
Connections:  Ollama API   http://ollama:11434      ← 聊天、prompt 翻譯
              OpenAI API   http://vllm:8000/v1      ← 之後換 vLLM，同一欄位
Images:       Engine ComfyUI、URL http://comfyui:8188、匯入 workflow JSON、Prompt node ID
同一個 compose network，service name 互找；Open WebUI 唯一對外開 port
```

**換 TRELLIS 要修三處**

| 處 | ComfyUI | TRELLIS |
|---|---|---|
| 輸入 | 文字 | **圖片** → 要先有圖（使用者上傳，或 Ollama → ComfyUI 先產） |
| 觸發 | 按鈕（路 A） | **tool call（路 B）** |
| 輸出 | png，原生顯示 | **glb**，Open WebUI 不渲染 3D，只能下載或自架 viewer |

**產品級「多模態」兩條路線**

```
路線 1 編排：orchestrator LLM → 專門模型 A/B/C 接力（tool call）
  優：各自獨立升級、各自最強、好部署、VRAM 可分時 ← 自建只能走這條
  缺：模態之間資訊會漏（LLM 要先「描述」才能交給下一個）
路線 2 原生統一（omni）：一組權重，文字/圖/音都是 token，進出同一個 transformer
  優：無翻譯損失，跨模態編輯才做得到
  缺：巨大、訓練成本極高、自建幾乎不可能
```

「多模態很厲害」幾乎都只講 **input 端**。output 端要分：獨立 diffusion 接力（路線 1，Imagen / Veo）vs 同一組權重吐圖 token（路線 2，GPT-4o 原生產圖、Gemini Flash Image、Janus）。分辨法：能不能做「把圖裡的貓換成狗其他不動」的跨模態編輯。

---

## 11. TRELLIS.2 事實表（2026-10 時）

| 項目 | 值 |
|---|---|
| 是什麼 | Microsoft，image-to-3D 生成模型，4B，MIT 授權（nvdiffrast / nvdiffrec 各自授權，商用前看） |
| 架構 | 三階段 flow（sparse structure → shape → texture），DiT + Sparse 3D VAE（f16c32）+ 稀疏卷積（FlexGEMM） |
| 2×2 格 | Transformer × Diffusion（以 DiT 為準，卷積是輔助） |
| 核心表示 | O-Voxel，處理開放面 / 非流形 / 內部封閉結構，不需有損轉換 |
| 輸出 | 帶 PBR（Base Color / Roughness / Metallic / Opacity）的 mesh → `.glb` |
| 硬體 | Linux only、≥24GB VRAM、CUDA 12.4、flash-attn；**M1 跑不了**（自訂 CUDA / Triton kernel 無 Metal 後端） |
| 速度（H100） | 512³ ~3s、1024³ ~17s、1536³ ~60s |
| 安裝 | `setup.sh` 從原始碼編譯多個套件 → Docker 化要帶完整 CUDA toolkit |
| 引擎 | 無通用引擎，自帶 `Trellis2ImageTo3DPipeline`；多人要自己包 FastAPI + 佇列 + lock |
| 訓練 | 附 `train.py` + configs，可從頭訓或 fine-tune |
| 與本專案 | 與聊天服務路線無關；「第二類工作負載」：單 GPU、單請求佔滿、無併發調度 |

```python
pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B").cuda()  # 啟動載一次，常駐
lock = asyncio.Lock()                      # 一次只讓一個請求碰 GPU
@app.post("/generate")
async def generate(image_path: str):
    async with lock:                       # 第二個人在這排隊，不會 OOM
        mesh = pipe.run(Image.open(image_path))[0]
    return {"path": out_path}
```

---

## 12. 產圖地端部署（text-to-image）

| 層 | LLM 路線 | 產圖路線 |
|---|---|---|
| 應用層 | Open WebUI | Open WebUI（內建接 ComfyUI） |
| 引擎 | Ollama → vLLM | ComfyUI |
| 模型 | GGUF | safetensors ×3 |
| 訓練 | PEFT + LLaMA-Factory | kohya / ai-toolkit 訓 LoRA |
| GPU | 同一張卡 | 同一張卡，無 KV cache、無 continuous batching |

**授權先篩（2026-10 時，上線前再查 model card）**

| 模型 | 授權 | VRAM | 特點 |
|---|---|---|---|
| FLUX.1 schnell | Apache-2.0 | ~24GB fp16 / ~12GB fp8 | 1–4 步起手式；蒸餾版難訓 LoRA |
| FLUX.2 dev | 非商用 | 更大 | 品質最好，商用另談 |
| Qwen-Image 1.0 | Apache-2.0 | ~24GB+ | 圖內文字（含中文）最強 |
| Qwen-Image 2.1 | research-only | — | **同系列不同版授權不同** |
| SD 3.5 | Stability Community（營收上限條款） | ~16GB | 生態最大、LoRA 最多 |

```bash
# M1 驗證（16GB 抓 fp8 量化版；慢但能走流程）
git clone https://github.com/comfyanonymous/ComfyUI && cd ComfyUI
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision torchaudio && pip install -r requirements.txt
# flux1-schnell-fp8.safetensors → models/checkpoints/
python main.py --force-fp16        # http://127.0.0.1:8188；1024² 約 60–120s（4090 約 2–5s）
```

```yaml
# Linux compose
services:
  comfyui:
    image: ghcr.io/ai-dock/comfyui:latest
    runtime: nvidia
    environment: [NVIDIA_VISIBLE_DEVICES=0]
    volumes: ["./models:/workspace/ComfyUI/models", "./output:/workspace/ComfyUI/output"]
    ports: ["127.0.0.1:8188:8188"]           # 只綁 localhost，前面 nginx 做帳號
```

```
VRAM：一張 24GB 卡能不能同時跑 LLM + 產圖
FLUX.1 schnell fp8：權重 12 + T5 fp8 ~5 + VAE/latent ~1–2 ≈ 18–19 GB 峰值
LLM（Qwen 7B Q4 + 40 人 KV）≈ 5 + 8–10 GB
合計 > 24 → 不能同時常駐
選項：① 兩張卡 ② 一張卡，產圖用完卸載（每次多 5–10s）③ 小模型（Sana、FLUX.2 klein，<8GB）
```

**多人**：ComfyUI 一條 FIFO。4090 一張 schnell 約 3s → 40 人同時按，最後一個等 2 分鐘。加吞吐 = 多卡 + 多 instance + round-robin，後端問題不是 ML 問題。

---

## 思維鏈：完整時間軸（Ollama → ComfyUI → TRELLIS 三段接力）

使用者先聊出火焰法師角色設定，說「畫出來」，再說「做成 3D」。

```
t    | CPU/GPU | 元件                          | 處理什麼                      | 發生什麼
-----|---------|-------------------------------|-------------------------------|--------------------------------------------
t0   | CPU     | 瀏覽器 → Open WebUI            | user 訊息                      | 「幫我想一個火焰法師設定」
t1   | CPU     | Open WebUI                    | 對話歷史（空）                 | 撈 DB，組 messages，POST Ollama /api/chat
t2   | GPU     | Ollama / Qwen                 | prompt token                   | prefill + decode，串流吐角色設定
t3   | CPU     | Open WebUI                    | user + assistant 訊息          | 寫進 DB（對話的唯一真相）
t4   | CPU     | 瀏覽器 → Open WebUI            | user 訊息 + Image 開關          | 「把他畫出來」
t5   | CPU     | Open WebUI                    | 歷史（2 則）                   | POST Ollama，system prompt 要它把前文翻成英文產圖 prompt
t6   | GPU     | Ollama / Qwen                 | 整串歷史                       | 吐 "a fire mage, red robes, glowing staff…"（LLM 當 prompt 工程師）
t7   | CPU     | Open WebUI                    | prompt 字串                    | 塞進 workflow JSON 的 prompt 節點，POST ComfyUI /prompt
t8   | CPU     | ComfyUI queue                 | prompt_id                      | 排 FIFO
t9   | GPU     | ComfyUI / text encoder        | 文字                           | → embedding
t10  | CPU     | ComfyUI sampler               | 初始 latent                    | 隨機數 → 64×64×16（沒經過 VAE encoder）
t11  | GPU     | ComfyUI / DiT                 | 整個 latent                    | 去噪 4 步（schnell），每步完整 forward，無 KV cache，latent 整個覆寫
t12  | GPU     | ComfyUI / VAE decoder         | 乾淨 latent                    | → 1024×1024×3 像素
t13  | CPU     | ComfyUI → Open WebUI          | fire_mage.png                  | 存檔，WebSocket 推完成；Open WebUI 拉回，當 assistant 訊息寫 DB
t14  | CPU     | 瀏覽器 → Open WebUI            | user 訊息                      | 「把這個做成 3D」
t15  | CPU     | Open WebUI                    | 歷史 + 工具清單（generate_3d） | POST Ollama
t16  | GPU     | Ollama / Qwen                 | 整串歷史 + 工具定義            | 吐 tool call JSON {"name":"generate_3d","args":{"image":"fire_mage.png"}}
t17  | CPU     | Open WebUI tool runner        | 工單                           | 執行你寫的 Python：POST http://trellis:8000/generate
t18  | CPU     | TRELLIS service queue         | 請求                           | 排隊，拿 lock
t19  | GPU     | TRELLIS pipeline / DiT+VAE    | 圖 → latent → O-Voxel → mesh   | 三階段 flow 去噪 + decode，1024³ 約 17s（H100）
t20  | CPU     | TRELLIS service               | mesh                           | 後處理（簡化、UV、烘焙），寫 fire_mage.glb，回 {"path": …}
t21  | CPU     | Open WebUI                    | tool result                    | 當文字塞回歷史，再 POST Ollama
t22  | GPU     | Ollama / Qwen                 | 歷史 + 結果                    | 吐人話：「做好了：[下載 fire_mage.glb]」
t23  | CPU     | Open WebUI                    | assistant 訊息                 | 寫 DB，顯示（glb 只給下載，不渲染）
```

**關鍵觀察**
- 三個引擎在 GPU 上是**輪流**（t2 → t11 → t19），不是同時；一張卡要分時或分卡
- Ollama 跑了四次（t2 / t6 / t16 / t22），每次收整串歷史，算完就忘
- ComfyUI 和 TRELLIS 都不知道有「對話」：各只收一段字串 / 一張圖
- 「頭髮改白色」能做是因為 t6 的 prompt 存進歷史；ComfyUI 收到的是全新字串
- 「腿再粗一點」走到 TRELLIS 會卡：它沒有「上一個 mesh」的概念，輸入是圖不是文字
- 路由決定全在 Open WebUI（t4 按鈕、t16 tool call 解析），引擎不決定任何事

---

## 陷阱清單

**這條線上實際撞到的**

- ❌ 「TRELLIS.2 只是一個 kit」→ ✅ repo = 專用引擎（pipeline 程式碼），權重另在 HF → 研究型模型沒有通用引擎，作者只能自附 pipeline
- ❌ 「有 model + 有 Ollama 引擎 → 應該能跑」→ ✅ 引擎內建一組寫死的架構實作，不是通用 GPU 執行器 → 權重只有數字，「怎麼乘」在引擎裡
- ❌ 「對 process 來說都只是看到一個資源檔案」→ ✅ 引擎看到的是「一堆有名字的矩陣」，必須事先有名字對照表 + forward graph + 外層迴圈 + I/O 四層 → DiT 四層全對不上
- ❌ 「轉成 GGUF 就能進 Ollama」→ ✅ 檔案格式 ≠ 架構 → 缺的是 DiT graph / VAE / 去噪迴圈，不是讀檔
- ❌ 「都是 transformer 架構為什麼引擎差這麼多」→ ✅ transformer block 只是引擎程式碼的一小段；引擎跟著「外層迴圈的瓶頸」設計 → LLM 優化 batching + 分頁，DiT 優化 compile / fp8 / 減步數，互不適用
- ❌ 「TRELLIS.2 是純圖片 model」→ ✅ image-to-3D，分類要說輸入→輸出 → 方向決定引擎
- ❌ 「它也用 GPU，所以類似 LLM」→ ✅ 用 GPU 只證明是神經網絡 → GPU 是共同點不是區分點
- ❌ 「diffusion 不是 transformer、不是神經網絡」→ ✅ 是神經網絡；「diffusion」講生成法不講架構，裡面可以是 UNet 或 DiT → 兩條軸：積木 × 生成法
- ❌ 「diffusion 跟 LLM 是包含 / 屬於 / 等價」→ ✅ 都不是，不同軸 → LLM = transformer × 自回歸 × 文字；FLUX = transformer × diffusion × 圖
- ❌ 「自回歸 vs diffusion = 收斂 vs 發散」→ ✅ 差在順序與範圍：一次定一個位置不回頭 vs 全部位置一起反覆修 → 兩者都收斂到一個樣本
- ❌ 「latent 是每次把全文放進去 rebuild 的動作」→ ✅ latent 是名詞：VAE 壓縮後的圖，被餵進 DiT 的那個 tensor → 「每步整個放進去」是對的，但那是對 latent 做的事，不是 latent 本身
- ❌ 「latent / embedding / hidden state 是同一種東西」→ ✅ latent 在模型外流動、embedding 是進第一層的投影、hidden state 是層與層之間 → latent 每步整個變，所以 hidden state 全作廢，DiT 無 KV cache
- ❌ 「pipeline 就是水管，TRELLIS 用 pipeline 所以 LLM 也能這樣連起來」→ ✅ 兩義差兩層：① 單一模型焊死的執行流程（引擎）② 多模型 HTTP 串接（應用層）→ 連線在應用層用 API（此條是術語未先定義造成）
- ❌ 「用 LLM 去操作 3D gen」想成 LLM 直接驅動 → ✅ LLM 只吐文字（tool call JSON）；應用層 parse、打 HTTP、塞回結果 → 「操作」一定要有翻譯官
- ❌ 「多模態輸入 = 把圖轉成 prompt 文字」→ ✅ 轉成 embedding 直接塞進序列 → 轉文字會丟細節
- ❌ 「強調會作圖的 LLM = 專門產圖的程序」→ ✅ 兩種都可能：獨立 diffusion 接力（路線 1）或同一組權重吐圖 token（路線 2） → 分辨法：能否跨模態編輯
- ❌ 「Open WebUI 是 Ollama 的前端」→ ✅ 獨立 web app + 對話 DB，下面掛任意多 backend → 引擎無狀態；路由決定在應用層

**延伸會錯的預測**

- ❌ 「vLLM 的 continuous batching 經驗套到 ComfyUI」→ ✅ 一次一張線性排隊，40 人 = 40 × 單張時間 → diffusion 每步 GPU 已滿載
- ❌ 「產圖模型是一個檔」→ ✅ 三個檔：text encoder / 主模型 / VAE → VAE 獨立訓練可共用
- ❌ 「text-to-image 會用到 VAE encoder」→ ✅ 推理只用 decoder，起點是隨機數 → encoder 只在訓練和 img2img
- ❌ 「VLM 看圖的請求跟純文字一樣重」→ ✅ 一張 1024² 圖 ≈ 1,300 token ≈ 650 MB KV → 40 人帶圖差一個數量級
- ❌ 「同系列模型授權一樣」→ ✅ Qwen-Image 1.0 Apache / 2.1 research-only → 看版本的 model card
- ❌ 「蒸餾版（schnell）快又好直接用」→ ✅ 快 10 倍但通常難再訓 LoRA → 🐊 階段選 dev 版
- ❌ 「一張 24GB 卡同時常駐 LLM + FLUX」→ ✅ 18–19 + 13–15 > 24 → 分卡、分時卸載、或換小模型

---

## 心智模型總結

```
GPU 只說明「是神經網絡」；模型種類 = 輸入→輸出 × 生成方式；引擎跟生成方式走
權重 = 一堆有名字的矩陣；引擎 = 名字對照表 + forward graph + 外層迴圈 + I/O；block 只是一小段
檔案格式 ≠ 架構 ≠ 精度
自回歸：序列成長、1 token/輪、KV cache、EOS 終止、頻寬瓶頸 → batching + 分頁
diffusion：latent 覆寫、全算/輪 ×CFG、固定 N、無 cache、算力瓶頸 → compile / fp8 / 減步數 / FIFO
積木軸：CNN/UNet（局部、同形狀進出）vs Transformer/DiT（全局、patch 當 token、adaLN）
latent = VAE 壓縮的圖（名詞），模型外流動；embedding 進第一層；hidden state 在層間
三件組：text encoder → 去噪網絡（UNet|DiT）→ VAE decoder；推理不用 encoder
VLM = vision encoder + projector + LLM；圖變 embedding 不變文字；prefill 後就是 LLM
引擎形狀相同：常駐、握權重、聽 port、排隊；Ollama/vLLM | ComfyUI | 自帶 pipeline 各管一族
pipeline ①焊死的單模型流程（引擎）/ ②HTTP 串多模型（應用層）
LLM 只吐文字；「操作」= tool call JSON → 應用層打 HTTP → 結果塞回；兩 model 永不直接接觸
Open WebUI = orchestrator + 對話 DB；引擎無狀態；路由在應用層；同卡多引擎輪流不同時
自建 = 路線 1 編排；omni 原生統一是路線 2，自建不可行
產圖部署：授權先篩 → ComfyUI → Open WebUI 內建接 → VRAM 跟 LLM 擠不進 24GB → 多人靠多 instance
```
