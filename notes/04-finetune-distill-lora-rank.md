# 04 · fine-tune / 蒸餾 / LoRA / 秩 / safetensors 與 GGUF

- 記錄日期：2026-10-16
- 學習方式：深挖一條線（從「fine-tune 跟蒸餾差在哪」挖到矩陣的秩、LoRA 的 VRAM 與推論成本，再接到權重檔格式：safetensors / GGUF 是什麼、誰吃誰）
- 目前定位：🥚 概念 → 🐊 fine-tune 的概念前置。尚未動手訓練。

---

## 0. 一句話版本

fine-tune = 以一份 checkpoint 為起點繼續訓練；蒸餾 = 訓練訊號來自另一個模型的 fine-tune（或 pretraining）；LoRA = 把改變量 ΔW 限制成「≤ r 條 if-then 規則」的 fine-tune——改的是方向不是位置，省的是梯度與 optimizer state，代價是表達力天花板且 merge 不掉。權重檔分兩種盒子：safetensors 是 HF 生態系的「原始碼」（訓練與 vLLM 都從這出發），GGUF 是為 llama.cpp 家族編譯好的單檔產物（Ollama 吃這個，不能拿回去訓練）；兩者差的 GB 數來自精度不是格式。

---

## 1. 分層模型（骨架）

```
應用層       SYSTEM prompt / RAG / skill / Modelfile 的 SYSTEM、PARAMETER   → 改 context，W 不動
推理引擎層   Ollama / llama.cpp（吃 GGUF）、vLLM / MLX（吃 safetensors）     → 執行 W·x (+ A·(B·x))
模型層       checkpoint：safetensors（+config.json+tokenizer）/ GGUF（單檔） → W 與 ΔW 住在這裡
             adapter_model.safetensors                                     → ΔW 的 A、B
訓練工具鏈   transformers / PEFT / TRL / LLaMA-Factory                       → 只吃 safetensors；fine-tune 這個「動作」只發生在這層
GPU          VRAM：權重 / 梯度 / optimizer state / activation                → 誰被凍結決定誰不用付後三項
```

Ollama 不會訓練。`ollama create` 只寫 manifest，權重 blob 共用（= `docker build` 加一層、base image 不複製）。

---

## 2. fine-tune 的定義與術語階梯

**核心規則**：fine-tune 的定義只有一條——初始化 = 載入一份已存在的 checkpoint，而不是隨機。其他（loss、資料來源、更新範圍、資料量）全部獨立。

| 名稱 | 起點 | 資料 | 更新 |
|---|---|---|---|
| Pretraining (from scratch) | 隨機 | 兆級 token 原始文本 | full |
| Continued pretraining (CPT) | checkpoint | 十億級 token 領域文本 | 通常 full |
| Full fine-tune / SFT | checkpoint | 千～百萬筆標註對 | full |
| PEFT (LoRA / QLoRA) | checkpoint | 同上 | 只有 adapter |

**關鍵細節**
- 「training new model」= 第一列，不是 fine-tune。
- 起點可以是 base（`Qwen2.5-7B`）或已 instruct-tuned（`Qwen2.5-7B-Instruct`），後者省掉教對話格式。
- 你 fine-tune 的是 `model-0000x.safetensors`，不是 Ollama 裡的 `qwen2.5:7b`（那是轉成 GGUF 的推理用副本）。

---

## 3. 蒸餾

**核心規則**：蒸餾 = 訓練訊號來自 teacher 模型的任何訓練。學生已 pretrained → 是 fine-tune 的一種（你會碰到的全是這種）；學生從頭訓 → 是 pretraining 的一種（Gemma 2 小尺寸、Minitron）。

```
黑箱蒸餾 (sequence-level)
  teacher 生成文字 → 當 SFT 資料 → 學生做普通 SFT
  loss：cross-entropy，hard label（對的 token 是 1）
  需要：teacher 的 API 即可；tokenizer 可不同
  例：DeepSeek-R1-Distill-Qwen/Llama（2025）= R1 生 80 萬筆 → 對 Qwen 做 SFT

白箱蒸餾 (logit-level)
  teacher 每個位置吐出整個機率分佈 → 學生用 KL divergence 逼近
  loss：KL(teacher ‖ student)，soft label
  需要：teacher 權重同機跑 forward（VRAM 多一顆模型的權重 + activation，不需梯度）；vocab/tokenizer 要對齊
```

**為什麼 soft label 有價值**：hard label 只說「這格是狗」；soft label 說「70% 狗、20% 狼、5% 狐狸」，多出的類別相似結構（dark knowledge）是小模型能逼近大模型的主因。

**跟目標的關係**：DSL fine-tune 幾乎一定會用黑箱蒸餾，只是名字叫「合成資料」——區網內大模型把 DSL 範例擴寫成幾千筆問答，再 QLoRA 訓小模型。

---

## 4. 三條獨立軸（最常被綁成一條）

| 軸 | 問的是 | 選項 |
|---|---|---|
| A. 訓練訊號從哪來 | 標籤誰給 | 人寫/收集 vs teacher 模型 |
| B. 更新哪些權重 | 存什麼 | full vs LoRA / QLoRA |
| C. 學生是誰 | 架構 | 同一顆 vs 另一顆（通常更小） |

- fine-tune vs 蒸餾 只差 A。增量 vs 完整 是 B。兩者無關。
- LoRA 能用來蒸餾**不是因為**訊號來自 teacher，而是 LoRA 只決定 backward 時梯度流到哪些參數，不管 loss 是 CE 還是 KL。

```
forward:  x → base(凍結) + A·B(可訓) → logits
loss:     [任意] CE / KL
backward: ∂loss/∂A, ∂loss/∂B   ← LoRA 只碰這裡
```

---

## 5. 改 context vs 改 W

```
output = f_W( context )

prompt / RAG / SYSTEM / skill  →  改 context（輸入），W 一個 byte 不動
fine-tune                      →  改 W（函數本身）
```

**Modelfile（Ollama 專有，不是通用術語）= Ollama 的 Dockerfile**

| Ollama | Docker | 層 |
|---|---|---|
| `FROM qwen2.5:7b` | `FROM ubuntu` | 模型層（指向 GGUF blob） |
| `SYSTEM "..."` | `ENV` / `CMD` | 應用層設定烘進包裡 |
| `PARAMETER temperature 0.2` | `ENV` | 推理引擎參數 |
| `TEMPLATE` | — | chat template |
| `ADAPTER ./lora/` | `COPY` 一層 | 模型層（LoRA 權重，推理時載入，不訓練） |
| `ollama create` | `docker build` | 新 manifest，blob 共用 |

- 改 W 改的是**習慣**（格式、用詞、黑話→概念映射）；要背一千份內部文件是 context（RAG）的工作。
- 為什麼 fine-tune 歪掉會變 idiot：loss 只有「把這批資料的 token 機率推高」，**沒有任何一項說「別忘記原本會的」**→ catastrophic forgetting。症狀：一般問答跳針、只剩資料裡那幾種句式、instruction following 失效。

---

## 6. 秩（rank）

**核心規則**：秩 = 矩陣裡獨立配方（原料）的數量。列數 = 成品數量，秩 ≠ 列數。

把權重矩陣 W（d_out × d_in）看成一組配方，不是資料表：

```
y = W · x
第 i 列 W[i,:] = 第 i 個輸出維度對 N 個輸入分量的加權配方
```

矩陣乘法三種讀法（全對，之後看 QKᵀ 直接用）：
- 每一格 = 左列 · 右欄 的內積
- 錨定結果的**列** → 右矩陣各列的加權
- 錨定結果的**欄** → 左矩陣各欄的加權

```
A (4×2)   B (2×4)
[1 0]     [1 2 3 4] ← b1
[0 1]     [1 0 1 0] ← b2
[1 1]
[2 1]

A·B 第 i 列 = A[i,1]·b1 + A[i,2]·b2
  = [1 2 3 4], [1 0 1 0], [2 2 4 4], [3 4 7 8]
4×4，4 列，但原料只有 b1、b2 → 秩 ≤ 2
```

**乘積的秩上限**

```
rank(X·Y) ≤ min( 結果列數, 結果欄數, 內側維度 k )
```

內側 k 一定是上限：右矩陣有 k 列（列原料 k 條）、左矩陣有 k 欄（欄原料 k 條），兩邊數出來都是 k（列秩 = 行秩）。「≤」變「=」要原料彼此獨立。

```
A(4×2)·B(2×4) → 4×4，k=2 → 秩 ≤ 2
B(2×4)·A(4×2) → 2×2，k=4 → 秩 ≤ min(2,2,4) = 2   （能乘，只是沒用）
LoRA: A(3584×16)·B(16×3584) → 3584×3584，k=16 → 秩 ≤ 16
```

**低秩的手感**：把 rank-r 拆成 r 個 rank-1 相加

```
ΔW = u₁·v₁ᵀ + … + u_r·v_rᵀ
uₖ·vₖᵀ 作用在 x 上 = uₖ · (vₖ·x)
  vₖ·x：x 跟方向 vₖ 像不像（一個數）
  uₖ：像的話往 uₖ 推輸出
→ rank-1 = 一條 if-then：「輸入長得像 v，就把輸出往 u 推」
→ rank-16 = 16 條規則疊加
```

Full-rank 的感覺：每一列都在說新的事，無法壓縮。預訓練的 W 是這樣。

---

## 7. LoRA：約束，不是優越

**核心規則**：ΔW = A·B，rank ≤ r，ΔW **每一格都有值**（u、v dense），改的是方向不是位置。「局部」是位置受限（某些格為 0），低秩是結構受限；局部是低秩的特例，低秩大得多。

**LoRA 論文的賭注**：預訓練 W 高秩，但「學會某任務」對 W 的**改變量**是低秩的。r 開到 d 就退化成該矩陣的 full fine-tune。

| | Full fine-tune | LoRA |
|---|---|---|
| ΔW 自由度 | 任意 | ≤ r 條 if-then |
| 任務的 ΔW 真低秩 | 浪費算力但結果好 | 幾乎不輸，成本省 ~100 倍 |
| 任務需大改（新語言/新領域） | 做得到 | underfit |
| 忘記原能力 | 容易 | 較難（限 r 維子空間），**不是不會** |
| 2026 時實證 | 程式/數學 CPT 類明顯較好 | 格式/風格/指令類幾乎打平 |

**兩個獨立旋鈕**
- `r`：矩陣內部的約束（原料數）
- `target_modules`：掛在哪些矩陣（q/k/v/o？加 MLP gate/up/down？）——這才是模組層級的「局部」；比對的是 safetensors 裡的 tensor 名字（`*.q_proj`）

**Adapter 大小（Qwen2.5-7B：28 層、d=3584、GQA 使 k/v 輸出 512、r=16、掛 q/k/v/o）**

```
每投影參數 = d_in×r + r×d_out
  q, o: 3584×16 + 16×3584 = 114,688
  k, v: 3584×16 + 16×512  =  65,536
每層: 2×114,688 + 2×65,536 = 360,448
28 層: ≈ 10.1M 參數
bf16: ≈ 20 MB   （base q4 GGUF ≈ 4.7 GB 的 0.4%）
```

Adapter 是兩個矩陣，存在 `adapter_model.safetensors`，不是「向量檔」。訓練時寫出矩陣初始化為 0 → ΔW 從 0 起步，秩隨訓練長到接近 r。

**PEFT 的命名跟本筆記的字母相反**（打開 adapter 檔會撞到）：

```
本筆記        PEFT 檔內名字                              形狀         角色
B（讀取）  →  ...q_proj.lora_A.weight                   [r, d_in]    先碰 x，萃取 r 個數
A（寫出）  →  ...q_proj.lora_B.weight                   [d_out, r]   把 r 個數混成輸出
計算順序：lora_B(lora_A(x))。字母反了，角色沒變：先出現的那個讀 x。
```

---

## 8. 推論時的 rank

**核心規則**：秩不是訓練超參數（調完就丟），是 ΔW 本身的永久性質。merge 後**成本差消失，表達力差不消失**。

| | 推論成本 | 行為 |
|---|---|---|
| 不 merge：`y = W·x + A·(B·x)` | 每 token 每矩陣多 `2·d·r` ≈ 115k 乘法，對 `d²` ≈ 12.8M 是 +0.9% | 受 rank 限制 |
| merge：`W' = W + A·B` | 與 full FT 模型完全相同，零額外成本 | 一樣受限——W' 形狀同 full FT，但它帶的 ΔW 永遠秩 ≤ r |

**Merge 實務（2026-10 時）**
- merge 要在 bf16 base 上做（PEFT `merge_and_unload` → 轉 GGUF）。
- 4-bit 權重不能直接加 bf16 的 ΔW（格子已量化掉）→ Ollama `ADAPTER` 走「不 merge」，llama.cpp 每層算完 `W·x` 再加 `A·(B·x)`。
- 不 merge 的好處：同一顆 base 掛多份 adapter（DSL / 客服語氣各 20 MB），vLLM 2026 時支援多 LoRA 服務，請求層級指定。Full FT 做同樣的事要存兩份 15 GB。
- 路線 1「先算出 A·B 存起來」= merge；路線 2「每次算 A·(B·x)」= 不 merge。

---

## 9. 訓練 VRAM 估算（Qwen2.5-7B，7.6B 參數，AdamW + 混合精度）

每個**可訓**參數要付：

```
權重 bf16    2 B
梯度 bf16    2 B
Adam m, v    8 B (fp32 ×2)
master 權重  4 B (fp32)
            ─────
            16 B / 可訓參數
```

```
Full FT:  7.6B × 16 B ≈ 122 GB   ← 未含 activation；單張 80 GB 塞不下
QLoRA:    base 4-bit 凍結  7.6B × 0.5 B ≈ 3.8 GB（凍結 = 不付梯度、不付 Adam）
          adapter 可訓    10M × 16 B  ≈ 0.16 GB
          activation（gradient checkpointing 後）≈ 2–6 GB
                                      ≈ 8–12 GB  ← 24 GB 消費卡輕鬆；M1 16 GB 統一記憶體勉強可
logit 蒸餾再加：teacher 權重（只 forward，不付梯度/Adam）+ teacher activation
```

省的大頭是**梯度 + optimizer state 那 14 B/參數**，凍結的參數完全不用付。QLoRA 的 base 在 VRAM 裡是 4-bit（3.8 GB），跟磁碟上 bf16 的 15 GB 無關——載入時 bitsandbytes 現場量化。

---

## 10. 權重檔的盒子：safetensors

**核心規則**：檔案格式 = 盒子；盒子裡每顆 tensor 用什麼編碼（dtype / 量化方案）是另一回事。safetensors 只管裝，不規定編碼。fine-tune 不「是」safetensors——訓練工具鏈要的是「一組帶名字、能讓梯度流的 PyTorch tensor」，safetensors 只是它們落地的盒子。

```
safetensors 檔
[8 bytes] header 長度 (u64, little-endian)
[N bytes] header：純 JSON
          { "model.layers.0.self_attn.q_proj.weight":
              { "dtype": "BF16", "shape": [3584, 3584], "data_offsets": [0, 25690112] }, ... }
[其餘]    所有 tensor 的原始 bytes，照 offset 排
```

**為什麼要發明它（2022，HF）**：之前的 `.bin`/`.pt` 是 pickle，載入 = 反序列化任意物件 = 可執行任意程式碼，下載陌生權重等於 `curl | python`。safetensors：零程式碼執行；header 讀完就知道全圖 → mmap、只載要的 tensor（`device_map` 分卡載入靠這個）；語言無關。

**「HF 那套標準」是三件事疊起來，safetensors 只是其一：**

```
HF repo
├─ config.json                      ← schema：架構名、層數、hidden、head 數、vocab、quantization_config
├─ model-00001-of-00004.safetensors ← 資料：權重分片
├─ model.safetensors.index.json     ← 哪個 tensor 在哪個分片
├─ tokenizer.json / tokenizer_config.json
└─ generation_config.json

transformers 庫 ← ORM：讀 config.json 蓋 PyTorch 模組樹 → 用 tensor 名字把 bytes 塞進對應 nn.Linear
```

比喻：safetensors = 資料庫資料檔；config.json = schema；transformers = ORM；GGUF = 為某一個引擎建好索引、壓縮過的唯讀快照。

**翻直覺的事實**：safetensors 也能裝量化權重。AWQ / GPTQ 版（vLLM 常用的 4-bit）發布就是 safetensors，裡面是 packed int32 + scale tensor，解碼規則在 `config.json` 的 `quantization_config` + loader 程式。「safetensors = 全精度」是錯的。

---

## 11. 權重檔的盒子：GGUF

**核心規則**：GGUF 是 llama.cpp 生態系的產物，為該引擎家族（llama.cpp / Ollama / LM Studio）打包的單檔。它把**編碼也寫進規格**：tensor type 是 enum（F16, Q8_0, Q4_K, Q6_K…），量化格子怎麼切是格式本身的一部分。

```
GGUF 單檔
├─ KV metadata
│   general.architecture = "qwen2"
│   qwen2.context_length = 32768
│   qwen2.block_count = 28, embedding_length = 3584, attention.head_count_kv = 4
│   tokenizer.ggml.tokens = [...]        ← 整個 vocab 內嵌
│   tokenizer.ggml.merges = [...]
│   tokenizer.chat_template = "..."      ← Jinja
└─ tensors
    blk.0.attn_q.weight   Q4_K  [3584, 3584]   ← 名字重排，不是 HF 的 model.layers.0.self_attn.q_proj
    blk.0.attn_k.weight   Q4_K  [3584, 512]
    blk.0.ffn_down.weight Q6_K  ...
```

單檔自足 = Modelfile 只需一行 `FROM`。名字重排 + 型別變 enum + tokenizer 內嵌，三件事加起來就是「不能拿回去訓練」。

**量化名字怎麼讀（Q4_K_M）**：`Q4` 每權重約 4 bit；`K` k-quant 家族（每 32 個權重一個 block、block 有自己的 scale、再疊 super-block scale）；`M` medium 混合——重要矩陣（attn_v、部分 ffn_down）用 Q6_K，其餘 Q4_K。所以平均不是 4 bit：

```
Q4_K_M ≈ 4.6–4.8 bits/weight
7.6B × 4.7 / 8 ≈ 4.5 GB   ← 對上 Ollama 的 4.7 GB
bf16  = 16 bits/weight
7.6B × 2 B ≈ 15.2 GB      ← 對上 HF 的 15 GB
差的 10 GB 全是精度；格式 overhead 是 KB 級
```

---

## 12. 格式 × 引擎 × 環境

**核心規則**：兩個都不是委員會標準，都是生態系產物，但角色不對稱——safetensors 是「原始碼」（實驗室發布的起點），GGUF 是「編譯產物」（從 safetensors 轉出來、只能跑在一個引擎家族上）。

```
模型實驗室發布（Qwen / Meta / Mistral）
        │  safetensors + config.json + tokenizer   ← 所有人從這裡出發
        ▼
┌─ HF 生態系 ──────────────────────────┐   ┌─ llama.cpp 生態系 ────────────────┐
│ transformers / PEFT（訓練）          │   │ convert_hf_to_gguf.py + quantize  │
│ vLLM / SGLang / TGI（Linux GPU 推理）│──▶│ GGUF                              │
│ MLX（M1 推理，吃 safetensors）       │   │ llama.cpp / Ollama / LM Studio    │
└──────────────────────────────────────┘   └───────────────────────────────────┘
```

| 你要做的事 | 格式 | 為什麼 |
|---|---|---|
| fine-tune | safetensors | PEFT 認模組樹名字、要 bf16 格子讓梯度流 |
| Linux + NVIDIA 多人推理（vLLM） | safetensors（bf16 或 AWQ） | vLLM 直接讀 HF repo，不經 GGUF（2026-10 時 GGUF 支援為實驗性） |
| M1 驗證（Ollama） | GGUF | Ollama = llama.cpp，只吃 GGUF |
| M1 驗證（MLX） | safetensors | Apple 框架，HF 格式 + 自己的量化欄位 |

你的兩個環境會用**不同格式跑同一顆模型**：M1 Ollama 的 `qwen2.5:7b`（GGUF Q4_K_M）和正式機 vLLM 的 `Qwen/Qwen2.5-7B-Instruct`（safetensors）是同一份權重的兩個編譯產物；輸出微小差異是精度差，不是格式差。

**誰的 kernel**：GGUF 的 Q4_K block 格子是為 CPU / Metal / 消費卡的 kernel 設計的；AWQ/GPTQ 的 packed int4 是為 NVIDIA tensor core 的 kernel 設計的。在 A100 上同樣 4-bit，走 AWQ safetensors 比走 GGUF 更可能跑得快——盒子相同（safetensors）不代表 kernel 相同，決定速度的是編碼對應的 kernel。

**可跑：看兩種盒子的 header，M1 / Linux 相同**

```bash
# safetensors
pip install -U huggingface_hub safetensors
hf download Qwen/Qwen2.5-0.5B --include "*.safetensors" --local-dir ./q05   # 2026-10 時 CLI 叫 hf；舊版 huggingface-cli
python peek.py q05/model.safetensors
```

```python
# peek.py：不用 torch，純讀 header
import json, struct, sys
with open(sys.argv[1], "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]   # 前 8 bytes = header 長度
    hdr = json.loads(f.read(n))             # header 是純 JSON
for name, m in hdr.items():
    if name == "__metadata__": continue
    if ".layers.0." in name or "embed" in name:
        print(f"{name:55s} {m['dtype']:5s} {m['shape']}")
```

```bash
# GGUF（直接看 Ollama 拉下來的 blob）
pip install gguf                                   # llama.cpp 官方 Python 套件，附 gguf-dump
ls -lS ~/.ollama/models/blobs/ | head -3           # M1：最大的 blob 是權重
gguf-dump ~/.ollama/models/blobs/sha256-<最大的> | head -80
# Linux（systemd 安裝）路徑改為 /usr/share/ollama/.ollama/models/blobs/，需 sudo
```

觀察：safetensors 的 `k_proj [128, 896]` 比 `q_proj [896, 896]` 小 = GQA（對上第 7 章 k/v 用 512 不用 3584）；GGUF 的 KV 裡找 `tokenizer.chat_template`、`context_length`；tensor 清單裡 `attn_v`、`ffn_down` 是 Q6_K、其他 Q4_K = `_M` 的意思；兩邊 shape 一一對應但名字、型別標法各一套。

---

## 思維鏈：一次 QLoRA 訓練 step（Linux + NVIDIA，SFT）

```
t   | CPU/GPU | 元件                     | 處理                | 發生什麼
----|---------|--------------------------|---------------------|---------------------------------------------------
t0  | CPU     | DataLoader + tokenizer   | 這個 batch 的 N 筆  | 文字 → token ids，pad/pack，搬進 VRAM
t1  | GPU     | base 權重 (4-bit, 凍結)  | 全部 token，逐層    | 每層先 dequantize 4-bit → bf16，算 W·x（kernel 執行）
t2  | GPU     | adapter A, B (bf16, 可訓)| 同層同批 token      | 算 A·(B·x) 加到 W·x；activation 存下（或 checkpointing 只存邊界）
t3  | GPU     | loss                     | 每個輸出位置        | CE 對 hard label（若 logit 蒸餾：teacher 另跑一次 forward，算 KL）
t4  | GPU     | autograd backward        | 全部層，反向        | 梯度要「流過」base 才能到前一層，但只「存」給 A、B；base 不存梯度
t5  | GPU     | AdamW                    | 10M 個 adapter 參數 | 更新 m、v、A、B；base 的 4-bit 格子一個 byte 不動
t6  | CPU     | trainer                  | —                   | log loss/step；到 save_steps 時存 adapter_model.safetensors（~20 MB）
```

對照：推論時掛 adapter（Ollama `ADAPTER`，不 merge），每生成一個 token：

```
t   | CPU/GPU | 元件            | 處理          | 發生什麼
----|---------|-----------------|---------------|-----------------------------------------
t0  | CPU     | 引擎 scheduler  | 這個請求      | 取上一個 token，查 KV cache 位置
t1  | GPU     | base (4-bit)    | 1 個新 token  | 逐層 dequantize + W·x（讀整份權重）
t2  | GPU     | adapter         | 同一 token    | 逐層多算 A·(B·x)，+0.9% 計算，不多讀 4-bit 權重
t3  | GPU     | lm_head/sampler | logits        | 取下一個 token；寫回 KV cache
```

從 HF 發布到 Ollama 跑起來（格式這條線）：

```
t   | 在哪台 | 元件                          | 輸入 → 輸出                                  | 格式/精度
----|--------|-------------------------------|----------------------------------------------|------------------------
t0  | HF Hub | 實驗室發布                    | — → safetensors bf16 + config.json + tokenizer | safetensors / bf16，15 GB
t1  | 訓練機 | transformers + PEFT（QLoRA）  | safetensors → adapter_model.safetensors        | safetensors / bf16，20 MB；base 現場 4-bit
t2  | 訓練機 | PEFT merge_and_unload         | base bf16 + adapter → merged safetensors       | safetensors / bf16，15 GB（merge 必須在 bf16 上）
t3  | 訓練機 | llama.cpp convert_hf_to_gguf  | merged safetensors → GGUF F16                  | GGUF / F16；名字重排、tokenizer 內嵌
t4  | 訓練機 | llama-quantize                | GGUF F16 → GGUF Q4_K_M                         | GGUF / ~4.7 bit，4.7 GB
t5  | 區網   | 複製 + ollama create（Modelfile FROM ./x.gguf） | GGUF → manifest + blob      | 同上，blob 共用
t6  | M1/正式機 | Ollama（llama.cpp，Metal / CUDA） | 載入 GGUF → 推論                            | 推論時 dequantize block → 計算
（不 merge 的替代路：t2–t4 跳過，Modelfile 用 FROM base.gguf + ADAPTER ./adapter/，llama.cpp 推論時加 A·(B·x)）
```

---

## 陷阱清單

- ❌ fine-tune 給增量、蒸餾給完整模型 → ✅ 兩者只差「訓練訊號從哪來」（軸 A）；增量/完整是軸 B，獨立 → 三條軸綁在一起會讓你以為「用 LoRA 就不能蒸餾」或「蒸餾就必須 full」。
- ❌ LoRA 能蒸餾是因為來源是 other model → ✅ LoRA 只管梯度流到哪些參數，loss 是 CE 還是 KL 都一樣 → 更新範圍與 loss 來源正交。
- ❌ 受體是 pretraining → ✅ 受體是 pretrained **checkpoint**（權重檔）；pretraining 是動作 → fine-tune 的定義就是「初始化載 checkpoint」，分類依據只有這條。
- ❌ training new model 算 fine-tune → ✅ 隨機初始化 = pretraining；CPT 在中間（載 checkpoint 但吃大量原始文本）。
- ❌ 改 Modelfile 的 SYSTEM / 加 skill / RAG 也在「調模型」 → ✅ 改 context 不改 W，連訓練步都沒發生 → `ollama create` 只寫 manifest，權重 blob 共用。
- ❌ adapter 是向量檔 → ✅ 兩個矩陣 A（d×r）、B（r×d）→ 當向量算大小會少一個維度。
- ❌ 秩 = 列數 / 「秩 ≤ 16 表示結果不超過 16 列」 → ✅ 秩 = 獨立原料數；A·B 仍是 3584 列，只是 3584 列全由 16 條原料調出 → 列數是成品數量，秩是原料數量。
- ❌ B·A 不能乘 / 秩看外側維度 → ✅ 只要內側 k 相等就能乘；rank ≤ min(結果列, 結果欄, k) → 矩陣乘法最常見的看錯方向。
- ❌ 低秩 = 局部修改 → ✅ rank-1 = u·vᵀ，每一格都有值；受限的是方向不是位置 → 若信「局部」，會錯誤預測「無關輸入不受影響」；實際上只要 x 跟 v 有投影就被推，LoRA 一樣會傷通用能力，評測防退步照做。
- ❌ 低秩所以優 → ✅ 是約束，是賭「任務的 ΔW 本來就低秩」；賭輸 underfit → 新語言/新領域類任務 full 或 CPT 較好。
- ❌ rank 只是訓練時的複雜度指標，推論無差 → ✅ ΔW 的永久性質；merge 後計算成本等同 full FT，但 ΔW 秩 ≤ r 的天花板永遠在。
- ❌ fine-tune 能讓模型背內部文件（反覆撞）→ ✅ 改 W 改習慣（格式、黑話映射），事實庫是 context/RAG 的事 → W 裝不下也裝不穩。
- ❌ 訓越久越好（train loss 還在降）→ ✅ loss 沒有保留項，epoch 多到一定程度就開始吃掉通用能力；用 held-out 一般題集的指標中途偵測，不看 train loss。
- ❌ fine-tune「是」safetensors / 訓練綁定某個檔案格式 → ✅ 訓練綁定的是「帶名字、bf16、能讓梯度流」的 tensor 組 + config.json 的模組樹；safetensors 只是落地的盒子 → GGUF 名字重排 + 型別 enum + 無 bf16 格子，所以回不去。
- ❌ safetensors = 全精度、GGUF = 量化 → ✅ 盒子不決定精度：safetensors 每顆 tensor 自己標 dtype，AWQ/GPTQ 4-bit 也是 safetensors；GGUF 則把量化格子寫進格式規格 → 「檔案格式 ≠ 數值精度」（反覆撞）。
- ❌ HF 的 15 GB vs Ollama 的 4.7 GB 是格式造成的 → ✅ bf16 16 bit vs Q4_K_M ~4.7 bit，全是精度；格式 overhead 是 KB 級。
- ❌ safetensors / GGUF 其中一個是「標準」 → ✅ 都是生態系產物；safetensors 是發布起點（原始碼），GGUF 是為 llama.cpp 家族編譯的產物；vLLM、MLX、PEFT 都走 safetensors，Ollama 走 GGUF。
- ❌ 同樣 4-bit 在 A100 上速度應該差不多 → ✅ 決定速度的是編碼對應的 kernel：GGUF Q4_K 的 block 格子為 CPU/Metal 設計，AWQ 的 packed int4 為 tensor core 設計 → 盒子相同不代表 kernel 相同。

---

## 心智模型總結

```
fine-tune   := 載 checkpoint 繼續訓練       （初始化決定分類）
蒸餾        := 訊號來自 teacher 的訓練       （只動軸 A；學生已 pretrained 時是 fine-tune 的子集）
三軸獨立    := 訊號來源 × 更新範圍 × 學生架構
改 context  := prompt / RAG / SYSTEM / skill   W 不動
改 W        := fine-tune                       改習慣，不裝事實
秩          := 獨立原料數；rank(X·Y) ≤ min(列, 欄, k)
LoRA        := ΔW = A·B，≤ r 條 if-then，每格都改、方向受限、位置不限
省在哪      := 凍結參數不付 梯度 2B + Adam 8B + master 4B
merge       := 成本差消失，表達力差不消失；4-bit base 不能 merge → 推論時走 W·x + A·(B·x)
歪掉        := loss 無保留項 → 忘；LoRA 限 r 維子空間 → 較難忘，不是不會
盒子        := safetensors（HF 原始碼：訓練 / vLLM / MLX）  vs  GGUF（llama.cpp 編譯產物：Ollama）
盒子 ≠ 精度 := dtype / 量化方案 是 tensor 自己的事；GB 數差來自 bit/權重
回不去      := GGUF 名字重排 + 型別 enum + 無 bf16 格子 → 訓練永遠從 safetensors 起
```
