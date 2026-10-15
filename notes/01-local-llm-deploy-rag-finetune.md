# 自建地端 LLM 應用：從部署到 fine-tune 的完整筆記

- 記錄日期：2026-10-15
- 學習方式：概念優先（conceptual-first），先建立全貌再動手部署
- 目前定位：概念整理階段，尚未實際部署；下一步是在 M1 本機驗證流程

> 整理自 2026-10 的一系列討論。目標：公司內部約 40 人試用，先在區網提供聊天服務，後續加入 RAG 與 fine-tune，讓模型理解公司黑話與內部 DSL。

---

## 目錄

1. [三層架構：模型 / 推理引擎 / 應用層](#1-三層架構模型--推理引擎--應用層)
2. [部署路線圖](#2-部署路線圖)
3. [平台差異：Mac / Windows / Linux 與 Docker](#3-平台差異mac--windows--linux-與-docker)
4. [推理引擎：Ollama 與 vLLM](#4-推理引擎ollama-與-vllm)
5. [模型內部：一個 token 的生成流程](#5-模型內部一個-token-的生成流程)
6. [模型本身：尺寸、結構、精度、格式](#6-模型本身尺寸結構精度格式)
7. [三條路：純推理 / RAG / fine-tune](#7-三條路純推理--rag--fine-tune)
8. [fine-tune 深入](#8-fine-tune-深入)
9. [黑話與 DSL 的策略](#9-黑話與-dsl-的策略)
10. [Harness 與 Claude 整合](#10-harness-與-claude-整合)
11. [決策速查與常見陷阱](#11-決策速查與常見陷阱)

---

## 1. 三層架構：模型 / 推理引擎 / 應用層

任何地端 LLM 服務都可以拆成三層，遇到新工具先分清楚它在哪一層，就不會混淆。

```
┌─────────────────────────────────────────────────────────┐
│  應用層（客戶端）                                          │
│  Open WebUI、自己寫的程式、MCP server、LangChain、agent    │
│  → 透過 OpenAI 相容 API 跟下面講話                          │
└───────────────────────────┬─────────────────────────────┘
                            │  HTTP / JSON / SSE 串流
┌───────────────────────────▼─────────────────────────────┐
│  推理引擎（runtime / inference server）                    │
│  Ollama（包 llama.cpp）、vLLM、llama.cpp server、SGLang、TGI │
│  → 把權重載進 GPU、提供 API、排程多個請求                     │
└───────────────────────────┬─────────────────────────────┘
                            │  載入
┌───────────────────────────▼─────────────────────────────┐
│  模型（權重檔）                                            │
│  Qwen、Gemma、Llama、Mistral、DeepSeek…                   │
│  → 幾 GB 到幾十 GB 的數字，自己不會動                        │
└─────────────────────────────────────────────────────────┘
```

類比：模型是資料庫檔案，推理引擎是資料庫伺服器，應用層是查資料的介面。

三個常見誤解：

| 誤解 | 正確理解 |
|---|---|
| vLLM 是 Ollama 的套件 | 兩者是同一層的獨立競爭者，互不包含 |
| 用 Ollama 就不能用 Gemma | Ollama 是 runtime，Gemma 是模型，Ollama 可以跑 Gemma、Qwen、Llama 任何一個 |
| 換模型要換整套 | 應用層只認 API，換模型或換引擎都不用動上層程式 |

### 引擎的兩個子層

嚴格來說「推理引擎」還可以再分：

- **引擎（engine）**：真正做矩陣運算的核心。llama.cpp、vLLM 的 CUDA 核心、TensorRT-LLM
- **伺服器（server）**：包在引擎外面，提供 API、管理模型、處理併發。Ollama（包 llama.cpp）、vLLM 自帶的 OpenAI server

日常使用混著講即可。

---

## 2. 部署路線圖

### 需求背景

- 內部使用，約 40 人，實際同時在線通常 5–10 人
- 嘗試性質，先讓大家在區網內用起來
- 後續目標：理解公司黑話、解讀內部 DSL（「新解釋器」）
- 資料敏感性是選擇地端的主因

### 階段一：讓大家先用起來（1–2 週）

目標只有一個：區網內一個網址，大家登入就能聊。

| 項目 | 選擇 | 理由 |
|---|---|---|
| 硬體 | 一台 Linux 伺服器 + 一張 24–32 GB 的 GPU（RTX 4090 / 5090），48 GB 更從容 | 40 人不會同時在線，單卡夠 |
| 模型 | Qwen 14B 或 27B 的 4-bit 量化版；Gemma 4 也可 | 中文能力好，授權寬鬆 |
| 引擎 | 先 Ollama，嫌慢再換 vLLM | 省事優先 |
| 前端 | Open WebUI | 自帶帳號、對話歷史、檔案上傳、基本 RAG |
| 部署 | Docker Compose | 一次帶起引擎 + 前端 |

這階段最重要的產出不是系統，是「大家實際怎麼用、哪裡答不好」的紀錄。這些就是後面 RAG 的需求和 fine-tune 的訓練資料。

### 階段二：餵內部知識（RAG）

多數「希望它懂公司內部的東西」的需求，RAG 就能解決，而且改文件就生效、不用重訓。

1. 整理內部文件（規格書、Wiki、SOP、程式註解），先選幾個高價值的
2. 選中文表現好的 embedding model（bge-m3 一類），向量庫用 Qdrant 或 pgvector
3. 先用 Open WebUI 內建的 RAG 試，效果不夠再自己寫 pipeline
4. 建一組測試題，量化回答品質，之後每次調整都跑同一組

### 階段三：fine-tune

只有在 RAG 解決不了時才做：改變模型的「風格 / 行為 / 固定格式」，或讓它理解一套內部術語和語法。

1. 用 LoRA / QLoRA，工具選 LLaMA-Factory 或 Unsloth，24 GB 卡可以訓 14B
2. 資料來源：階段一累積的對話 + 人工整理的 DSL 配對範例。幾百到幾千筆高品質資料勝過大量垃圾
3. 訓完一定用階段二的測試題比對，確認沒退步
4. GPU 可以白天推理、晚上訓練，但不能同時

### 本機先玩：M1 Mac

M1 是最適合先驗證流程的環境，但技術堆疊要換：

- 引擎用 Ollama（原生跑，吃 Metal）或 MLX，**不能**用 vLLM
- 只有 Open WebUI 放進 Docker，模型不放（原因見第 3 節）
- 依統一記憶體選模型：16 GB → 9B；32 GB → 14B 舒服、27B 勉強；64 GB → 27B/32B 正常
- 能驗證：模型選型、prompt 和術語表設計、RAG pipeline、使用流程
- 不能驗證：多人併發效能、正式環境 VRAM 配置

---

## 3. 平台差異：Mac / Windows / Linux 與 Docker

| | 模型 runtime | 模型可放進 Docker？ | vLLM |
|---|---|---|---|
| Mac M 系列 | Ollama / MLX 原生 | 否（容器吃不到 GPU） | 不支援 |
| Windows + NVIDIA | Ollama 原生 | 可（經 WSL2） | 要在 WSL2 內 |
| Linux + NVIDIA | Ollama 或 vLLM | 可 | 可 |
| AMD GPU | Ollama（ROCm） | 可，但支援卡型有限 | 有，次要 |

### 為什麼 Mac 的 Docker 吃不到 GPU

不是因為 Mac 沒有 GPU（M1 有，而且統一記憶體跑模型很舒服）。問題出在 Docker 的架構：

```
Linux 主機                          Mac 主機
┌──────────────┐                   ┌──────────────────────────┐
│ 容器          │                   │ 容器                      │
│  ↓ 共用 kernel │                   │  ↓                       │
│ Linux kernel  │ ← NVIDIA 驅動     │ Linux VM（Docker Desktop）│
│  ↓            │   + Container     │  ↓ 沒有 Apple GPU 的      │
│ GPU           │   Toolkit 映射    │    Linux 驅動，也沒直通    │
└──────────────┘                   │ macOS（Metal 驅動在這）    │
                                   │  ↓                       │
                                   │ Apple GPU                │
                                   └──────────────────────────┘
```

1. Docker 容器本質是 Linux 程序，必須跑在 Linux kernel 上
2. macOS 不是 Linux，所以 Docker Desktop 先開一台 Linux 虛擬機，容器跑在裡面
3. Apple GPU 只有 macOS 的 Metal 驅動，Linux 側沒有可用驅動，Docker Desktop 也沒做 GPU 直通
4. VM 裡的 Linux 看不到 GPU，只剩 CPU

Windows 的 Docker Desktop 同樣跑 VM（WSL2），但 NVIDIA 有做 WSL2 的 GPU 直通驅動，所以能吃到。

結論：**主因是虛擬化架構（多一層 VM），次因是驅動（Linux 側沒有 Apple GPU 驅動）**。統一記憶體不是原因，反而是 Mac 原生跑模型的優勢。

### 主機層前置作業（Linux）

compose 之前要做好的事，實際踩坑多半在這裡：

```bash
# 1. NVIDIA 驅動裝好
nvidia-smi

# 2. Docker + NVIDIA Container Toolkit，這步沒過 compose 的 GPU 設定全無效
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# 3. 模型檔先拉到本機 volume，不要讓容器每次啟動重抓
huggingface-cli download Qwen/Qwen3-14B-AWQ --local-dir ./models/qwen3-14b-awq
```

### 最小 compose 範例（Linux + vLLM）

```yaml
services:
  vllm:
    image: vllm/vllm-openai:v0.x.y        # 釘版本，不要 latest
    runtime: nvidia
    environment:
      - HF_HOME=/models
    volumes:
      - ./models:/models
    command: >
      --model /models/qwen3-14b-awq
      --quantization awq
      --max-model-len 16384
      --gpu-memory-utilization 0.9
      --served-model-name qwen
    ports: ["8000:8000"]
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: all, capabilities: [gpu]}]

  webui:
    image: ghcr.io/open-webui/open-webui:main
    environment:
      - OPENAI_API_BASE_URL=http://vllm:8000/v1
      - OPENAI_API_KEY=none
    volumes:
      - ./webui-data:/app/backend/data   # 對話歷史和使用者都在這，要備份
    ports: ["3000:8080"]
    restart: unless-stopped
    depends_on: [vllm]
```

### M1 上的啟動方式

```bash
# 1. Ollama 原生跑（吃 Metal），不進容器
brew install ollama
ollama serve                       # 或直接開 app
ollama pull qwen3:14b              # 或 gemma4:12b

# 2. 只有 Open WebUI 進 Docker，指向宿主機的 Ollama
docker run -d -p 3000:8080 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

Ollama 也提供 OpenAI 相容 API（`localhost:11434/v1`），所以在 Mac 上寫的程式之後搬到 Linux + vLLM 只要改 base URL。

### 上線後要補的

- `max-model-len` 和 `gpu-memory-utilization` 照 VRAM 試，太大啟動 OOM，太小 context 不夠
- Open WebUI 第一個註冊帳號是 admin，之後關掉註冊或改成需審核
- 區網給內部 DNS，前面擺 nginx / Caddy 做 https，不然瀏覽器一些功能在 http 下會被擋
- `restart: unless-stopped` + 簡單監控（至少看 `nvidia-smi` 有沒有人把卡吃滿）
- 整個流程熟手半天，第一次做留一天給踩坑

---

## 4. 推理引擎：Ollama 與 vLLM

### 兩者的設計基因

| | Ollama（llama.cpp） | vLLM |
|---|---|---|
| 出身 | 社群志工專案，C/C++，從 CPU 起家 | UC Berkeley Sky Computing Lab（2023，PagedAttention 論文），後併入 PyTorch Foundation |
| 設計目標 | 哪裡都能跑、單人本機體驗最好 | 在資料中心的 N 卡上同時服務幾百個請求 |
| 後端 | CUDA、Metal、Vulkan、ROCm、SYCL、CPU，七八個 | CUDA 為主，ROCm 次要，無 Metal |
| 模型格式 | GGUF（自家格式，要轉換） | HuggingFace 原生格式，不用轉 |
| 量化 | 自家 Q4_K_M、Q5_K_M 等二十幾種 | AWQ、GPTQ、FP8 |
| 多卡 | 層切分（依序經過各卡，同時只一張在算） | tensor parallel（每層切開，所有卡同時算） |
| 參數 | 幾乎不用調 | 要自己算好再開 |
| VRAM 不足時 | 自動把部分層放 CPU，能動但慢 | 啟動失敗，直接報錯 |

**不是 NVIDIA 出的**：vLLM 來自學界和開源社群。NVIDIA 自家的推理引擎是 TensorRT-LLM，效能更高但封閉難用，一般不碰。vLLM 快不是因為有硬體廠內線，是設計目標不同。

**不是語言問題**：兩邊核心都是 C++ / CUDA，vLLM 的 Python 只是外殼。

### 一個請求的生命週期

```
harness 送 prompt（HTTP POST，JSON）
   │
   ▼
引擎 server 收到（Python async / Go goroutine）
   │   ✗ 不是每個請求開一個 process：模型權重住在 VRAM，
   │     載入要幾十秒，開新 process 等於重載，不可能
   │   ✓ 一個常駐引擎 process，模型只載一次，所有請求進同一個 scheduler queue
   ▼
tokenizer（CPU）：文字 → token id
   │
   ▼
scheduler：這一步要算哪些序列？組成 batch
   │
   ▼
GPU forward pass：一次處理整個 batch
   │   prefill：把 prompt 全部餵進去，建 KV cache（計算密集）
   │   decode： 之後每步只算 1 個新 token（頻寬密集，大部分時間在這）
   ▼
sampling：從 logit 挑下一個 token
   │
   ▼
detokenize（CPU）→ SSE 串流回去
   │
   └──→ 新 token 接回序列，回到 scheduler，下一步
         請求結束 → 這條序列的 KV cache 釋放
```

CPU 與 GPU 的分工：

- **CPU 是指揮**：tokenize、排程、組 batch、下 kernel（非同步）、拷回 token id、detokenize、串流
- **GPU 是工人**：權重、KV cache、所有矩陣運算都在 VRAM，資料不過 CPU 的手
- Ollama 的細節：Go server 會為每個「被載入的模型」開一個 llama.cpp runner 子 process（per model，不是 per request），裡面有 N 個 slot

### 多人平行的本質：GPU 的 batch 維度，不是 CPU thread

這是理解效能差異的核心。同時 8 個人在用，引擎**不是**開 8 個 thread 各算各的，而是：

```
每一步（step）：
  scheduler 挑出目前活著的序列，例如 8 條
  組成一個 batch，張量 shape = [8, seq_len, hidden]
  一次 forward pass 跑完整個模型
  → 得到 8 個新 token，一人一個
  各自 append、串流回去
  下一步
```

一次 kernel call 同時處理所有人。GPU 本來就是為大 batch 設計的：decode 階段的瓶頸是「每步把全部權重從 VRAM 讀進運算單元」的頻寬，讀一次權重順便算 8 條幾乎免費。

**batch 是什麼**：深度學習的資料形狀概念，不是 OS 概念。它是張量的第一個維度，「這一步同時算的序列數」。它不是任務也不是排程實體，排程實體是上層的 scheduler。

### 差異在三層，不只 kernel

```
scheduler   ← 這一步誰進 batch、prefill 和 decode 怎麼混
memory      ← KV cache 怎麼配（固定 slot vs 分頁）
kernel      ← batch 餵進去之後算得多快
```

| 層 | Ollama（llama.cpp） | vLLM |
|---|---|---|
| scheduler | 固定 N 個 slot，slot 滿就排隊；長 prompt 的 prefill 會卡住其他人的 decode | continuous batching：每步動態進出，有人結束立刻補位；chunked prefill 把長 prompt 切片混進 batch |
| memory | 每個 slot 預留固定連續空間（`num_ctx` 大小） | PagedAttention：分頁式 KV cache，用多少配多少 |
| kernel | 為 batch=1 優化；有 flash attention（`-fa`） | FlashAttention / FlashInfer、PagedAttention kernel、Marlin 量化 kernel、CUDA Graphs、算子融合 |

**瓶頸不在 CPU**：scheduler 本身的計算量是微秒等級。瓶頸是「上層策略決定了 GPU 能不能被餵飽」：固定 slot 浪費 VRAM → 併發上限低 → batch 小 → 頻寬白費；prefill 卡住 decode → GPU 時間分配不均。

**為什麼 llama.cpp 不改掉**：分頁式 KV cache 要重寫 attention kernel，而 llama.cpp 有七八個後端、二十幾種量化格式，任何 kernel 改動要全部做一遍；它的核心承諾是可攜性，不是榨乾一張 N 卡；社群是志工主導。它在進步（flash attention、統一 KV cache），差距在縮小，但優先順序不同。

**還 Ollama 一個公道**：llama.cpp server 模式**有**把多個 slot 疊成 batch 一起 forward，也有自己的 continuous batching。差在**程度**，不是有無。

### 效能感受

| 場景 | 結果 |
|---|---|
| 一個人問一句 | 差不多，看量化格式。瓶頸都是記憶體頻寬，同一個物理限制 |
| 同時 1–5 人 | 兩者接近 |
| 同時 5–15 人 | Ollama 開始卡，vLLM 還順 |
| 同時 20 人以上 | vLLM 吞吐可能 2–4 倍，Ollama 基本不行 |

### 何時從 Ollama 換 vLLM

條件式：**N 卡 + 多人 + 開始嫌慢** 三個湊齊才換，不是一有 N 卡就換。換的時候前端 Open WebUI 只改一行 API 位址。

vLLM 在 N 卡上的具體優勢：
1. PagedAttention + continuous batching（多人時最主要）
2. 多卡 tensor parallel（70B 這種單卡放不下的）
3. 量化 kernel 更快（AWQ / GPTQ / FP8 專用 CUDA kernel）
4. 直接吃 HF 格式（新模型當天能跑，LoRA adapter 直接掛）
5. 正式環境該有的：Prometheus metrics、結構化輸出、tool calling 解析

這五項全建立在 CUDA 上，離開 N 卡就全沒了。

### slot 是什麼

llama.cpp server 自己的叫法，不是 ML 通用詞。一個 slot = 一個併發座位 = 一塊預留的 KV cache + 這條序列的狀態。Ollama 的 `OLLAMA_NUM_PARALLEL` 設的就是 slot 數。

**slot 是記憶體和狀態的單位，不是算力單位**。所有 slot 共用同一顆 GPU，每步疊成一個 batch。像停車格：格子數決定能停幾台，但大家開的是同一條路。

### VRAM 的分配與上限

**引擎 process 的記憶體**

```
GPU 側（VRAM，大頭）              CPU 側（RAM，配角）
├─ 模型權重                       ├─ Python / Go runtime
│   32B 4-bit ≈ 18–20 GB         ├─ tokenizer
│   32B BF16 ≈ 64 GB             ├─ CUDA context
├─ KV cache                       └─ 載入時權重經過 RAM 再上 GPU
│   每條序列隨 context 增長          （llama.cpp 用 mmap 可不全進 RAM）
│   多人時可比權重還大
└─ activation 暫存（相對小）
```

VRAM 可以用「GPU 版 heap」來理解：CUDA 有自己的配置器（`cudaMalloc`，PyTorch 再包一層 caching allocator）。

**vLLM 的 `--gpu-memory-utilization`**

語意是「這個 process 最多可以用總 VRAM 的多少比例」，預設 0.9，**是可調參數不是硬性要求**。啟動時依序：

```
1. 檢查卡上目前空閒 VRAM ≥ 總容量 × 比例？否 → 報錯退出
2. 載入模型權重                            放不下 → OOM
3. 跑一次假 forward 量測 activation 需求
4. 剩下的全部切成 KV cache 分頁池
5. 算出分頁夠放幾個 token，對照 max-model-len
   連一條都裝不下 → 報錯，訊息會說「KV cache 只能放 N，max-model-len 是 M」
```

全部都是**啟動時報錯，不會偷偷降級，也不會死機**。

兩個 process 共用一張卡：各自分配，加總 ≤ 1，例如 0.45 + 0.45 或 0.6 + 0.3。兩個都用預設 0.9 開，第二個啟動時發現空閒只剩 10%，直接報錯退出。

為什麼預設 0.9：vLLM 的設計假設是「一張卡專門服務一個模型」，資料中心常態，KV cache 開到最大才對。要分享就得自己調。

**啟動後是硬上限**

| | vLLM | Ollama |
|---|---|---|
| 併發超過上限 | 新請求在 queue 等 | 超過 slot 數排隊 |
| 運行中 KV 空間不夠 | 搶佔：暫停某條序列，KV 丟掉或搬 CPU，之後重算。慢但不崩 | 可能 runner 子 process OOM 崩掉，Ollama 自動重啟，那批請求回錯誤 |
| 單一請求超過 context 上限 | `max-model-len` 是硬上限，超過回 HTTP 400，**不會偷砍** | 超過 `num_ctx` 會**砍掉最前面的 token 繼續跑**，使用者感受是「它忘了前面說的」 |
| 載入時放不下 | 啟動失敗 | 自動把部分層放 CPU 跑，能動但慢很多 |

Ollama 預設 `num_ctx` 只有 4K 或 8K，這是「為什麼它變笨了」最常見的原因，試用時記得調大。

**context 遺失和 VRAM 不足是兩件事**：前者是 `num_ctx` 上限觸發的截斷，後者是啟動失敗或程序崩潰。兩個引擎都不會因為 VRAM 不夠而「悄悄丟掉一部分記憶」。

### 多個模型的載入行為

| | vLLM | Ollama |
|---|---|---|
| 啟動時 | `--model` 指定的那一個，載了就不放 | 什麼都不載 |
| 多模型 | 一個 process 一個模型，要開多個 process 各分 VRAM | 用到才載，VRAM 夠就並存，不夠就踢掉最久沒用的（LRU） |
| 閒置 | 不卸載 | `OLLAMA_KEEP_ALIVE` 預設 5 分鐘後卸載，設 `-1` 永不卸 |
| 上限 | VRAM 加總 | `OLLAMA_MAX_LOADED_MODELS` 預設 GPU 數 × 3 |

三個模型分別 10 / 20 / 30 GB 要**同時常駐並服務**：權重 60 GB + 各自的 KV cache + activation ≈ 70–80 GB，意思是 80 GB 的 A100 / H100 或兩張 48 GB 卡。

不夠的話：vLLM 第三個啟動失敗；Ollama 不會失敗但會「踢來踢去」，每次切換重載十幾秒，體感極慢。

多人環境建議：**只開放一個主力聊天模型**，keep_alive 設不卸載，加上 RAG 用的 embedding model（幾百 MB，可一直並存）。真需要「三種用途」，一個底模 + 三個 LoRA adapter 通常就夠，VRAM 只要一份。

---

## 5. 模型內部：一個 token 的生成流程

### 完整管線

```
[CPU] tokenizer：文字 → token id 序列（BPE，字串演算法，序列依賴，無法平行）
         │
         ▼
[GPU] embedding：查表，token id → 向量（從 [vocab × hidden] 的權重矩陣取一列）
         │
         ▼
[GPU] Transformer 層 ×N（7B 約 32 層，70B 約 80 層，依序經過，不是 loop）
      每一層：
        ┌─ RMSNorm
        ├─ attention：讓 token 之間互相看
        │     q = W_q·x   「我在找什麼」
        │     k = W_k·x   「我有什麼可被找到」
        │     v = W_v·x   「找到我，我給你什麼」
        │     分數 = q · 前面所有 k → softmax → 加權所有 v → W_o 整理
        │     （前面 token 的 k、v 存在 KV cache，不重算）
        ├─ residual add
        ├─ RMSNorm
        ├─ MLP：每個 token 自己加工，跟其他 token 無關
        │     h = SiLU(W_gate·x) ⊙ (W_up·x)    4096 → 11008
        │     y = W_down·h                      11008 → 4096
        └─ residual add
         │
         ▼
[GPU] final norm → lm_head（投影到 vocab 大小，每個 token 一個 logit）
         │
         ▼
[GPU/CPU] sampling：從 logit 挑下一個 token（temperature、top-p）
         │
         ▼
[CPU] detokenize → 串流出去
         │
         └──→ 新 token 接回序列尾巴，回到 embedding，下一步（這才是 loop）
```

### tokenizer 與 embedding：兩種「查表」

| | 輸入 | 輸出 | 本質 | 放哪 |
|---|---|---|---|---|
| tokenizer | 字串 | 整數陣列 | 字串演算法（BPE 合併，有先後依賴） | CPU |
| embedding | 整數陣列 | 向量矩陣 | 從權重矩陣取列（gather） | GPU |

```
"自建地端模型"  →  tokenizer  →  [36584, 12021, 7890, 2234]
36584          →  embedding  →  [0.021, -0.187, 0.442, ...]  （長度 = hidden size）
```

tokenizer 放 GPU 沒意義：資料量極小、邏輯是 if/else 和 hash map、合併步驟互相依賴、搬資料和下 kernel 的開銷比做完還久。CPU 和 GPU 之間傳的只是那串整數。

**tokenizer 跟特定 LLM 綁死**：訓練時用哪套 tokenizer，就只認那套 id。`36584` 在 Qwen 的 embedding 表裡是「自」，在 Llama 的表裡可能是不相干的東西。每個模型附自己的 `tokenizer.json`，換模型一定換 tokenizer。它只是執行在 CPU，不是獨立於模型。類比：資料庫的 schema。

串流時偶爾看到一個中文字先出現半個亂碼，就是 detokenize 還沒湊齊完整的 UTF-8 字元。

### attention 與 MLP 的分工

| | attention | MLP |
|---|---|---|
| 做什麼 | token 之間互相看，「該注意誰」 | 每個 token 自己處理自己，「拿到資訊後怎麼變」 |
| 矩陣 | W_q、W_k、W_v、W_o，各 4096×4096 ≈ 1,677 萬 | W_gate、W_up、W_down，各 4096×11008 ≈ 4,500 萬 |
| 參數佔比 | 約 1/3 | 約 2/3 |
| 成本隨 context 增長？ | 是（要讀前面所有 token 的 KV cache） | 否（每 token 固定） |
| 存什麼 | 「該看誰」的模式 | 普遍認為是「知識」存放處，像模糊的 key-value 記憶 |

**為什麼效能討論一直講 attention**：它是唯一成本隨 context 長度增長的部分，而且它綁著 KV cache，KV cache 是多人服務時的資源瓶頸。不是因為它是 GPU 主要在做的事，MLP 才是計算大頭。

**MLP** = Multi-Layer Perceptron，1960 年代的詞，本質就是幾個矩陣乘法夾著一個非線性函數。現代模型多用 SwiGLU 變體（多一個 gate 矩陣）。

### attention kernel 與 KV cache 的排法

attention kernel 是「讀 KV cache、跟 Q 算、輸出」這段 GPU 程式，它是 KV cache 的唯一讀者。KV cache 怎麼排在 VRAM，kernel 就得知道怎麼讀。

**llama.cpp：連續配置**

```
VRAM:
slot 0: [████████░░░░░░░░░░░░░░░░]  用了 1/3，其餘預留
slot 1: [██░░░░░░░░░░░░░░░░░░░░░░]  用得很少，其餘預留
slot 2: [████████████████████░░░░]  用得多
         ↑ 空白全部浪費，不能給別人用（內部碎片）
```

**vLLM：分頁（PagedAttention）**

```
VRAM block 池（每塊例如 16 個 token）:
[B0][B1][B2][B3][B4][B5][B6][B7][B8]...
序列 A 的 block table: B0 → B3 → B7
序列 B 的 block table: B1 → B2
序列 C 的 block table: B4 → B5 → B6 → B8
         ↑ 用多少配多少，實體位置不需連續
```

這就是作業系統虛擬記憶體的 page table 搬到 VRAM 上。

**為什麼要改 kernel 而不只改配置器**：kernel 讀 KV cache 時不能再「從起點連續讀」，要先查 block table 找實體位置再跳過去讀，這個間接定址要寫進 CUDA kernel 且要夠快。配置策略 + 配合它的 kernel，缺一不可。同樣 VRAM，分頁能塞的併發數常常是 2–4 倍。

**FlashAttention 是另一個維度**：把 Q·K、softmax、×V 融合在一個 kernel，中間結果留在 GPU SRAM 不寫回 VRAM。跟記憶體排法無關，純算法優化。llama.cpp 也有實作，所以「kernel 算得快不快」兩邊差距不大，結構性差距在分頁。

### prefill 與 decode

| | prefill | decode |
|---|---|---|
| 做什麼 | 把整段 prompt 一次餵進去，算出每個 token 的 KV cache | 每步只算 1 個新 token |
| 瓶頸 | 計算密集（compute-bound），prompt 越長越久 | 頻寬密集（memory-bound），每步要讀全部權重 + 這條序列的 KV cache |
| 時間佔比 | 一次 | 大部分時間在這 |

生成 500 個 token = 整個 32 層跑 500 遍。這就是為什麼 decode 瓶頸是頻寬：每遍都把全部權重從 VRAM 讀一次。也是為什麼 4-bit 量化不只省空間還變快：讀的資料量少了 1/4。

### KV cache 不跨請求

KV cache 是每個請求自己的，請求結束就釋放。下一個人問同樣的問題也是重新 tokenize、重新 prefill。

例外是 **prefix caching**（vLLM 和 llama.cpp 都有）：多個請求的 prompt 開頭完全相同（例如同一段 system prompt），引擎會留著那段的 KV cache 重用。RAG 檢索的 chunk 每個問題都不同，命中率低，實務上不太能靠它。

---

## 6. 模型本身：尺寸、結構、精度、格式

### Qwen 目前的主要尺寸（2026-10）

- Qwen 3.6（2026-04）：27B dense、35B-A3B MoE
- Qwen 3.5：4B、9B、27B
- Qwen 3（2025）：0.6B–32B 的 dense 系列，30B-A3B 和 235B-A22B MoE。dense 14B 和 32B 是 3.5/3.6 沒出的尺寸，仍常用
- 授權：大多 Apache 2.0；Gemma 用 Google 自己的 Gemma Terms of Use，內部用沒問題，對外服務要看條款

### dense vs MoE

差異只在 MLP 那一塊：

```
Dense：每一層一個 MLP，每個 token 都走完整個 MLP
       Llama、Gemma、Qwen 14B / 27B / 32B

MoE：  MLP 換成 N 個「專家」（各是獨立 MLP）+ 一個 router
       token 進來 → router 算分數 → 只挑 top-k（通常 2 或 8）→ 只經過這 k 個 → 加權合併
```

MoE 有兩個參數量：

| | 決定什麼 |
|---|---|
| 總參數（例如 35B） | VRAM 要多大 |
| 活躍參數（例如 3B） | 每 token 算多快 |

| | Dense 14B | MoE 30B-A3B |
|---|---|---|
| VRAM（4-bit） | ~9 GB | ~18 GB |
| 每 token 計算 / 頻寬 | 14B 的量 | 3B 的量 → 快很多 |
| 品質 | 14B 等級 | 通常介於 14B 到 30B 之間 |

**誤解**：「專家」不是領域專家。router 按 token 分，不按主題分，同一句話每個 token 可能走不同專家，哪個專家學到什麼是訓練自然形成的。

**對 fine-tune 的影響**：MoE 比 dense 麻煩（router 負載平衡、LoRA 掛哪些專家），第一次做建議 dense。

### 精度與量化

「4-bit / 8-bit / BF16」是**每一個權重用幾個 bit 存**，是數值精度，不是檔案格式。權重是幾十億個像 `0.0213`、`-0.187` 的小數，大多落在 -1 到 1 之間。

| 精度 | 每個數字 | 7B 模型 | 說明 |
|---|---|---|---|
| FP32 | 4 bytes | 28 GB | 訓練標準，推理不用 |
| FP16 / BF16 | 2 bytes | 14 GB | 發布時的原始精度 |
| FP8 / INT8 | 1 byte | 7 GB | 幾乎無損 |
| INT4 | 0.5 byte | 3.5 GB | 本機推理常用 |

**FP16 vs BF16**：都 16 bit，差在怎麼分。FP16 = 1 符號 + 5 指數 + 10 尾數（精度高、範圍窄）；BF16 = 1 符號 + 8 指數 + 7 尾數（範圍同 FP32、精度粗）。訓練時梯度範圍廣，FP16 易溢位，所以現代模型都用 BF16。推理時等價。

**整數量化怎麼做**：4 bit 只能表示 16 個整數，存不了小數。做法是一組權重共用一個小數 scale，每個權重只存「第幾格」：

```
一組 32 個權重，範圍 -0.4 到 0.5
scale = 0.06，zero point = 7
 0.0213 → 存 7  → 還原 (7-7)  × 0.06 =  0.00   （誤差 0.02）
-0.187  → 存 4  → 還原 (4-7)  × 0.06 = -0.18   （誤差 0.007）
 0.442  → 存 14 → 還原 (14-7) × 0.06 =  0.42   （誤差 0.02）
```

INT4 不是「4 bit 的小數」，是「4 bit 整數索引 + 每組一個 16 bit scale」。平均每權重約 4.5 bit，所以 Q4 模型不是剛好 BF16 的 1/4。推理時 kernel 讀 4-bit 整數、乘 scale 還原成 16-bit 再做矩陣乘法。

**品質損失**：8-bit 幾乎分不出來；4-bit benchmark 掉 1–3%，日常感覺不到；3-bit 以下明顯變笨。

**哪些東西可以量化**

| | 預設 | 可量化？ |
|---|---|---|
| 權重 | BF16 | 可，4/8-bit |
| KV cache | BF16 | 可，FP8 / Q8，多人時很省 |
| activation | BF16 | 通常不動 |
| 矩陣乘法本身 | BF16 或 FP8 | 4-bit 權重先還原再乘，算力還是 16-bit 的 |

### 為什麼同樣 4-bit 有那麼多種

「4-bit」只是名義值，實際格子怎麼切、切多細、哪些數字用更細的格子，每種方法不同。差異來自五個維度：

1. **group size**：一組幾個數字共用一個 scale。128 個一組開銷小但格距粗；32 個一組更貼身但 scale 佔空間
2. **格子是否等距**：INT4 等距；NF4（QLoRA 用）按常態分布分位數排，中間密兩端疏；IQ 系列用查表（codebook）
3. **混合精度**：Q4_K_M 把敏感的層（attention V、output、embedding）升到 5–6 bit；Q4_K_S 全部 4-bit
4. **有無資料校準**：Q4_K_M 純看權重分布；AWQ、GPTQ、imatrix 先跑校準文本，找出被用得重的權重特別保護。校準資料的語言分布會影響結果
5. **實際 bit/權重**：

| 名稱 | 實際 bit/權重 | 14B 模型 |
|---|---|---|
| IQ4_XS | ~4.25 | ~7.5 GB |
| AWQ 4-bit (g128) | ~4.3 | ~7.8 GB |
| Q4_0 | ~4.5 | ~8 GB |
| Q4_K_S | ~4.6 | ~8.1 GB |
| Q4_K_M | ~4.8 | ~8.5 GB |
| Q5_K_M | ~5.7 | ~10 GB |

**怎麼選**：Ollama / GGUF 用 **Q4_K_M**（公認甜蜜點），VRAM 寬裕上 Q5_K_M 或 Q6_K；vLLM 用 **AWQ** 或 **FP8**（40 系以上）。

### 檔案格式 vs 數值精度

兩層：

- **檔案格式（容器）**：HF safetensors、GGUF
- **數值精度（內容）**：BF16、Q4_K_M、AWQ

```
Ollama 的 qwen3:14b         = GGUF 容器 + Q4_K_M 精度
HuggingFace 的 Qwen3-14B    = safetensors 容器 + BF16 精度
HuggingFace 的 Qwen3-14B-AWQ = safetensors 容器 + AWQ 4-bit 精度
```

**HuggingFace 格式**：`transformers` 函式庫定義的目錄結構，幾乎所有模型發布的原生格式，訓練和 fine-tune 工具都直接吃它。

```
config.json                        ← 架構參數（層數、hidden size、attention 頭數）
model-00001-of-00004.safetensors   ← 權重，分片存
model.safetensors.index.json       ← 哪個張量在哪個分片
tokenizer.json / tokenizer_config.json
generation_config.json
```

**GGUF**：llama.cpp 自己定義的單檔格式，把權重重新量化、tokenizer、設定全打包成一個檔。優點一個檔搞定、CPU 也能跑；缺點要多一道轉換，fine-tune 的 LoRA 要先合併再轉。

NVIDIA 於 2026-09 宣布收購 Hugging Face（約 129 億美元，預計 2027 上半年交割），承諾平台維持開放、不要求用 NVIDIA 硬體。HF 格式本身是 transformers 定義的，收購前就是業界標準。

### VRAM 估算

```
權重 VRAM ≈ 參數量(B) × 每參數 bytes × 1.1（overhead）
  BF16：2 bytes    8-bit：1 byte    4-bit：0.55–0.6 bytes
再加 KV cache：單人 8K context 約 1–3 GB，多人要乘
```

| 模型 | 4-bit 權重 | 8-bit | BF16 | 單人實用（4-bit + KV） |
|---|---|---|---|---|
| Qwen3.5 4B | ~2.5 GB | ~4.5 GB | ~9 GB | 4–6 GB |
| Qwen3.5 9B | ~5.5 GB | ~10 GB | ~19 GB | 7–9 GB |
| Qwen3 14B | ~9 GB | ~16 GB | ~30 GB | 11–13 GB |
| Qwen3.6 / 3.5 27B | ~16 GB | ~29 GB | ~55 GB | 19–22 GB |
| Qwen3 32B | ~19 GB | ~34 GB | ~65 GB | 22–26 GB |
| Qwen3.6 35B-A3B | ~20 GB | ~37 GB | ~70 GB | 23–26 GB，速度像 3B |
| Qwen3 30B-A3B | ~18 GB | ~32 GB | ~61 GB | 21–24 GB，速度像 3B |

**對應硬體**

| 硬體 | 建議 |
|---|---|
| M1 16 GB | 9B 4-bit 剛好，14B 很擠 |
| M1 32 GB | 14B 舒服，27B 4-bit 可以但系統空間不多 |
| M1 64 GB | 27B / 32B 沒問題，MoE 速度很好 |
| 24 GB 卡，單人 | 27B 或 32B 4-bit 都放得下 |
| 24 GB 卡，40 人共用 | 14B 4-bit 最穩，留 10 GB 以上給 KV cache；或 35B-A3B 4-bit（聰明 + 快，但 KV 空間一樣緊） |
| 48 GB 卡 | 27B / 32B 8-bit，或 4-bit + 充裕 KV cache，40 人最從容 |

**fine-tune 另算**（QLoRA，底模 4-bit）：14B 約 20–24 GB，27B 約 36–40 GB。24 GB 卡 fine-tune 上限是 14B。

---

## 7. 三條路：純推理 / RAG / fine-tune

### 一句話原則

```
純推理：     函數不變，輸入不變
prompt / RAG：函數不變，改輸入        y = f_A(x + δ)
fine-tune：  輸入不變，改函數        y = (f_A + δ)(x)
```

### 情境一：純推理

就是第 5 節那張圖。權重從頭到尾不動，模型只知道訓練時學到的東西。問它公司黑話，它只能猜。

### 情境二：RAG，改的是輸入

兩個階段、兩個模型。**LLM 推理引擎從頭到尾不知道 RAG 的存在**，它只看到「有人送了一段比較長的 prompt」。

**離線階段：建索引（文件新增或更新時做，應用層自己跑，跟引擎啟動無關）**

```
內部文件 → 切成 chunk（例如每段 500 字）
   │
   ▼  每個 chunk 走一遍 embedding model：
[CPU] embedding model 的 tokenizer：文字 → token id
[GPU] embedding 查表 → Transformer ×N（通常 12–24 層，比 LLM 小）
[GPU] pooling：把整段輸出壓成一個向量（例如 1024 維）
   │              ↑ 取代了 LLM 的 lm_head + sampling
   ▼                沒有 token loop，一段文字一次 forward 就結束
向量資料庫：存 (向量, chunk 原文, metadata)
```

**線上階段：回答問題**

```
使用者問題
   │
   ├─ 1. 問題也走一遍 embedding model → 一個向量
   │
   ├─ 2. 向量庫算相似度（cosine），取最近的 top-k 個 chunk
   │
   └─ 3. 組 prompt：
        system: 你是公司內部助理，依據下列資料回答…
        [chunk 1 原文]
        [chunk 2 原文]
        [chunk 3 原文]
        user: 使用者問題
            │
            ▼
        走一次完整的 LLM 推理管線（跟情境一完全一樣）
        → tokenizer → prefill（此時才建 KV cache）→ decode → 結束釋放
```

關鍵修正：
- 向量只用來**找**文件，找到後塞進 prompt 的是**原文**，不是向量
- RAG 的上下文**不在 KV cache 裡預存**，每個請求都是現算的
- 為什麼吃 VRAM：每個請求的 prompt 從 50 token 變 1500 token，KV cache 和 prefill 時間都跟著放大，10 人同時用就是 30 倍

**兩套 tokenizer 同時存在**：embedding model 一套，LLM 一套，互不相干。chunk 大小用 embedding model 的 tokenizer 算，context 塞不塞得下用 LLM 的算，兩個數字常不一樣。

**「embedding」一詞的兩個意思**：
1. LLM 內部的 embedding 層：`[vocab × hidden]` 查表，第一層，輸入 token id 吐向量
2. Embedding model（RAG 用）：獨立模型，吃整段文字，輸出一個語意向量，不生成文字。它自己也有 tokenizer 和內部 embedding 層

### 情境三：fine-tune，改的是權重

不是推理，是訓練，管線多了「往回走」的半段。**Ollama / vLLM 沒有訓練 API**，訓練是另一套工具鏈，跟推理引擎共用的只有權重檔。

**前半段：forward，跟推理幾乎一樣**

```
訓練資料：(輸入, 期望輸出) 配對
  輸入：「解釋這段 DSL：spawn_rate = 0.3 * tier」
  期望輸出：「這表示生成率隨玩家層級線性增加…」
   │
   ▼
[CPU] tokenizer：把「輸入 + 期望輸出」整段轉 token id
[GPU] embedding → Transformer ×N → lm_head → logits
       ↑ 一次把整段餵進去（像 prefill），不做 sampling，用訓練框架跑
```

**後半段：backward，推理沒有的部分**

```
[GPU] loss：比對「模型預測的下一個 token 機率」和「真實的下一個 token」（cross-entropy）
   │
   ▼
[GPU] backward：loss 對每一層權重算梯度，從最後一層往前傳回第一層
   │
   ▼
[GPU] optimizer：依梯度微調權重（AdamW）
   │
   └──→ 下一批資料，重複幾千次
```

**產出與載入**

| 訓練方式 | 產出 | 大小 |
|---|---|---|
| 全量 fine-tune | 一整套新的 HF 權重目錄 | 跟原模型一樣大，14B 約 28 GB |
| LoRA | `adapter_model.safetensors` + `adapter_config.json` | 幾十到幾百 MB |

```
LoRA adapter
  ├─ vLLM：--enable-lora，掛 adapter 目錄，請求時指定用哪個
  │        原模型不動，可同時掛多個 adapter 給不同用途
  │
  └─ Ollama：
       a. PEFT 合併 adapter 進原權重 → llama.cpp convert 腳本轉 GGUF → Modelfile 建新模型
       b. adapter 單獨轉 GGUF → Modelfile 用 ADAPTER 指令掛上
```

**整體圖**

```
                 訓練世界                          推理世界
          ┌──────────────────┐             ┌──────────────────┐
HF 權重 → │ LLaMA-Factory 等  │ → adapter → │  vLLM / Ollama   │ → API → Open WebUI
          │ forward+backward  │   或新權重   │  只有 forward    │
          └──────────────────┘             └──────────────────┘
               需要訓練用 GPU                     需要推理用 GPU
             （同一張卡可以輪流用，不能同時）
```

### 三條路比較

| | 動到管線的哪裡 | 知識存在哪 | 更新成本 | 擅長 |
|---|---|---|---|---|
| 純推理 | 不動 | 原始權重 | 無 | 通用知識 |
| RAG | 輸入端（prompt 變長） | 向量庫 + 原文 | 改文件即生效 | 事實、文件、會變動的內容 |
| fine-tune | 權重（attention / MLP 矩陣） | 權重裡 | 重新訓練 | 風格、格式、語法、固定術語 |

### 為什麼不是所有事都用 RAG

兩個都能讓輸出變，但 δ 加在不同地方：

1. **RAG 的 δ 每次重付，fine-tune 付一次**：RAG 每個請求都要塞資料進 prompt，佔 context、拉長 prefill、吃 KV cache。40 人一起用差很多
2. **Context 裝不下兩千筆範例**：RAG 一次最多塞十幾筆；fine-tune 把兩千筆壓縮成權重改動，推理時全部生效
3. **RAG 給的是資訊，模型還得當場用對**：靠 in-context learning 現場歸納規則，14B 以下的小模型常套錯；fine-tune 直接把規則訓進處理過程
4. **有些東西不是資訊，塞不進 prompt**：「永遠只輸出 JSON」「語氣簡潔」「看到這種結構優先這樣解讀」是行為，小模型靠 system prompt 遵從度不穩
5. **RAG 依賴檢索命中**：沒找到就回到 f_A；fine-tune 的 δ 永遠在

RAG 贏的地方：資料會變（換文件即生效）、資料量大（權重裝不下）、要可追溯（能說根據哪份文件）、不想碰訓練。

**考試比喻**：RAG = 開書考，書帶進去翻，書再厚也行，但要翻得到、看得懂、現場套用。fine-tune = 讀過書再考，帶不進多少東西，但內化的東西快又穩。

**兩者互補**：fine-tune 讓模型「會讀 DSL」，RAG 讓它「知道今天的規格寫了什麼」。

---

## 8. fine-tune 深入

### 本質：用你的資料、透過梯度下降，改變模型計算的那個函數

```
model A：函數 f_A，輸入 token 序列 → 輸出分布
fine-tune 後：f_B = f_A + δ，δ 由你的資料決定
在資料覆蓋的範圍內，f_B 的輸出不同於 f_A
範圍外，f_B ≈ f_A
```

兩個精確化：
- 改的不是「怎麼看 token」（embedding 通常不動），是「怎麼處理 token 序列」（後面 32 層的轉換路徑變了）
- 不是 A 換成 B，是 A + 一點 δ。原本的 A 絕大部分都在，只在訓練資料覆蓋的區域被推了一下

### 「矩陣 = 一個轉換」

模型裡每個權重矩陣都是一個函數：吃一個向量、吐另一個向量。整個模型是幾百個這種轉換串起來。32 層 × 7 個矩陣（attention 4 個 + MLP 3 個）= 224 個矩陣，這全部就是「底模」。**訓練就是在決定這幾百個矩陣裡的數字。**

### LoRA：方法，不是套件

```
LoRA（Low-Rank Adaptation）  ← 2021 年微軟論文提出的技術 / 方法
PEFT                         ← HuggingFace 實作這個方法的函式庫
LLaMA-Factory / Unsloth      ← 包著 PEFT 的上層工具
QLoRA                        ← LoRA 變體：底模 4-bit 載入省 VRAM，旁路仍 BF16
```

### LoRA 的結構：掛在矩陣旁邊的旁路，不是新的一層

adapter 不是模型結構的一部分，模型原本沒有它。它是 LoRA 在訓練時外掛的：

```
原本：   x ──→ [ W ] ──→ y                        W: 4096×4096，1,677 萬個數字

LoRA：   x ──┬→ [ W ]（凍結）──────────┐
             │                          ├─(相加)→ y
             └→ [ A ]→[ B ]（可訓練）───┘
                A: 4096×r，B: r×4096，r 通常 8–64
                r=16 時只有 13 萬個數字，不到 1%
```

- A 先把 4096 維壓到 r 維，B 再放回 4096 維。中間只有 r 維，所以旁路能表達的改動是「低秩」的
- B 初始化為 0，訓練開始時旁路輸出是 0，模型行為跟原本完全一樣
- B·A 相乘後還是 4096×4096，所以 W 的**每一格都被影響**，只是影響的「形狀」受限。不是「局部」，是「低秩」
- 比喻：全量是允許重畫整張圖，LoRA 是只給 16 支特定顏色的筆描在原圖上

**為什麼低秩就夠**：經驗上 fine-tune 對權重的改動本來就是低秩的，你是在「微調」一個已經會很多事的模型，不是從零教。rank 8–64 通常抓到大部分效果。

**掛在哪些矩陣**（`target_modules`）：

```
第 1 層  ┌─ W_q    + (A₁B₁)
         ├─ W_k    + (A₂B₂)
         ├─ W_v    + (A₃B₃)
         ├─ W_o    + (A₄B₄)
         ├─ W_gate + (A₅B₅)
         ├─ W_up   + (A₆B₆)
         └─ W_down + (A₇B₇)
第 2 層  ┌─ W_q    + (A₈B₈)
         ...
第 32 層 └─ W_down + (A₂₂₄B₂₂₄)
```

- 最早論文只掛 W_q、W_v；QLoRA 論文實驗後發現全掛明顯更好，現在慣例是 7 個全掛
- 只掛 attention 能改「注意什麼」（格式、風格）；加上 MLP 才能有效改「怎麼轉換」（新語法規則、固定術語）。DSL 案例需要後者
- 通常不掛：embedding 表和 lm_head（除非要新增 token）、norm 層（參數極少）
- LLaMA-Factory 和 Unsloth 預設就是 7 個全掛、embedding 不動

**推理時兩種用法**：
- 不合併：每次算 W·x 順便算 B·A·x 加上去。多一點計算，但底模可共用給多個 adapter
- 合併：算出 W' = W + B·A 存成新 W'。沒額外開銷，但底模和 adapter 分不開

### 全量 vs LoRA

| | 改什麼 | 結果 | VRAM |
|---|---|---|---|
| 全量 fine-tune | 直接改 224 個 W 裡每個數字 | 一份全新權重 | 推理的 3–4 倍以上，24 GB 卡連 7B 都吃力 |
| LoRA | W 不動，訓 224 對 A、B | W 的有效值變 W + BA，用 adapter 檔表達這個差 | 24 GB 卡可訓 14B |

兩者最終效果都是「模型對同樣輸入給不同輸出」，差別在改動的自由度和成本。

### 4-bit 底模上的 fine-tune（QLoRA）

4-bit 的整數格子不可微分，梯度沒辦法微調一個整數索引，所以不能直接訓 4-bit 權重。QLoRA 繞過去：

```
底模（4-bit NF4，凍結）：格子、scale 全部不動
     +
LoRA adapter（BF16，可訓練）：全精度小矩陣

forward：4-bit 權重臨時還原成 BF16 → 乘 x → 加上 adapter 輸出
backward：梯度只流進 adapter，底模完全不更新
```

訓完的產物是 BF16 的 adapter，底模還是原來那份 4-bit 檔。

**格子什麼時候會變**：在「合併 + 重新量化」這個獨立步驟：

```
底模 4-bit → 還原成 BF16
+ adapter（BF16）
= 合併後的 BF16 新權重          ← 數值跟原來不同
→ 重新跑一次量化 → 新的 Q4_K_M  ← 格子和 scale 此時按新數值重算
```

所以格子確實會變，但不是「fine-tune 調整格子」，是「fine-tune 改了數值，重新量化時格子跟著重切」。合併後最好用校準資料重做 AWQ，不要沿用舊 scale。用 vLLM 掛 adapter 不合併，底模格子從頭到尾是原來的。

唯一直接訓量化格子的方法叫 QAT（quantization-aware training），是模型廠商發布前做的事。

### fine-tune 能改什麼、不能改什麼

**能力來自預訓練，fine-tune 只能調整怎麼用這些能力。** 預訓練用幾兆 token，fine-tune 用幾千到幾萬筆，差六七個數量級。比喻：預訓練是大學四年，fine-tune 是入職兩週的 on-boarding。

| 底模狀態 | fine-tune 後 |
|---|---|
| 預訓練沒看過程式碼，完全不會寫 | 還是不會，頂多學會表面格式，內容是錯的 |
| 會寫程式但輸出亂、不照指令、愛廢話 | 有效，變乾淨、照格式、符合你風格 |
| 會寫 Python，但不會你們的 DSL | 有效，前提是 DSL 的概念能對應到它已知的東西 |

**擅長**：格式（永遠輸出 JSON、不加開場白）、風格語氣、行為（遇到某類問題怎麼回）、術語對應、小型新語法（概念能對應已知語言）、把已有但不穩定的能力拉穩。

**做不到**：教大量新知識（那是 RAG）、創造推理能力（3B 推理差，tune 完還是差）、跨越尺寸（7B 變 32B）、教一套跟任何已知東西都不像的全新語言。

**為什麼 DSL 案例大概率可行**：DSL 幾乎都是「把已知概念用另一套符號寫」。模型預訓練看過 YAML、JSON、Lua、各種設定檔，你的 DSL 的變數、運算、條件、結構對它是舊概念新寫法。fine-tune 教的是對應關係。但 DSL 裡只有你們自己知道的語意（例如 `tier` 怎麼算），它只能從範例歸納，範例沒覆蓋的會瞎猜。

**實際期望值**：拿 Qwen 14B 這種本身會寫程式的模型，fine-tune 後得到「一個會寫程式、而且懂你們 DSL 和黑話的 14B」。不是更聰明，是更懂你們。

### Catastrophic forgetting

fine-tune 太猛（資料太窄、訓太久、learning rate 太高）會讓模型忘記原本會的事。訓完 DSL 解釋器發現連一般對話都變怪，就是這個。LoRA 比全量較不易發生，但還是要用一般題目測。

### fine-tune 與蒸餾

| | fine-tune | 蒸餾（distillation） |
|---|---|---|
| 是什麼 | 一種**訓練方法**：用資料更新權重 | 一種**訓練目標**：讓小模型模仿大模型 |
| 資料來源 | 任何 | 必須來自「老師」模型 |
| 老師存在嗎 | 不需要 | 一定有 |

蒸餾兩種：
- logit 蒸餾：學生學老師輸出的完整機率分布，需同時跑老師和學生，開源界少用
- 資料蒸餾：讓老師生成高品質問答，拿來 fine-tune 學生。實務上**就是一個 SFT fine-tune**，只是資料是老師產的

DSL 案例很可能走第二種：給 Claude 或 GPT 你的 DSL 語法說明，生成幾千筆「DSL ↔ 解釋」配對，人工抽查修正，再 LoRA fine-tune Qwen 14B。從方法看是 fine-tune，從目的看是蒸餾。學生上限就是老師，老師理解錯的地方學生一起錯，人工抽查不能省。

### 訓練工具

**上層工具**

| 工具 | 特色 | 適合 |
|---|---|---|
| LLaMA-Factory | Web UI 點一點就能訓；支援模型最多；中文社群大 | 第一次做，從這個開始 |
| Unsloth | 重寫關鍵 kernel，速度快 2 倍、VRAM 省一半；單卡為主 | 24 GB 卡想訓 14B，或想省時間 |
| Axolotl | YAML 設定檔驅動，多卡支援好，可重現性高 | 進 CI、跑多卡 |
| torchtune | PyTorch 官方，乾淨少依賴，支援模型較少 | 想看清楚底下在做什麼 |

**函式庫層**：transformers（模型定義）、PEFT（LoRA 實作）、TRL（訓練迴圈，SFT / DPO / GRPO）、bitsandbytes（4-bit 載入，QLoRA 基礎）、DeepSpeed / FSDP（多卡切分）

**Mac**：MLX-LM，M1 可直接 `mlx_lm.lora` 訓小模型，速度不如 N 卡但驗證流程夠用

**企業級**：NVIDIA NeMo，完整但重

**建議**：
1. 先在 M1 用 MLX-LM 跑一次 7B 的 LoRA，不求效果，只求把「資料格式 → 訓練 → 合併 → 載進 Ollama」走通
2. 真要訓就上 Linux 用 LLaMA-Factory（要 UI）或 Unsloth（要快）
3. 資料格式統一用 ShareGPT 或 Alpaca 格式的 JSONL，所有工具都吃
4. 工具差異遠小於訓練資料品質差異，選定就別換，時間花在整理 DSL 配對資料

### 兩個軸放一起

```
                dense              MoE
全量 fine-tune   最貴               更貴、更麻煩
LoRA            標準起手式          可以，較新、坑較多
```

坑最少的組合：**dense 14B + LoRA（QLoRA）**。

---

## 9. 黑話與 DSL 的策略

兩層，處理方式不同，都不用急著訓練。

### 第一層：黑話（公司內部術語）

純粹詞彙對應，最容易：
- 整理術語表：術語 → 白話解釋 → 一兩個使用範例。幾十到幾百條就很有用
- 量少放 system prompt，量多當 RAG 第一優先文件
- 效果立竿見影，模型本身會推理，只是不知道你們的詞

### 第二層：DSL（有語法、有語意）

先看複雜度：
- 接近既有語言（JSON / YAML 結構、類 Lua 腳本、表格式設定）→ 模型大概率半懂，給語法說明 + 10–20 個「DSL ↔ 自然語言」對照範例放 prompt，就能讀懂、解釋、甚至生成
- 自創語法、語意跟任何常見語言都不像 → prompt 撐不住長文時才需要 fine-tune

### 「新解釋器」要做什麼，先定義是哪一種

| 類型 | 難度 | 做法 |
|---|---|---|
| 1. 讀 DSL → 白話解釋這段在做什麼 | 最容易 | RAG + few-shot 就夠 |
| 2. 白話需求 → 生成 DSL | 中等 | fine-tune 效益最明顯（要學格式紀律） |
| 3. 檢查 DSL 有沒有錯 | 別只靠模型 | 接真正的 parser / validator，模型負責解釋錯誤訊息 |

### 著手順序

1. 寫術語表和 DSL 語法說明，這份文件本身就有價值，也是訓練資料的骨架
2. 階段一平台上線後把這兩份丟進去，讓大家用，蒐集「它哪裡解釋錯」
3. 錯的案例人工修正，累積成「DSL ↔ 解釋」配對資料
4. 累積到幾百筆、prompt 方式明顯到頂了，再上 LoRA

### 重點：接 parser 當工具

如果 DSL 有現成 parser，把它接進來當工具（function calling），讓模型先呼叫 parser 拿 AST 或錯誤訊息再解釋，比靠模型「看字猜」可靠很多。模型負責翻譯和解釋，正確性交給程式。

### 分工總表

| 需求 | 走哪邊 | 為什麼 |
|---|---|---|
| 黑話 → 白話對照表 | RAG（或 system prompt） | 是事實，會增加，量不大 |
| 規格文件的內容 | RAG | 量大、會改 |
| DSL 的讀法和解釋方式 | fine-tune | 是規則 / 能力，要穩定、每次生效、範例多到 prompt 裝不下 |
| 輸出格式和語氣 | fine-tune | 是行為，小模型靠 prompt 不穩 |

---

## 10. Harness 與 Claude 整合

### 建好之後怎麼用

手上只有一個「會回話的 API」，三種用法：

1. **Chatbot**：Open WebUI 就是最簡單的 harness（網頁、帳號、歷史、傳檔案）。40 人試用這就夠
2. **自己寫程式呼叫**：OpenAI 相容 API，任何語言用 OpenAI SDK 改 `base_url`。DSL 解釋器、RAG pipeline、Slack bot 都是這條路
3. **接進 agent harness**：harness 泛指「包在模型外面、決定它怎麼被用」的程式。Chatbot 只是最陽春的一種；會跑迴圈、呼叫工具、讀寫檔案的 agent 也是。Open WebUI 和 Claude Code 都是 harness，差在複雜度

### Harness 的三種意思

| 類型 | 例子 |
|---|---|
| Agent harness（會用工具、跑迴圈） | OpenHands、Aider、OpenCode、Goose、Continue（IDE 內），或用 LangGraph / Pydantic AI 自己寫 |
| Eval harness（評測） | lm-evaluation-harness（標準 benchmark）；promptfoo、DeepEval（自己出的測試題，比對 RAG 或 fine-tune 前後）；Ragas（RAG 專用） |
| Test harness（給 DSL 解釋器） | 自己寫：「DSL 輸入 → 預期解釋」配對 + 跑分腳本，promptfoo 可當骨架 |

小模型的 tool calling 穩定度差很多，agent harness 選錯會很痛。

### 「用 Claude 串這個 model」：看誰當腦

**方向 A：Claude Code 當 harness，地端模型當腦**
技術上可行：Claude Code 支援透過 `ANTHROPIC_BASE_URL` 指向 LLM gateway，中間放 LiteLLM 或 agentgateway 之類的轉接層，把 Anthropic 格式請求轉送到 vLLM / Ollama 的 OpenAI 相容後端。但官方文件的設計情境是「透過 gateway 走 Claude 模型」，接地端模型屬第三方玩法，不在 Anthropic 支援範圍。更實際的問題：Claude Code 重度依賴長 context 和穩定 tool calling，14B–32B 地端模型撐不住，體驗會很差。先別走這條。

**方向 B：Claude 當腦，地端模型當工具（建議）**
寫一個小的 MCP server，把地端模型（以及 DSL parser）包成工具，Claude Code 或 Claude Desktop 透過本機 MCP 呼叫：
- 推理、規劃、寫程式交給 Claude
- 碰到內部黑話、DSL、不能外流的資料，Claude 呼叫地端模型處理，資料不出區網
- 術語表、RAG 掛在地端那一側，Claude 只拿到結果

這是企業常見的混合架構，MCP server 幾十行就能寫。注意 claude.ai 網頁版連不到區網內的服務，要用 Claude Code 或 Desktop 這類跑在本機、能連本機 MCP 的客戶端。

### 建議順序

1. 先用 Open WebUI 讓大家聊，驗證模型選型和術語表
2. 同時寫個 MCP server 把地端模型和 DSL parser 包起來，自己用 Claude Code 接著玩
3. 兩邊都穩了，再決定 agent 層要不要擴大

---

## 11. 決策速查與常見陷阱

### 決策速查

| 問題 | 答案 |
|---|---|
| M1 上用什麼引擎 | Ollama（原生，Metal）或 MLX；不能 vLLM |
| M1 上 compose 能起什麼 | 只有 Open WebUI；模型不進容器 |
| 正式機器用什麼 OS | Linux + NVIDIA |
| Linux 上先用哪個引擎 | Ollama，省事；N 卡 + 多人 + 嫌慢三個湊齊再換 vLLM |
| 40 人、24 GB 卡選什麼模型 | Qwen 14B 4-bit（最穩）或 35B-A3B 4-bit（聰明 + 快） |
| GGUF 用哪種量化 | Q4_K_M；VRAM 寬裕上 Q5_K_M / Q6_K |
| vLLM 用哪種量化 | AWQ 或 FP8（40 系以上） |
| 公司內部知識怎麼餵 | 先 RAG |
| DSL 怎麼教 | 先術語表 + few-shot，累積資料後 LoRA |
| fine-tune 用什麼 | dense 14B + QLoRA，LLaMA-Factory 或 Unsloth |
| 24 GB 卡 fine-tune 上限 | 14B |
| 要三個模型同時服務 | VRAM 70–80 GB；或改成一個底模 + 多個 LoRA adapter |
| Claude 怎麼接 | Claude 當腦，地端模型包成 MCP 工具 |

### 常見陷阱

- 低估 VRAM：模型跑得動，context 一長就 OOM
- Ollama 的 `num_ctx` 預設只有 4K / 8K，超過會偷砍前面的 token，使用者覺得「它變笨了」
- 兩個 vLLM process 都用預設 0.9，第二個起不來
- 多個模型在 24 GB 卡上踢來踢去，每次切換重載十幾秒
- 沒做評測就上線，開源模型品質落差大，先用自家資料測
- RAG 的 chunk 大小用錯 tokenizer 算
- fine-tune 期待它學會不會的能力（SLM 不會寫程式，tune 完還是不會）
- fine-tune 太猛造成 catastrophic forgetting，訓完連一般對話都怪
- 合併 LoRA 後沿用舊的量化 scale，應該重新校準量化
- 蒸餾資料沒人工抽查，老師錯的地方學生一起錯
- 忘記更新與備份流程：模型檔、向量庫、Open WebUI 的 data volume 都要納入
- RAG 餵進去的文件權限沒想清楚，不是所有人都該看得到

### 心智模型總結

```
模型 = 幾百個矩陣串成的函數 f，權重是幾十億個小數
推理引擎 = 把 f 載進 VRAM、每步把多個序列疊成 batch 一起算的常駐 process
Ollama vs vLLM = 「哪裡都能跑」vs「榨乾一張 N 卡」，差在 scheduler / memory / kernel 三層
prompt / RAG = 不動 f，改輸入 x
fine-tune = 不動 x，改 f（全量改每個數字，LoRA 加低秩旁路）
fine-tune 改的是「怎麼用已有能力」，不是「創造新能力」
```
