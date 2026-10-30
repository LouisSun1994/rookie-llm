# Case 01 · 群聯 aiDAPTIV+：iGPU、SSD 補記憶體、以及它到底掛在哪一層

> 日期：2026-10-30
> 學習方式：從一則新聞（群聯潘董 CES / Computex 2026 推 aiDAPTIV+ 內顯平台）出發，反覆用自己的話陳述 → 校正，沿「iGPU 是什麼 → 為何 SSD 能補記憶體 → 這套軟體住哪一層 → 計算路 vs 存取路 → 綁定對象」往下挖
> 目前定位：🐣 本機驗證（M1 已跑 Ollama + Open WebUI，做過第一次 LoRA POC）；這條線是 ML infra 概念支線，不在主路線上
> 本篇範圍：只記 aiDAPTIV+ 這套的概念與定位。PyTorch = libtorch、Rust 引擎（Candle / Burn / tch-rs）、driver / kernel 三義的細節另記於 Rust 專案；此處只留定位 aiDAPTIV 所需的最小骨架
> 時效標記：模型尺寸、Intel 世代、群聯產品狀態皆為 2026-10 時（來源為該次對話的搜尋結果），會變
> 對應筆記：引擎 → SDK → driver → kernel 那條鏈、DMA、ioctl 邊界的細節在 [rookie-rs note 03](https://github.com/LouisSun1994/rookie-rs/blob/main/notes/03-gpu-stack-driver-kernel.md)
> 標「補強」的 blockquote 是整理進 repo 時加註的延伸，不在原本這條線的對話裡

---

## 0. 一句話版本

aiDAPTIV+ 是一個**住在推論引擎裡的記憶體配置器（allocator）+ 預取器（prefetcher）**——跑在 CPU、使用者空間、被框架主動呼叫——把 DRAM / VRAM 當固定大小的快取視窗、SSD 當後備，趁 GPU 算第 N 層時把第 N+1 層要的權重從 SSD 搬上桌。它用**容量換頻寬**，只在 MoE 模型上成立；它不碰 GPU 的計算路徑、不寫任何 driver；綁定的是群聯自家高耐久 SSD 與有整合的引擎，不是 Intel、不是 CUDA、不是 ISA。

---

## 1. 分層模型（骨架）

```
應用層        Open WebUI / 自寫程式 / MCP
──────────────────────────────────────────────────────────────────────────
推論引擎      libtorch(PyTorch) / llama.cpp(Ollama) / OpenVINO / vLLM
              ├─ 排 tensor、排每層運算順序、呼叫 GPU kernel
              └─ 記憶體配置器 ◀━━━ aiDAPTIVLink 換掉的只有這一格
                 ┌──────────────────────┴──────────────────────┐
          計算路徑（完全沒動）                        存取路徑（新增的那條）
          廠商 SDK：cuBLAS / libcudart ｜ oneDNN / oneMKL    檔案 API：pread / pwrite / io_uring
          user-mode driver：libcuda.so ｜ Level Zero          OS 檔案系統 → block layer
          ═══ 使用者空間 / 核心空間 邊界 ═══                   ═══════════════════════════
          kernel-mode driver：nvidia.ko ｜ xe.ko / i915.ko    nvme.ko（NVMe 標準，社群通用）
──────────────────────────────────────────────────────────────────────────
GPU 層        獨顯 → VRAM ｜ iGPU → 直接讀 DRAM               儲存：群聯 aiDAPTIVCache SSD
                            └────────── 兩條路在 DRAM 交會 ──────────┘
```

每個概念屬於哪一層：

| 概念 | 層 |
|---|---|
| iGPU、XVE / XMX、統一記憶體、記憶體頻寬、PCIe 通道 | GPU 層（硬體） |
| MoE、專家、router、active params、MXFP4 | 模型層 |
| aiDAPTIVLink（配置器 + 預取策略） | 推論引擎層的**一個元件** |
| aiDAPTIVCache（SLC 模式高耐久 SSD） | 儲存硬體；走 OS 標準 NVMe 路徑 |
| aiDAPTIVPro Suite（fork 的 PyTorch、接好的 Ollama、GUI） | 打包好的引擎環境，不是新引擎 |
| nvme.ko、檔案系統、block layer | OS；群聯不碰 |
| CUDA / Level Zero / 兩種 driver | GPU 計算路徑；群聯不碰 |

---

## 2. iGPU（內顯）是什麼

**核心規則**：iGPU 沒有自己的 VRAM，跟 CPU 共用系統 DRAM（統一記憶體，unified memory）。「把 DRAM 當 VRAM」不是行銷話術，是它的定義。Apple M 系列、AMD Strix Halo 同一套邏輯。

**關鍵細節**
- Intel iGPU 內有兩種運算單元：
  - **XVE**（向量引擎，傳統 ALU / EU，SIMD16 / 32，一 cycle 一組向量乘加）——任何世代都有
  - **XMX**（矩陣引擎，小型 systolic array，一 cycle 直接吃一塊 8×16 tile 做 INT8 / FP16 / BF16 乘加；與 NVIDIA Tensor Core 同類）——Arc 獨顯先有，筆電 iGPU 到 Lunar Lake / Arrow Lake-H（Series 2）才有
- 沒 XMX 也算得動（GEMM 拆開就是乘加，Xe-LP 有 DP4A 指令），但每 core 矩陣吞吐差 **8–16 倍**；core 數差沒那麼多，差在單元種類
- 群聯「指定規格」= **商業門檻，不是能力門檻**：沒 XMX 時 prefill 慢 50–100 倍，體驗爛到砸招牌，所以選擇不支援（OpenVINO 其實有 Xe-LP 的 kernel）
- 對 LLM decode 有利的兩點：推論是頻寬 / 容量瓶頸不是算力瓶頸，iGPU 算力夠；統一記憶體讓「SSD → DRAM 放好、GPU 直接讀」零拷貝

**對照表（2026-10 時，約略值）**

| 平台 | iGPU 架構 | 向量單元 | XMX | 約 INT8 | aiDAPTIV 名單 |
|---|---|---|---|---|---|
| i9-14900K（Raptor Lake，桌機） | UHD 770，Xe-LP | 32 EU | 無 | ~1–2 TOPS | ✗ |
| Core Ultra 9 285K（Arrow Lake，桌機） | Xe-LPG | 4 Xe / 64 EU | 無 | ~8 TOPS | ✗（桌機 iGPU 只是「能亮螢幕」規格） |
| Core Ultra Series 1 筆電（Meteor Lake） | Xe-LPG | — | 無 | — | ✗ |
| Core Ultra 9 285H / 288V（Series 2 筆電） | Xe-LPG+ / Xe2 | 8 Xe | 有 | ~67–77 TOPS | ✓ 可用下限 |
| Core Ultra Series 3（Panther Lake，Intel 18A，2026 初） | Xe3 | 12 Xe / 96 XVE | 有 | ~120 TOPS | ✓ 主推甜蜜點 |
| RTX 4090（對照） | Ada | 16384 CUDA | 512 Tensor | ~660 TOPS | ✓（獨顯版起家處） |

Panther Lake 剛好湊齊四件事：XMX、LPDDR5X-8533 ≈ 136 GB/s、PCIe Gen5 x4 給 SSD、32GB LPDDR5X 焊板上是常見配置（= 群聯 demo 基準）。

---

## 3. 為什麼 SSD 補記憶體能成立（以及何時不成立）

**核心規則**：decode 是記憶體綁定（memory-bound），prefill 是運算綁定（compute-bound）。SSD 補的是「容量」，付出的是「頻寬」，所以只有「每 token 要搬的 bytes 夠少」的模型撐得住 → **MoE**。

**關鍵細節——算給自己看**

```
GPT-OSS-120B（2026-10 時）：總參數 ~117B；每 token 啟用 ~5.1B（36 層 × 每層 128 專家選 4）
                           MXFP4 權重 ≈ 61 GB → 正常要 64 GB 以上的統一記憶體（群聯 blog 說 96 GB，含 KV / 餘裕）

decode 一個 token：
  運算量   ≈ 2 × 5.1B ≈ 10 GFLOP
  搬權重   ≈ 5.1B × 0.5 byte ≈ 2.5–3 GB
  算術強度 ≈ 10 GFLOP / 3 GB ≈ 3–4 FLOP/byte      ← 遠低於任何 GPU 的平衡點 → ALU 95% 在閒置

Panther Lake iGPU：~120 TOPS INT8；DRAM ≈ 136 GB/s
  算力上限 ≈ 120 TOPS / 10 GFLOP ≈ 數千 token/s（打三折仍上千）
  頻寬上限 ≈ 136 GB/s / 2.7 GB ≈ 50 token/s        ← 真正的天花板（全部在 DRAM 時）

prefill 一個 4k prompt：
  運算量 ≈ 4096 × 10 GFLOP ≈ 40 TFLOP，權重只讀一次
  Panther Lake ≈ 1–2 s；RTX 4090 ≈ 0.1 s           ← ALU 數量只在這裡有感；RAG / agent 長 context 的「延遲感」來源

SSD 加入後（PCIe Gen5 x4 ≈ 12–14 GB/s）：
  dense 70B Q4 ≈ 35 GB，每 token 全搬：35 / 13 ≈ 2.7 s/token → < 0.4 token/s   ← 不能用
  MoE 每 token 4 專家 ≈ 2–3 GB 全未命中：13 / 2.5 ≈ 5 token/s
      每層 4 專家 ≈ 50–70 MB：SSD 搬 ≈ 5 ms，DRAM 讀 ≈ 0.5 ms → 全未命中 GPU 會等
      熱門專家留 DRAM 命中後 → 個位數到十幾 token/s                              ← 勉強能用
```

**成立條件（整套東西的核心節奏）**

```
時間 ──────────────────────────────────────────▶
GPU：  [算第 N 層   ][算第 N+1 層 ][算第 N+2 層 ]
SSD：  [搬第 N+1 層 ][搬第 N+2 層 ][搬第 N+3 層 ]

條件：T_搬(第 N+1 層所需 bytes ÷ 有效 SSD 頻寬) ≤ T_算(第 N 層)  → GPU 永遠不等，SSD 的慢被藏掉
       dense：每層都要搬，且搬 ≫ 算 → GPU 每層空窗
       MoE  ：只搬 4 個專家，搬得完 → 成立；命中率決定最終 token/s
```

- **為什麼不直接用 OS swap / mmap**：OS 分頁是**被動**的——碰到不在記憶體的頁 → page fault → 停下等 SSD → 每次 4 KB → 不知道下一步要什麼。aiDAPTIV 是**主動**的 application-aware prefetch：知道計算順序（算 N 時拉 N+1）、知道 router 結果（只拉那 4 個專家）→ I/O 與運算重疊。llama.cpp 的 mmap 屬前者
- **KV cache 同理**：context 超過預算時舊 KV block 寫去 SSD；同一份文件再被問（RAG 常見）直接讀回、不重算 prefill（DGX Spark demo：首字延遲 40 s → 9 s，2026-01 CES）
- **概念不新，新的是產品化**：DeepSpeed ZeRO-Infinity（2021）、FlexGen、Apple「LLM in a flash」、llama.cpp mmap 都是 offload 到 SSD。群聯的差異化：自家控制晶片 → 高耐久 SLC 快取盤；中介軟體做專家預取與快取；綁 Intel / OEM 出貨；時機（DRAM 貴且缺，「32GB 筆電能賣 AI PC」對品牌廠有吸引力）
- 「解決記憶體牆」說大了 → 準確是「**用 SSD 補容量不足，代價是速度，只在 MoE 成立**」。它賭的是「邊緣模型都走 MoE」這個趨勢；demo 清一色 MoE 不是巧合

> **補強**
> - 「只在 MoE 成立」說窄了。成立條件是上面那條 T_搬 ≤ T_算，MoE 是靠「縮小每 token 要搬的 bytes」來滿足它；另一條路是「放大每 byte 的運算量」——prefill、多條序列一起 batch decode、以及訓練（forward + backward ≈ 3 倍運算、optimizer state 每步只碰一次）都會把算術強度拉高幾十到幾百倍，dense 模型就撐得住。這正是 2024 年第一版能用 4×4090 + SSD 微調 **dense** 70B 的原因（§6 自己有寫），也是 FlexGen 用 offload 換 throughput 的原理。精確說法：**單條序列的互動式 decode 下，只有 MoE 成立**；換成訓練或高 batch，dense 也成立，只是看的是 throughput 不是延遲。
> - prefill 在 offload 下不只是「運算綁定」：4k prompt 的 4096 個 token 各選 4 個專家，每層 128 個專家幾乎全會被點到，等於整份 61 GB 權重都要過一遍。DRAM 只有 32 GB 時至少 30 GB 得從 SSD 來：30 / 13 ≈ 2.3 s 的下限，比 XMX 算的 1–2 s 還長。預測差別：iGPU + SSD 的首字延遲（TTFT）卡在 SSD 頻寬，不在矩陣單元；這也是為什麼群聯的 demo 拿 KV cache 回存（省掉重算 prefill）當賣點，而不是拿 TTFT 當賣點。
> - 「被動分頁」在 GPU 側也有對應物：CUDA Unified Memory（`cudaMallocManaged`）超額配置時，GPU 碰到不在 VRAM 的頁會觸發 GPU page fault，由 driver 把頁從 DRAM 遷過來——跟 OS swap 一樣是「碰到才搬、顆粒小、不知下一步」，大模型 decode 下一樣慢得不能用。所以「主動 vs 被動」這條分界，在 OS 層（swap / mmap）和 driver 層（UVM）各有一個被動版本，aiDAPTIV 選的是在引擎層做主動版本。mmap 的「每次 4 KB、不知下一步」另有細節（readahead、`madvise`），見 rookie-rs note 03 §8 的補強。

---

## 4. aiDAPTIV+ 是什麼：三件組與它在堆疊裡的位置

**核心規則**：它比推論引擎**小一層**——是插進引擎的一個元件，不是引擎本身。比喻：推論引擎 = 引擎本體，aiDAPTIV = 渦輪增壓器——裝在引擎上改變引擎行為，沒它引擎照跑只是馬力小，但它自己不是引擎。群聯賣的是渦輪 + 一台裝好渦輪的車，不自己造引擎。

| 元件 | 層 | 做什麼 |
|---|---|---|
| aiDAPTIVLink | 引擎內元件 | 配置器 + 預取器：接手「權重與 KV 放哪、什麼時候搬」這一件事；不算矩陣、不排 kernel、不載模型格式 |
| aiDAPTIVCache | 儲存硬體 | SLC 模式、高 DWPD 的 SSD（權重 / KV 反覆讀寫會殺 TLC）；差異化藏在盤內韌體，對 host 透明 |
| aiDAPTIVPro Suite | 打包環境 | fork 過的 PyTorch、接好的 Ollama、GUI 工具；讓它看起來像完整引擎，實際算的仍是別人的引擎 |

**關鍵細節——它怎麼接進引擎（不是攔截，是被呼叫）**
- 框架設計時就留了配置器插槽：PyTorch `CUDAPluggableAllocator`、OpenVINO plugin 架構、llama.cpp backend 介面。aiDAPTIVLink 是實作這些介面的 C / C++ `.so`，引擎啟動時載入；流程是「引擎說我要配記憶體 → 呼叫 aiDAPTIV 的函式」，不是 aiDAPTIV 去攔引擎
- 沒有任何使用者空間程式能偷偷把別人的 `malloc` 導去 SSD——那要 kernel 才做得到
- 載模型時：權重寫去 SSD，只回引擎一個 handle，記憶體先不放。執行到第 N 層前（forward hook）呼叫「確保第 N 層在記憶體裡」→ 配置器從 SSD 讀進預留暫存區 → 算完標記可淘汰。MoE 更細：router 算完知道哪 4 個專家才拉那 4 個，其他 124 個不碰
- GPU 物理上只能算自己記憶體裡的東西。VRAM / DRAM 變成**固定大小的快取視窗**，權重在 SSD 與視窗間輪轉，GPU 永遠只看到視窗裡的東西
- **誰持有 / 誰觸發 / 誰執行**：帳本（什麼在哪）由配置器持有（CPU、引擎 process 內）；觸發是引擎的 forward hook / router 結果；搬運由 `nvme.ko` + SSD 控制器 DMA 執行；運算由 GPU 執行。GPU 從頭到尾不知道管家存在

> **補強**
> - 「換掉的只有配置器這一格」（§1 圖）說窄了。配置器插槽（`CUDAPluggableAllocator`）看到的只是「給我 N bytes」「還你這塊」，它不知道這塊是第 N+1 層的專家 e41——而 §3 整套預取邏輯都建立在「知道計算順序、知道 router 結果」上。那些資訊只在框架的 module / parameter 層有：forward pre-hook、parameter 換入換出（ZeRO-Infinity、HF Accelerate 的 disk offload 都是在這層做）。所以 aiDAPTIVLink 至少要插兩層：**參數層的 hook 決定搬什麼、何時搬**，配置器層頂多管暫存區的池子。這也解釋了為什麼群聯出貨的是 fork 過的 PyTorch 而不只是一個 `.so` plugin——allocator 介面不夠用。結論不變：仍在引擎內、使用者空間、被框架呼叫；只是「一格」要改成「引擎裡的兩個插槽」。
> - 「權重 / KV 反覆讀寫會殺 TLC」要拆開：NAND 的磨損來自**寫入 / 抹除**，讀取幾乎不耗壽命。推論時權重是唯讀的，從 SSD 串流一百萬次也不會寫壞盤；真正在寫的是 KV cache 回存和訓練時的 optimizer state / 梯度換出（2024 版的主場）。預測差別：純 iGPU 推論場景，一般 TLC 盤的壽命不是問題，差異會出現在**持續隨機讀的吞吐與延遲 QoS**（SLC 模式在這點也有優勢，但那是效能不是耐久）；耐久度這道護城河主要護的是訓練與 KV 寫回的用例。
> - 「使用者空間程式無法偷改別人的 malloc」限跨 process；同一個 process 內 `LD_PRELOAD` 的 symbol interposition 做得到（jemalloc 就是這樣部署）。「留插槽」是乾淨門，不是唯一門。細節見 rookie-rs note 03 §8 補強。

---

## 5. 計算路 vs 存取路：為什麼一個 driver 都不用寫

**核心規則**：GPU 側**沒有跨廠商的指令標準** → 每家廠商必須自己出 user-mode + kernel-mode driver，引擎只能透過廠商 API 一路往下。SSD 側**有標準（NVMe）** → Linux `nvme.ko` 是社群通用 driver，群聯 / 三星 / WD 的盤都用同一顆。所以群聯什麼 driver 都不用提供，GPU 那條路一根毛都沒動。

```
計算路（沒變）：引擎 ──▶ CUDA API / oneDNN ──▶ libcuda.so / Level Zero ──▶ nvidia.ko / xe.ko ──▶ GPU
存取路（新增）：aiDAPTIV 配置器 ──▶ 檔案 API（pread / pwrite、io_uring、O_DIRECT）──▶ OS 檔案系統 / block layer ──▶ nvme.ko ──▶ 群聯 SSD
交會點：DRAM —— 右邊把權重從 SSD 放進 DRAM；左邊再從 DRAM 搬去 VRAM（獨顯）或直接讀（iGPU）
```

**關鍵細節**
- 不是「把 CUDA 換成 NVMe API」，是「**加上**一條檔案 API 的路」。應用程式連 NVMe 這個字都不用知道（SPDK 那種使用者空間直驅 NVMe 是極端低延遲儲存系統在用的，這裡不需要）
- API 層級上它做的事 = 把一塊記憶體寫成檔案、再讀回來（與 `np.save` / `np.load` 同類系統呼叫），只是做得講究：大塊對齊的直接 I/O、繞過 page cache、非同步佇列讓讀寫與 GPU 運算重疊
- 它只碰兩個全世界最標準化的介面：**引擎的配置器插槽** + **OS 的檔案 I/O**。技術含量在兩頭——預取策略（軟體，懂模型結構）、SSD 韌體（硬體，耐久度）；中間 driver 層完全不碰 → 這就是它能跨 NVIDIA / Intel / AMD、快速接 PyTorch / llama.cpp / OpenVINO 的原因
- 回程的資料與通知是兩條路（對理解時間軸重要）：GPU / SSD 算完或讀完，**資料**由 DMA 直接落在 VRAM 或釘住的 DRAM 頁，不經 CPU；**完成通知**走中斷 → OS kernel → driver → 更新一個 fence / 旗標。配置器只看旗標，不「解釋反饋」

---

## 6. 綁定對象與邊界

**核心規則**：運算側盡量支援所有人，儲存側只能用自家盤。**綁定對象是群聯的 SSD，不是 Intel。**

| 層 | 綁什麼 | 2026-10 時 |
|---|---|---|
| 硬體 | aiDAPTIVCache SSD（耐久度 + 韌體配合是前提，也是收錢處） | 必須 |
| 軟體 | 有整合的引擎：PyTorch（群聯 fork）、OpenVINO、llama.cpp / Ollama、LLMWare | 沒整合的接不了：自寫 CUDA / Vulkan 推論程式、Apple MLX、Candle / Burn（Rust）、未接的商業引擎；唯一間接路徑是 tch-rs / llama.cpp binding（底下仍是有接的引擎） |
| GPU 廠商 | 不綁 | CUDA 是起家處（2024 第一版：4×RTX 4090 + 2 顆 Pascari AI100 SSD 微調 70B，把 optimizer state / 權重分片換到 SSD）；CES 2026 同時 demo AMD / Intel / NVIDIA（含 DGX Spark 跑 Llama 3.1 8B） |
| ISA | 不綁 | DGX Spark 的 Grace 是 Arm；聯發科手機 UFS 的 POC 也是 Arm。中介軟體是 host 端使用者空間軟體，Linux 上編譯一次就過 |

**Mac 不行的三個原因（不是 OS、不是協議）**
1. **沒有內部 PCIe 插槽**：NAND 焊死 + Apple 自家控制器，群聯賣不了盤進去；只剩 Thunderbolt 外接。Thunderbolt **本身就是 PCIe 穿隧（tunnel）**，OS 仍用 NVMe driver 認它、協議沒變，只是同樣協議塞進窄管子：TB4 ≈ PCIe Gen3 x4（~3 GB/s）、TB5 ≈ Gen4 x4（~6 GB/s）vs 內部 Gen5 x4（~14 GB/s）；多一跳控制器的幾微秒延遲相對 NAND 幾十微秒不算什麼，頻寬那刀是實的
2. **MLX / Metal 沒開配置器鉤子**：統一記憶體由 Metal 統管，第三方沒有 API 可以插進去說「這塊 tensor 你先別載」。llama.cpp 在 Mac 上 mmap 串流權重是繞過 Apple 自己做，不是 Apple 給的介面
3. **沒有商業動機**：Apple 自己賣大記憶體（統一記憶體可到 512GB），「DRAM 堆大」與「DRAM 小 + SSD 補」是直接競爭的路線。前兩個技術上有繞法，第三個沒有

**客群與商業框架**：2024 打中小企業 / 銀行 / 醫院（地端、資料不出門、幾十萬台幣取代 H100 機）→ 2026 iGPU 這步真正的客群是 **PC 品牌廠**——讓 aiDAPTIV 成為 AI PC 標配料件、每台筆電多買一顆它的 SSD。企業地端是 beachhead，消費端 NAND 出貨量才是本業延伸。潘董自己的框架是混合雲：簡單推論地端、複雜回雲端。

**對本專案的定位**：正式機 Linux + NVIDIA 若 VRAM 放得下目標模型（SLM 路線）就與這套無關；它解的是「買不起大記憶體但想跑大 MoE」這一格。同類的開源路徑是 llama.cpp mmap（被動）與 FlexGen / ZeRO-Infinity（offload）。若哪天 llama.cpp / Ollama 自己把 MoE 專家預取做好，aiDAPTIVLink 的軟體價值會被稀釋，剩下盤的耐久度優勢——群聯也知道，所以往 OEM 綁料件走。

> **補強**：「若哪天 llama.cpp / Ollama 自己把 MoE 專家預取做好」——一半已經發生。llama.cpp 2025 年起有 `--n-cpu-moe N` / `--cpu-moe`（底層是 `--override-tensor` 把專家 tensor 指定放 CPU 端 DRAM），這是 MoE-aware 的**放置**：attention 與熱路徑在 GPU、專家留 DRAM，跑 gpt-oss-120b 這類模型時是常見配置。還沒有的是「專家在 SSD、依 router 結果預取」這段串流邏輯。所以 aiDAPTIVLink 的軟體護城河比上面寫的更薄一層：它贏的是 DRAM 也放不下時的那一段，不是「MoE-aware」本身。（2026-10 之後的狀態我沒有驗證。）

---

## 7. 獨顯 + SSD vs iGPU + SSD 路線對照

```
獨顯 + SSD（aiDAPTIV 原始版，2024）
         ┌──────── CPU ────────┐
         │  引擎 + aiDAPTIV     │
         └──┬────────────┬──────┘
   ①檔案API │            │ ③cudaMemcpyAsync
            ▼            ▼
   SSD ──②DMA──▶ DRAM ──④PCIe x16──▶ VRAM ──▶ GPU 運算
 (Gen5 x4)      (純中繼站)            (~1 TB/s，獨立)

iGPU + SSD（Intel 版，2026）
         ┌──── CPU + Xe GPU（同一顆晶片）────┐
         │  引擎 / OpenVINO + aiDAPTIV  Xe ◀┐ │
         └──┬─────────────────────────────┼─┘
   ①檔案API │                             │ ③直接讀（zero-copy）
            ▼                             │
   SSD ──②DMA──▶ DRAM（統一記憶體，~136 GB/s）┘
```

| | 獨顯 + SSD | iGPU + SSD |
|---|---|---|
| 權重落地次數 | 2（DRAM 中繼 + VRAM） | 1（DRAM） |
| PCIe 段數 | 2（SSD→DRAM Gen5 x4、DRAM→VRAM x16） | 1（SSD→DRAM） |
| 記憶體佔用 | DRAM 暫存區 + VRAM 各一份 | 一份 |
| GPU 運算時的頻寬 | VRAM 獨立 ~1 TB/s，搬運不干擾 | 與 CPU 存取、SSD 寫入共用 ~136 GB/s，搬運會搶 |
| CPU 的功 | tokenizer / sampling / kernel launch / 記憶體管理 + aiDAPTIV 預取排程 | **同樣的功**；少一次 host-to-device copy |
| CPU 影響吞吐的方式 | 幾乎沒有（頻寬分離） | 當「搶匯流排的鄰居」，不是額外運算負擔 |
| 淨效果 | 快，但吃兩份記憶體 | 省記憶體、路徑短，搶頻寬小幅負面，不是瓶頸 |
| 沒有 aiDAPTIV 時 | 一開始全 cudaMemcpy 進 VRAM，之後 SSD / DRAM 不參與 | 全 mmap / 讀進 DRAM 就不動 |

aiDAPTIV 做的就是把「一次性載入」變成「持續輪轉」，用時間換空間。

---

## 思維鏈：完整時間軸

**情境**：decode 一個 token；第 N 層是 MoE 層；router 選出的 4 個專家中 2 個已在 DRAM 快取、2 個不在。獨顯版；iGPU 版差異在備註。
**注意**：aiDAPTIV 內部實作非公開；以下是依其公開說法（forward hook、router 後拉專家、熱門專家快取、I/O 與運算重疊）推出的合理流程。

| t | CPU / GPU | 元件 | 處理哪些 | 發生什麼 |
|---|---|---|---|---|
| t0 | GPU | GPU kernel（CUDA） | 第 N 層 attention | 讀 VRAM 視窗內的 Wq / Wk / Wv / Wo + KV cache，算完 |
| t1 | GPU | GPU kernel | 第 N 層 router | 輸出 top-4 專家 id：{e3, e17, e41, e90} |
| t2 | CPU | 引擎 forward hook → aiDAPTIVLink 配置器 | 查帳本 | e3、e17 已在 DRAM 快取（熱門）；e41、e90 未命中 → 對這 2 個發非同步 pread（io_uring、O_DIRECT、大塊對齊） |
| t3 | CPU（短暫切核心態） | OS 檔案系統 → block layer → nvme.ko | 2 個讀請求 | 建 NVMe 命令與 DMA 描述符、敲門鈴通知 SSD 控制器；CPU 回使用者態，不等 |
| t4 | SSD 控制器 | aiDAPTIVCache 韌體 | e41、e90 權重 | 從 NAND 讀出，DMA **直接寫進 DRAM 釘住的緩衝區**；CPU 不搬 |
| t4' | GPU（與 t3–t4 同時） | GPU kernel | e3、e17 的 FFN | 已在視窗 → 先算；這段重疊就是「SSD 的慢被藏起來」 |
| t5 | CPU | nvme.ko 接中斷 → 配置器 | 完成通知 | 更新「e41、e90 已在 DRAM」旗標（資料早已靠 DMA 在記憶體裡，中斷只是旗標） |
| t6 | CPU | 配置器 → cudaMemcpyAsync | e41、e90 | DRAM → PCIe x16 → VRAM 視窗（**iGPU 版沒有這步**，Xe 直接讀 DRAM 位址） |
| t7 | GPU | GPU kernel | e41、e90 的 FFN | 等 fence 到位後算；若 t4–t6 比 t4' 長 → GPU 空窗（dense 模型每一層都卡在這裡） |
| t8 | GPU | GPU kernel | 第 N 層輸出 | 4 專家輸出加權合併、殘差，進第 N+1 層 |
| t9 | CPU | 配置器 | 淘汰與預取 | 標記 e41、e90 可淘汰（或依熱度留下）；依預取策略開始拉第 N+1 層的熱門專家 |
| t10 | GPU | GPU kernel | 第 N+1 層 attention | 回到 t0；36 層跑完 → lm_head → sampling（CPU）→ 吐出一個 token |

**iGPU 版差異**：t6 消失（少一跳、少一份記憶體）；但 t4 的 DMA 寫入與 t4' / t7 的 GPU 讀取搶同一條 ~136 GB/s 匯流排。
**KV cache**：context 超過預算時，配置器在 t9 類似時機把舊 KV block 走同一條存取路寫去 SSD；再被問到時讀回不重算 prefill。

> **補強**：t4 → t6 的「SSD 先 DMA 進 DRAM、再由 GPU copy engine DMA 進 VRAM」是獨顯的預設路徑，不是硬體限制。GPUDirect Storage（cuFile + `nvidia-fs.ko`）讓 SSD 控制器直接 DMA 進 VRAM，t6 就消失，獨顯也變成落地一次；需要 driver 與檔案系統配合，多在資料中心卡上。群聯是否用它我不知道。另一個細節：不論哪條路，t4 的 DRAM 緩衝區都必須是釘住的（pinned）——SSD 的 DMA 進來需要固定的實體頁，t6 的 `cudaMemcpyAsync` 要真的非同步也需要 pinned host memory；同一塊釘住的緩衝區可以同時當 SSD DMA 的終點與 GPU copy engine 的起點。DMA 本身是什麼，見 rookie-rs note 03 §9 的補充。

---

## 陷阱清單（這條線上實際撞到的）

1. ❌ iGPU 沒有運算元，「把 DRAM 當 VRAM」不科學 → ✅ 有 XVE（向量）也有 XMX（矩陣），只是算力比獨顯小；共用 DRAM 正是它的定義 → 為什麼：decode 是頻寬綁定，iGPU 算力本就夠；差別在記憶體架構不在 ALU
2. ❌ 「跑 120B 要 64GB」是指 VRAM → ✅ 指統一記憶體，iGPU 沒有 VRAM；MXFP4 權重 ~61 GB，正常要 64–96 GB；群聯的賣點是 32GB + SSD 也能跑 → 為什麼：iGPU 的 VRAM 就是 DRAM
3. ❌ SSD 當記憶體延遲一定不能用 → ✅ dense 確實不能用（70B Q4 < 0.4 token/s）；MoE 靠 prefetch 與運算重疊撐到個位數～十幾 token/s → 為什麼：T_搬 ≤ T_算 才不等；「每 token 要搬多少 bytes」是決定量，不是 SSD 延遲本身
4. ❌ CPU 要很強才撐得起 iGPU 推論；i9 有很多核心應該可以 → ✅ 門檻在 iGPU 的 XMX、DRAM 頻寬、PCIe Gen5 x4；CPU 只跑輕量排程；i9 的 UHD 770 連 XMX 都沒有 → 為什麼：矩陣運算在 iGPU，CPU 做的是管家工作；「上游運算顆粒度」要看 GPU 的單元種類，不看 CPU
5. ❌ iGPU 方案多了一段 CPU 的功 → ✅ 功一樣（tokenizer / sampling / kernel launch / 記憶體管理 + 預取排程），獨顯版 2024 就有；iGPU 下只差「少一跳」與「搶頻寬」 → 為什麼：記憶體架構變了，工作沒變；CPU 從「旁觀者」變「搶匯流排的鄰居」
6. ❌ 瓶頸是 ALU 數量 → ✅ 瓶頸是「有沒有矩陣單元」這個 0 / 1；有了它 decode 回到頻寬問題，沒有它連 prefill 都過不了 → 為什麼：XMX vs XVE 矩陣吞吐差 8–16 倍，core 數差遠沒這麼多
7. ❌ 以往 iGPU 不支援張量計算 → ✅ 算得動（向量 ALU + DP4A），但慢一個數量級；「指定規格」是商業門檻 → 為什麼：體驗爛會砸招牌，群聯選擇不支援 Xe-LP，不是做不到
8. ❌ 這套綁 Intel 生態系，CUDA / MLX 都不能用 → ✅ 綁的是群聯自家 SSD；CUDA 是起家處、Intel iGPU 是最新加入；MLX 不行的原因是插槽 / API / 動機，不是軟體相容 → 為什麼：中介軟體活在廠商中立的引擎層，往下接誰都行
9. ❌ 這平台在「驅動 SSD」、攔截 GPU driver 或儲存 kernel 的訊號 → ✅ 使用者空間、**被框架主動呼叫**的配置器；SSD 仍由 OS 的 `nvme.ko` 驅動 → 為什麼：使用者空間程式無法偷改別人的 `malloc`，那要 kernel；框架本來就留了配置器插槽
10. ❌ aiDAPTIV 是一個推論引擎 → ✅ 插進引擎的元件（渦輪 vs 引擎）；Pro Suite 打包得像完整引擎，但算的仍是 PyTorch / llama.cpp / OpenVINO → 為什麼：它不算矩陣、不排 kernel、不載模型格式
11. ❌ 套上它後，`CUDA API → libcuda.so → nvidia.ko → GPU` 這條要改成 call NVMe API → ✅ 計算路一字不改；**新增**一條檔案 API 路（pread / pwrite），兩路在 DRAM 交會 → 為什麼：GPU 無標準只能走廠商 API；儲存有 NVMe 標準，OS 已包好，應用程式只 call 檔案 API
12. ❌ 群聯要提供 SSD 的 user-mode / kernel-mode driver，並把 libtorch 接硬體 API 那段換成 SSD API → ✅ 一個 driver 都不用寫；`nvme.ko` 通用；差異化藏在盤內韌體 → 為什麼：NVMe 是跨廠商標準，GPU 才沒有
13. ❌ Mac 不行是因為 OS 不同 / 外接協議是 Thunderbolt 不是 PCIe → ✅ Thunderbolt 就是 PCIe 穿隧，OS 仍用 NVMe driver；真正卡住：無內部插槽（頻寬砍半以上）、Metal 無配置器鉤子、無商業動機 → 為什麼：同樣協議塞進窄管子；前兩者有繞法，第三者沒有
14. ❌ aiDAPTIV 是給 GPU / iGPU 用的 → ✅ 給引擎用，跑在 CPU、住在引擎 process 內；GPU 不能「使用軟體」，只執行 kernel 與讀寫看得到的記憶體 → 為什麼：GPU 回頭時東西已在桌上，它從頭到尾沒看見管家
15. ❌ 用 OS swap / mmap 效果一樣 → ✅ OS 分頁被動（page fault、4 KB、不知下一步）；aiDAPTIV 主動（知道計算順序與 router 結果）→ I/O 與運算重疊 → 為什麼：application-aware prefetch 這四個字是整套的價值所在，剩下是工程
16. ❌ 它「解決記憶體牆」 → ✅ 容量換頻寬，只在 MoE 成立，token/s 一定比全塞 DRAM 慢 → 為什麼：賭的是邊緣模型走 MoE 的趨勢；若趨勢不成立，價值大幅縮水
17. ❌ user-mode driver 把指令「轉成廠商 binary」，回程還要「解釋反饋」 → ✅ GPU kernel 機器碼早編好；user-mode driver 把「跑這個 kernel」包成命令封包推進佇列；結果資料由 GPU / SSD 直接 DMA 落在記憶體，中斷只更新 fence 旗標 → 為什麼：資料與通知是分開的兩條路；CPU 通常批次丟命令、在需要結果處才等 fence，否則 GPU 會餓死

> **補強**（對應第 3、9、10、16 條）：第 3、16 條的「只在 MoE 成立」限單條序列的互動式 decode，訓練與高 batch 下 dense 也成立（§3 補強）；第 9 條的「無法偷改別人的 malloc」限跨 process（§4 補強）；第 10 條「插進引擎的元件」要插的是參數層 hook + 配置器兩個位置，不只配置器一格（§4 補強）。

---

## 心智模型總結

```
iGPU = CPU 的室友：共用 DRAM 當 VRAM。能不能跑 LLM 看「有沒有 XMX」，跑多快看「DRAM 頻寬」
decode 頻寬綁定、prefill 運算綁定 → ALU 只在 prefill 有感；decode 快慢 = 每 token 搬幾 GB ÷ 頻寬
SSD 補記憶體 = 容量換頻寬：T_搬(N+1) ≤ T_算(N) 才成立 → dense 死、MoE 活、命中率決定 token/s
aiDAPTIVLink = 引擎裡的配置器 + 預取器（CPU、使用者空間、被框架呼叫）；不是引擎、不是 driver、GPU 看不見它
計算路（CUDA / Level Zero → driver → GPU）一字不動；新增存取路（檔案 API → nvme.ko → SSD）；兩路在 DRAM 交會
GPU 無標準 → 廠商 driver；SSD 有 NVMe → 通用 driver；群聯一個 driver 都不寫，差異化在 SSD 韌體 + 預取策略
綁定 = 自家 SSD（硬體）+ 有整合的引擎（軟體）；不綁 Intel、不綁 CUDA、不綁 ISA；Mac 卡在插槽 / API / 動機
產品化 ≠ 技術突破：ZeRO-Infinity / FlexGen / LLM in a flash / llama.cpp mmap 早就有，新的是耐久盤 + 預取 + OEM 綁料件
```
