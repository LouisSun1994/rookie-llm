# 02 — M1 啟動 Ollama 前的知識點：SLM、MoE、Docker Desktop 網路層

- 記錄日期：2026-10-16
- 學習方式：準備在 M1 16GB 上首次啟動引擎，從啟動指令往下挖，跨到模型分類（SLM / MoE / FFN）與 Docker Desktop 的網路層
- 目前定位：🥚 概念 → 🐣 本機驗證的邊界，尚未實際啟動

---

## 0. 一句話版本

M1 上「Ollama 原生跑、Open WebUI 進 Docker」；Open WebUI 找 Ollama 用的 `host.docker.internal` 是 dockerd 寫進容器 `/etc/hosts` 的一個名字，它在 Mac 上解到 Docker Desktop userspace proxy 認領的虛擬位址、在 Linux 上解到 kernel bridge 的 gateway——因為 macOS 沒有容器功能，中間多墊了一台 Linux VM。

---

## 1. 分層模型（骨架）

```
應用層       Open WebUI（Docker 容器）
             │  HTTP → host.docker.internal:11434
推理引擎層   Ollama = server（HTTP、模型管理）+ engine（llama.cpp，Metal backend）
             │  ollama serve 這個 process 就是引擎伺服器；ollama pull/run/ps 是 client
模型層       GGUF 權重 + tokenizer + chat template + 預設參數 + license（Ollama 的「模型」是這一包）
             │  llama3.1:8b / qwen3:8b / gemma3:4b
硬體層       M1 unified memory 16GB，GPU 可 wire 上限約 2/3（以 ollama serve 啟動 log 為準）

Docker 網路層（M1 專屬，正式機 Linux 沒有）
             Mac (macOS kernel，無容器功能)
              └─ Docker Desktop (app)
                  ├─ Linux VM ← 真正的 dockerd、容器、docker0 都在這
                  └─ userspace 網路 process (vpnkit/gvproxy) ← Mac ↔ VM 的「線」
```

M1 和 Linux 正式機的核心差異：Linux 上容器與 host **共用同一個 kernel**，沒有 VM 那層；Docker Desktop 多墊的那層是所有「M1 上 OK、Linux 上壞」坑的來源。

---

## 2. SLM 是什麼

**核心規則**：SLM 不是技術定義，是社群／行銷用語，邊界隨時間往上漂。「LLM」已從形容詞變成類別名（decoder-only Transformer + 大規模預訓練 + 指令微調），SLM 是這個類別裡尺寸小的那群。比喻：SQLite 和 PostgreSQL 都是 RDBMS，「嵌入式資料庫」指的就是 SQLite 那一類。

**關鍵細節**（2026-10 時）：

| 參數量 | 稱呼 | 代表 |
|---|---|---|
| < 1B | tiny / edge | Qwen3-0.6B, SmolLM |
| 1B – 4B | SLM 核心區 | Llama-3.2-3B, Qwen3-4B, Phi-4-mini, Gemma-3-4B |
| 4B – 10B | 灰色地帶 | Llama-3.1-8B, Qwen3-8B |
| 10B – 30B | mid | Qwen3-14B, Gemma-3-27B |
| 30B+ | LLM | |

聽到 SLM 時自動翻譯成：**「一個 3B–8B、Q4 塞得進一張消費級卡／16GB 筆電、bf16 塞得進 24GB 卡做 fine-tune、準備配 RAG 補知識或 fine-tune 補格式的 LLM」**。設計取捨：放棄當百科全書，保留語言理解、指令遵循、工具呼叫。

追問對方的兩個問題：幾 B、dense 還是 MoE；要它做什麼（「什麼都問」→ SLM 是錯的工具）。

對應我的目標：

| 需求 | SLM |
|---|---|
| 內部聊天、改寫 | 3B–8B 夠 |
| RAG 問公司文件 | 4B–8B 夠，瓶頸在 retrieval |
| fine-tune 學公司 DSL | 3B–8B 是甜蜜點 |
| 廣泛世界知識、長推理鏈 | 結構性輸，fine-tune 補不了 |

---

## 3. Dense vs MoE，FFN = MLP

**核心規則**：MoE 只改 FFN，attention 兩種模型都是每 token 全跑。每個 token 只讀 k 個 expert 的 FFN，但哪 k 個由 router 逐 token 決定，所以**全部 expert 都必須常駐 GPU 記憶體**。

```
dense 一層：  token → attention（全跑）→ FFN（全跑）→ 下一層
MoE 一層：    token → attention（全跑）→ router 挑 top-k → 只跑 k 個 expert FFN → 下一層
```

**關鍵細節**：
- FFN = MLP = position-wise feed-forward，同一個東西的不同叫法。HF 的 class 叫 `LlamaMLP`，GGUF tensor 叫 `blk.N.ffn_gate / ffn_up / ffn_down`（SwiGLU 三個矩陣）
- 結構：`x(4096) → up(14336) → SiLU → down(4096)`，Llama-3-8B 每層 FFN ≈ 176M 參數、attention ≈ 42M，**FFN 佔總參數約 2/3**，所以 MoE 改它就能把每 token 啟動量砍到 1/10
- 多人併發時 batch 內不同 token 選不同 expert，最後接近全部 expert 都被讀到，MoE 的頻寬優勢縮水

| | dense 30B | MoE 30B-A3B |
|---|---|---|
| 記憶體（Q4） | ~17 GB | ~17 GB |
| 每 token 讀取 | ~17 GB | ~1.7 GB |
| decode 速度 | 像 30B | 接近 3B |
| 16GB M1 | 放不下 | 放不下（記憶體決定，不是速度） |

---

## 4. M1 上的記憶體估算

**核心規則**：unified memory 沒有獨立 VRAM，但 macOS 限制 GPU 可 wire 的量（`iogpu.wired_limit_mb`，預設約 RAM 的 2/3）；Docker Desktop 的 VM 也在同一池搶。超載不會 CUDA OOM，會 swap 變慢——**Mac 經驗不要帶到 Linux**。

```
權重 ≈ params × bytes/param      Q4_K_M ≈ 0.58 B/param（~4.6 bpw）
KV cache/token = 2 × layers × kv_heads × head_dim × 2 bytes
   Llama-3-8B: 2 × 32 × 8 × 128 × 2 = 128 KB/token → 8k ctx ≈ 1 GB

16 GB 總量
 - macOS + 瀏覽器            ~3–4 GB
 - Docker VM (Open WebUI)    ~1.5–2 GB  ← Docker Desktop 記憶體上限調到 3–4GB
 - Ollama 8B Q4 @ 8k ctx     ~6.5 GB
 剩餘                         ~4 GB     → 32k ctx 就開始 swap
```

decode 是記憶體頻寬限制：M1 ~68 GB/s，8B Q4 權重 4.7GB → 理論上限 ~14 tok/s，實測 10–12；3B 接近翻倍。

---

## 5. Ollama 啟動與模型 tag

**核心規則**：`ollama serve` 和 `brew services start ollama` 是同一個 binary，差在誰管生命週期（shell vs launchd）。第一次用前景跑，看啟動 log 的 available memory、listen address、layer offload。兩個都開第二個會因 11434 被佔用起不來。

| | `ollama serve` | `brew services start ollama` |
|---|---|---|
| 誰管 | shell | launchd（≈ systemd） |
| 終端關掉 | 死 | 活 |
| log | stdout | `/opt/homebrew/var/log/ollama.log` |
| 改 env（OLLAMA_HOST 等） | 前面加 | 改 plist，麻煩 |

**關鍵細節**：
- tag 解析：`llama3.1:8b` → `llama3.1:8b-instruct-q4_K_M`，預設指 instruct 版不是 base 版
- `ollama show <model>` 看 context length、量化、架構；`ollama ps` 看記憶體與 CPU/GPU 比例（混合 = 權重沒全進 GPU，tok/s 掉一個數量級）
- `num_ctx` 預設不等於模型最大值（2026-10 時 4096），超過靜默截斷；Open WebUI 把整段歷史塞進 prompt，多聊幾輪就撞到
- 第一次請求慢是 load duration（mmap），不是推理慢；閒置 5 分鐘卸載（`OLLAMA_KEEP_ALIVE`）
- KV cache 不跨請求活著，第二輪的 prompt eval 時間隨歷史長度成長
- 選型（2026-10）：llama3.1:8b 當教材基準（英文強、繁中弱）；qwen3:8b 當上線候選（中文強、Apache 2.0）；gemma3:4b 當對照（tokenizer 對 CJK 省 token，授權是 Gemma Terms 不是 Apache）
- 查模型：`ollama.com/library`（菜單，能直接 pull）、`huggingface.co/models`（批發市場，fine-tune 階段從這拿）、`lmarena.ai` / `artificialanalysis.ai`（比較）

**看 listen port**：看 bind address 不只是 port
```bash
lsof -nP -iTCP -sTCP:LISTEN | grep ollama   # 127.0.0.1:11434 只本機；0.0.0.0 區網可連
curl -s localhost:11434                      # "Ollama is running"
OLLAMA_HOST=0.0.0.0:11434 ollama serve       # 改 bind
```

---

## 6. `host.docker.internal` 到底是什麼

**核心規則**：它是**一個名字**（名字層），不是網路（網路層）。dockerd 建容器時把它寫進容器的 `/etc/hosts`（或由 embedded DNS 回答，版本不同）。它解析到的 IP **不是查主機 IP 查來的，而是 dockerd 自己擁有、並已安排好有人應答的位址**。

| | Linux（正式機） | Docker Desktop（M1） |
|---|---|---|
| 名字是否預設存在 | 否，要 `--add-host=host.docker.internal:host-gateway` | 是，自動注入 |
| 解析到 | `172.17.0.1`（docker0 gateway，host kernel 上的真介面） | `192.168.65.254`（VM 上游網段，proxy 認領的虛擬位址） |
| 到 host 的路 | 一跳：容器 → docker0 = host | 兩跳：容器 → docker0(VM) → VM eth0 → Mac 上的 proxy → Mac loopback |
| Ollama 綁 127.0.0.1 能否被連到 | 不能，要綁 0.0.0.0 | 能，因為最後一跳是 Mac 上的 proxy 本機發起 |

**關鍵細節**：
- `host-gateway` 是 docker engine 認得的 magic keyword，**字面上照打**，不是變數；engine 建容器時替換成「這個 daemon 認為能到 host 的 IP」
- Mac 上 `dig host.docker.internal` no answer 是正常的：它只存在容器內的 resolver，不在 macOS resolver
- 不可能填 `127.0.0.1`（每個 netns 自己的 lo）或 `0.0.0.0`（bind 用的萬用位址，不是目的地）
- 不填主機對外 IP 的理由：換 Wi-Fi、斷網、多網卡都會壞；填 dockerd 自己控制的位址，環境怎麼變都穩
- 跨 M1 / Linux 都能跑的 compose 寫法：
  ```yaml
  extra_hosts:
    - "host.docker.internal:host-gateway"   # Mac 無害、Linux 必要
  ```
- 正式機最終解：Ollama 與 Open WebUI 同一個 compose network，用 service name `http://ollama:11434`，走 kernel bridge，不碰 host、不開對外 port

**三個 docker 預設 network 和它無關**：

```
bridge   容器有自己的 netns + eth0，接 docker0，NAT 出去 ← open-webui 在這，host.docker.internal 也是給這種容器用的
host     容器沒有自己的 netns，共用 host 的網路堆疊；Mac 上「host」= VM 不是 Mac
none     只有 lo
```
`docker network inspect` 只看得到 dockerd 管的網段（VM 內的 172.17.0.0/16），看不到 VM ↔ Mac 的 192.168.65.0/24，因為那段不歸 dockerd 管。

**驗證指令**：
```bash
docker exec open-webui cat /etc/hosts                        # 名字寫在哪
docker exec open-webui cat /etc/resolv.conf                  # 通常 127.0.0.11
docker exec open-webui getent hosts host.docker.internal     # 解到什麼
docker exec open-webui ip route                              # default via 172.17.0.1 → 只到 VM 門口
ifconfig | grep 192.168.65                                   # Mac 上沒有，證明不是網卡
ps aux | grep -E 'vpnkit|gvproxy'                            # 有，這個 process 就是那條線
# 進 VM 本身看兩層同時存在
docker run -it --rm --privileged --pid=host alpine nsenter -t 1 -m -u -n -i sh
  ip addr    # eth0 192.168.65.x + docker0 172.17.0.1
  ip route   # default via 192.168.65.1 ← proxy
```

---

## 7. 為什麼 Mac 多一層：主詞拆開

**核心規則**：容器不是一種東西，是 Linux kernel 的功能拼起來的（namespace、cgroup、veth + bridge + iptables）；dockerd 只是呼叫這些 syscall 的管理程式。macOS kernel（XNU）沒有這些功能，所以 Docker Desktop 必須開一台 Linux VM，再用一個 Mac 上的 userspace process 模擬整套 TCP/IP 當 VM 和 Mac 之間的「線」。

```
Docker Desktop（Mac 上的 app，com.docker.backend）
   管 VM、管 userspace 網路 process（它的子 process，隨 app 起落）、管 GUI 設定
dockerd（VM 裡的 Linux daemon）
   管容器、管 docker0、管 /etc/hosts 注入
docker CLI
   透過 socket 打到 VM 裡的 dockerd，Docker Desktop 只是把 socket 轉進去
```
Linux 正式機上這兩個主詞合併成一個：只有 dockerd。

用 userspace 模擬而不開真 vNIC：不用 root、不跟 VPN / 企業 proxy 打架、能做 `.254 → 127.0.0.1` 轉發；代價是效能差，**M1 上 Docker 內的網路吞吐不代表正式機**。

---

## 思維鏈：一個請求從瀏覽器到 token 回來（M1 版）

```
t   CPU/GPU   元件                         發生什麼
t0  CPU(Mac)  瀏覽器                       POST localhost:3000，Docker Desktop 把 3000 轉進 VM
t1  CPU(VM)   open-webui 容器              組 prompt：system + 全部歷史 + 新訊息
t2  CPU(VM)   容器 resolver                查 host.docker.internal → /etc/hosts → 192.168.65.254
t3  CPU(VM)   容器 → docker0 → VM eth0     TCP 往 .254，default route 丟給 192.168.65.1
t4  CPU(Mac)  userspace proxy (vpnkit)     收到目的地 .254，自己對 127.0.0.1:11434 開一條 TCP
t5  CPU(Mac)  Ollama server                收 /api/chat；模型沒載入 → mmap GGUF（load duration，秒級）
t6  CPU(Mac)  llama.cpp tokenizer          文字 → token ids（prompt eval count = 這裡的數量）
t7  GPU(Mac)  llama.cpp Metal              prefill：所有 prompt token 一次算，寫 KV cache（prompt eval rate）
t8  GPU(Mac)  llama.cpp Metal              decode：每 step 產 1 token，讀整份權重 + KV（eval rate，頻寬限制）
t9  CPU(Mac)  Ollama server                逐 token streaming 回 HTTP，原路經 proxy → VM → 容器
t10 CPU(VM)   open-webui                   渲染；請求結束 KV cache 丟棄，下一輪整段歷史重 prefill
```
Linux 正式機版：t0 直接到容器、t2 解到 service name 的容器 IP、t3–t4 合併成一跳 kernel bridge、沒有 proxy。

---

## 陷阱清單

- ❌ SLM 和 LLM 是兩種東西 → ✅ 同一類別（LLM 是類別名）、不同尺寸、不同用法假設 → 為什麼：「LLM」已經名詞化，S 只是尺寸形容詞
- ❌ dense 和 MoE 的差別是「每 token 是否跑整個模型」 → ✅ 是「每 token 是否跑整個 FFN」，attention 都全跑 → 為什麼：MoE 只替換 FFN，因為 FFN 佔 2/3 參數
- ❌ MoE 30B-A3B 速度像 3B 所以 16GB 放得下 → ✅ 記憶體像 30B，放不下 → 為什麼：router 逐 token 選 expert，所有 expert 都要常駐
- ❌ 把 Ollama 也丟進 Docker 比較整齊 → ✅ M1 上 Ollama 原生跑 → 為什麼：Docker Desktop 無 Metal passthrough，引擎變純 CPU
- ❌ `host.docker.internal` 是 docker 的 default bridge → ✅ 它是一個名字，bridge 是一條網路 → 為什麼：路牌 vs 路，不同層
- ❌ `host.docker.internal` 是 `host` network driver → ✅ 只是撞名；它給 bridge 模式容器用，host driver 是不隔離整個共用 → 為什麼：方向相反，一個開一條指定的路、一個拆掉隔離
- ❌ 解析結果是 Mac 的網卡位址 → ✅ Mac 上是 proxy 認領的虛擬位址（ifconfig 找不到），Linux 上才是 host 的真介面 → 為什麼：Mac 上整段是 userspace 模擬
- ❌ `host-gateway` 是要替換的變數 → ✅ 字面照打的 magic keyword → 為什麼：engine 在建容器時自己替換
- ❌ Docker Desktop 應該填 docker0 的 172.17.0.1 → ✅ 填 192.168.65.254 → 為什麼：172.17.0.1 是 VM 的 docker0，封包會停在 VM 到不了 Mac
- ❌ dockerd 要「查」主機 IP 才能寫 /etc/hosts → ✅ 它填自己擁有、已安排應答的位址 → 為什麼：查對外 IP 會在換網路、斷網、多網卡時壞
- ❌ Mac 在網卡上開了 192.168.65.0/24 → ✅ 是一個 userspace process 模擬出來的 → 為什麼：macOS 沒容器功能，Docker Desktop 用 VM + userspace 網路補
- ❌ Linux 沒那層是因為「對等」 → ✅ 是因為容器和 host 共用同一個 kernel → 為什麼：容器 = kernel 功能的組合，不是獨立的東西
- ❌ Mac 上 Ollama 綁 127.0.0.1 容器就連不到 → ✅ M1 上通常通，Linux 上一定不通 → 為什麼：Mac 最後一跳是 proxy 本機發起；Linux 容器連的是 172.17.0.1 這張介面
- ❌ Mac 上測到的容器間網路延遲可以當正式機基準 → ✅ 不能 → 為什麼：多了 userspace proxy

---

## 心智模型總結

```
SLM      = 同一種 LLM，≤10B，一張卡放得下，配 RAG 補知識 / fine-tune 補格式
MoE      = 記憶體像總參數、速度像啟動參數；只改 FFN；多人併發優勢縮水
FFN=MLP  = 每層最肥的 2/3，up→SiLU→down (+gate)

M1 佈局  = Ollama 原生 (Metal) + Open WebUI in Docker；Docker VM 記憶體上限壓到 3–4GB
名字     = host.docker.internal 寫在容器 /etc/hosts，由 dockerd 填「自己擁有的位址」
Mac      = 容器 → docker0(VM) → VM eth0 → Mac 上 userspace proxy(.254) → 127.0.0.1:11434
Linux    = 容器 → docker0 = host(172.17.0.1)；要 --add-host + Ollama 綁 0.0.0.0
正式機   = 同 compose network，service name 互找，不碰 host
主詞     = Docker Desktop(app) 管 VM 和 proxy；dockerd(VM) 管容器和 docker0；Linux 只剩 dockerd
```
