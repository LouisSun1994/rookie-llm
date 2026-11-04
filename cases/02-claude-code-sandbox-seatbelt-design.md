# Case 02 · Claude Code 的 sandbox：一個 harness 怎麼借 macOS Seatbelt 把 shell 關起來

> 日期：2026-11-04
> 學習方式：從一個現象出發（Claude Code 的 Bash 工具「同一個 `MEMORY.md` 用 `echo >>` 寫不進、用 Edit 寫得進」「`git` 有時 fatal 有時正常」「`osascript` 怎麼設都不通」），反覆用自己的話陳述 → 校正，沿「哪些工具被關 → 規則從哪來 → 誰執行 → 這個設計依賴什麼、會怎麼壞」往下挖
> 目前定位：🐣 本機驗證 ～ 🐊 fine-tune 第一次動手；這條線是 **harness 支線**（接 notes/05 的 harness / 引擎 / 模型職責邊界），不在主路線上
> 本篇範圍：只記 Claude Code 這個 harness 的 sandbox **設計**——邊界畫在哪、規則從哪來、為什麼這樣選、什麼情況會壞。Seatbelt 本身的 OS 概念（SBPL、ucred label、fork / exec 繼承、fd capability、XPC 代辦、DAC / MAC / VFS）細節在 rookie-rs note 05，此處只留定位這個 case 所需的最小骨架
> 時效標記：Claude Code 的 settings 欄位、`@anthropic-ai/sandbox-runtime` 套件、`sandbox-exec` 的 deprecated 狀態皆為 2026-11 時觀察，會變。Seatbelt 的 operation 清單來自逆向工程，各版 macOS 有增減
> 對應筆記：[rookie-rs note 05](https://github.com/LouisSun1994/rookie-rs/blob/main/notes/05-macos-sandbox-seatbelt-proc-cred.md)（OS 層全部細節）；harness 的職責邊界見 [notes/05](../notes/05-harness-and-first-lora-poc.md)
> 標「補強」「補充」的 blockquote 是整理進 repo 時加註的延伸，不在原本這條線的對話裡

---

## 0. 一句話版本

Claude Code 的 sandbox 是 **harness 層的一個 least-privilege 設計**：每次 Bash 呼叫前，從 settings 現產一張 default-deny 的 SBPL 白名單，用 Apple 出貨的 `sandbox-exec` 編譯後塞進 kernel、綁在那個 shell 的 process 上；**只有 Bash 走這條路**，Edit / Read / Write 走 harness 自己的 app 層規則，MCP server 是獨立 process 不受限；網路全部導向 harness 自己起的本機 proxy，再由 proxy 依 `allowed_domains` 放行。不認識的東西一律 deny（fail-closed），所以 Apple 改介面的代價是功能壞，不是安全破洞。

---

## 1. 分層模型（骨架）

```
harness 層     Claude Code (Node, user mode, 自己不被 sandbox)
               ├─ agent loop：模型吐 tool call 文字 → harness 決定執行
               ├─ Bash 工具        → spawn sandbox-exec -p "<SBPL>" zsh -lc '<指令>'   ← 唯一進 Seatbelt 的路
               ├─ Read / Edit / Write → Node 自己 open()，用 JS 在 app 層檢查 path 規則
               ├─ MCP server        → harness spawn 的獨立 process，不經 sandbox-exec
               ├─ WebFetch / Search → harness 自己發 HTTP
               └─ 本機 filtering proxy → harness 起的 user-mode process，依 allowed_domains 放行
抽象層         @anthropic-ai/sandbox-runtime：macOS → Seatbelt、Linux → bubblewrap + seccomp，後端可換
Apple user     /usr/bin/sandbox-exec + libsandbox：SBPL → bytecode → sandbox_init()
═══ syscall 邊界 ═══
kernel         MAC framework + Sandbox.kext：每個「取得資源」的 syscall 查該 process 的 ucred.cr_label
               VFS（含 DAC）、network、Mach IPC 各子系統上的 hook
```

每個概念屬於哪一層：

| 概念 | 層 |
|---|---|
| `sandbox.filesystem.read.denyOnly` / `allowWithinDeny`、`write.allowOnly` / `denyWithinAllow` | harness settings（產 profile 的輸入） |
| `/sandbox` 互動指令 | harness UI，改的是 settings |
| `dangerouslyDisableSandbox: true` | harness 層單次開關：那一次不 spawn sandbox-exec |
| `allowed_domains` | harness 起的 proxy；Seatbelt 只認 IP / port，不認域名 |
| `allowLocalBinding`、`allowUnixSockets` | 對應 Seatbelt 的 `network-bind`、unix-socket filter |
| SBPL profile 文字 | harness 現產，Apple 的 libsandbox 編譯 |
| `Operation not permitted`（EPERM） | kernel Sandbox.kext |
| `osascript -600` / XPC invalid | kernel 的 `mach-lookup` / `appleevent-send` 被擋，不是路徑 |
| 同一個 `MEMORY.md` Bash 寫不進、Edit 寫得進 | 兩個不同 process、兩套規則（kernel vs app 層） |
| `node: command not found` | shell 啟動檔差異（`bash -c` 不讀 rc），不是 sandbox |
| `GIT_CONFIG_GLOBAL=/dev/null` 能救 git | 側讀路徑導向白名單內；對 Mach IPC 無效 |

---

## 2. 邊界畫在哪：工具級，不是 session 級

**核心規則**：sandbox 不是「整個 session 進入某種模式」，是 **每一次 Bash 呼叫獨立套一次**，而且只套 Bash。

**關鍵細節**

| 工具 | 進 Seatbelt？ | 誰檢查 | 這條線上的例子 |
|---|---|---|---|
| Bash | 是，每條指令獨立 | kernel（label） | `echo >> MEMORY.md` 被擋；`git log` fatal |
| Read / Edit / Write | 否 | harness 自己的 JS permission 規則 | 同一個 `MEMORY.md` 用 Edit 寫成功 |
| MCP（Atlassian、Control_Chrome…） | 否，獨立 process | 各 MCP server 自己 | 走 Chrome 刪 Jira comment 沒問題 |
| WebFetch / WebSearch | 否 | harness | 查資料正常 |

> **補充**：WebFetch 和 WebSearch「不進 Seatbelt」的理由其實不同層。WebFetch 是 harness 這個 Node process 自己發 HTTP——label=NULL，跟 Edit 同一類（同機器、沒 label）；WebSearch（2026-11 時）是 Anthropic API 端的 server tool，搜尋的 HTTP 從 Anthropic 那邊發出，本機連一個 syscall 都沒有（不同機器）。所以表格裡「誰檢查 = harness」對 WebSearch 其實是「誰檢查 = API 端」。

- 為什麼只關 Bash：Bash 是唯一「執行任意程式碼」的工具，模型吐什麼它就跑什麼，harness 無法在 app 層逐條判斷；其他工具跑的是 harness 自己的程式碼，它知道自己在做什麼。**policy 要 mandatory 的地方，只有「harness 控制不了的那段」**
- 為什麼每次獨立：每次 Bash 呼叫是全新的 `zsh -lc`，跑完就死；沒有跨呼叫的狀態，每條指令可以各自決定要不要 `dangerouslyDisableSandbox`
- 代價：同一個檔案兩種結果看起來像矛盾；使用者要知道「哪個工具走哪條路」才能解讀錯誤

---

## 3. 規則從哪來：settings → 四層巢狀 → SBPL

**核心規則**：harness 不是在調 kernel，是在調 **每次產 profile 時餵進去的參數**；改 settings 立即生效，不用重啟。

**關鍵細節（2026-11 時觀察到的一個 session）**

| 層 | settings 欄位 | 該 session 的值 |
|---|---|---|
| 讀：先 deny | `read.denyOnly` | `/Users/admin`、`**/*.pem`、`**/.env` … |
| 讀：deny 裡再開洞 | `read.allowWithinDeny` | `/Users/admin/work`、`/Users/admin/.claude` |
| 寫：先 allow | `write.allowOnly` | `.`、`$TMPDIR`、`/Users/admin/work`、`/Users/admin/.claude` |
| 寫：allow 裡再封 | `write.denyWithinAllow` | `.claude/settings.json`、`~/.claude/projects`、`.claude/skills` … |

- SBPL 的語意是「同一 operation 後宣告的規則蓋前面」，harness 把 **路徑越長的規則排越後面**，效果就是 longest-prefix-match：`~/.claude/projects/.../MEMORY.md` 命中 `/Users/admin`(deny) → `/Users/admin/.claude`(allow) → `/Users/admin/.claude/projects`(deny)，最後一條贏
- 讀和寫是分開的 operation family（`file-read*` / `file-write*`），所以「讀得到、寫不進」是正常狀態
- 網路：`network-outbound` 全 deny，只放行到本機 proxy 的 port；`HTTP_PROXY` 環境變數已指向它；繞過 proxy 直連在 kernel 層就被擋。域名白名單（`allowed_domains`）在 proxy 做，因為 **Seatbelt 的 filter 只有 `remote ip` / `remote tcp`，沒有 hostname**——這是把一段判定搬回 user mode 的例子
- `allowUnixSockets: []` → docker、ssh-agent 的 socket 全擋；`allowLocalBinding` 對應 `network-bind`
- Mach IPC（Launch Services、Apple Events）整個關掉，所以 `osascript` 和 `open` 沒有任何 settings 能救，只有 `dangerouslyDisableSandbox` 或改走 MCP

一段簡化的產物長這樣：

```scheme
(version 1)
(deny default)
(allow process-exec* process-fork)
(allow file-read* (subpath "/Users/admin/work") (subpath "/Users/admin/.claude"))
(deny  file-read* (literal "/Users/admin/work/some-repo/.claude/settings.json"))
(allow file-write* (subpath "/Users/admin/work") (subpath "/Users/admin/.claude"))
(deny  file-write* (subpath "/Users/admin/.claude/projects"))
(deny  network-outbound)
(allow network-outbound (remote ip "localhost:<proxy-port>"))
```

---

## 4. 誰執行：出題與閱卷分開

**核心規則**：harness 寫考卷（SBPL），Apple 的 `sandbox-exec` 把考卷交給閱卷老師（kernel），之後 **harness 完全不在迴路裡**——它收不到任何 callback，只看到 exit code 和 stderr。

**關鍵細節**（展開見 rookie-rs note 05 §3–§5）
- `sandbox-exec` 是 loader：編譯、`sandbox_init()`、`execvp zsh`，然後自己被 exec 取代
- label 掛在那個 zsh 的 `ucred.cr_label`，fork / exec 自動繼承、只能收緊；zsh 跑的 git、git fork 的 ssh 全在同一張策略下
- 判定依據永遠是「發 syscall 的那個 process 自己的 label」：Edit 工具的 `open()` 是 Node 發的（label=NULL），Bash 的 `echo >>` 是 zsh 發的（有 label）
- 為什麼不在 harness 層自己判定：user mode 的 wrapper 可以被子 process 繞過（直接呼叫 syscall），kernel 是所有 process 都躲不掉的交會點。放在 app 層也攔不到
- 為什麼用 CLI 而不是直接呼叫 `sandbox_init()`：核心依賴一樣（編譯器仍是 libsandbox、引擎仍是 Sandbox.kext），CLI 已經夠薄，又省掉 Node 的 native addon

---

## 5. 為什麼這樣設計：五個取捨

| 取捨 | 選了什麼 | 換來什麼 | 付出什麼 |
|---|---|---|---|
| default-deny vs default-allow | default-deny + 白名單 | least privilege；不認識的東西自動擋（fail-closed） | 每個新工具、新 macOS service 都可能「莫名失敗」 |
| kernel 強制 vs app 層檢查 | Bash 走 kernel，其他走 app 層 | 任意程式碼那段有 mandatory 保證 | 同一檔案兩種結果，使用者要懂分層 |
| session 級 vs 呼叫級 | 每次 Bash 獨立套 | 無跨呼叫狀態，可逐條關閉 | 每次多一次 sandbox-exec 啟動成本 |
| Seatbelt 直連 vs proxy | 網路全導向 proxy | 域名級白名單（Seatbelt 做不到） | 多一個 user-mode process，繞 proxy 的工具會壞 |
| deprecated CLI vs 自己接 libsandbox | CLI | 薄、無 native code、整個工具鏈同一條船 | Apple 拿掉 CLI 時要換後端（抽象層已預留） |

**「reachable ≠ provided」**：技術上 harness 把 `/Users/admin` 加進白名單就能讀到 `~/.ssh`。這不是能力問題是策略問題——sandbox 的邊界同時是「這個 task 被授權範圍」的宣告。這和 notes/05 的 harness 邊界是同一件事：模型只產文字，**要不要執行、能執行到哪，是 harness 的責任**，不是模型的。

---

## 6. 會怎麼壞：失效模式與對應處理

| 現象 | 真正的層 | 處理 |
|---|---|---|
| `git` 任何子指令都 `fatal: unable to access '/Users/admin/.gitconfig'` | kernel：git 啟動時側讀白名單外的路徑 | `export GIT_CONFIG_GLOBAL=/dev/null XDG_CONFIG_HOME=$TMPDIR` 把側讀導向白名單內 |
| `osascript` -600 / `open` 失敗 | kernel：`mach-lookup` / `appleevent-send`，不是路徑 | env 繞不過；`dangerouslyDisableSandbox` 或改走 MCP |
| `curl` 到未列 host 失敗 | proxy 層：`allowed_domains` 沒列 | 該次呼叫加 `allowed_domains` |
| docker / ssh-agent socket 不通 | kernel：unix socket 全擋 | `allowUnixSockets` |
| 綁 port 失敗 | kernel：`network-bind` | `allowLocalBinding: true` |
| `node: command not found` | user：`bash -c` 不讀 `~/.zshrc`，nvm 沒載入 | 直接在 zsh 跑，或先 source |
| `sudo` 失敗 | user：沒有 tty；再往下是 `mach-lookup opendirectoryd` | 不在 sandbox 內做需要 root 的事 |
| 指令 120 秒 timeout | harness：timeout 不是 sandbox | 拆指令或改 timeout |
| 新版 macOS 後某工具突然壞 | Apple 新增 service / operation，profile 不認識 → deny | fail-closed，補 profile；壞的是功能不是安全 |
| Apple 拿掉 `sandbox-exec` | 整個 macOS 後端失效 | `sandbox-runtime` 換後端；Apple 自己的 App Sandbox 也靠同一套引擎，引擎不會消失 |

**Claude Code 跟 Apple 之間的依賴面（三層）**

| 層 | 狀態 |
|---|---|
| App Sandbox entitlements | 公開、支援；但要 code sign、不能動態套給任意子 process → 用不上 |
| `sandbox-exec` / `sandbox_init`（SDK `sandbox.h`） | 出貨、有 man page、10.8 起 deprecated、14 年沒拿掉；Bazel / Nix / Homebrew / Chromium 都靠它 → **接線點** |
| 完整 SBPL 語法、operation 清單 | 未文件化，逆向 + 讀 `/System/Library/Sandbox/Profiles/*.sb` → 寫規則的知識 |

macOS 上做 process-level sandbox 的所有人都在同一條船上；Linux 那邊 seccomp / namespace 有 kernel「不破壞 userspace」的 ABI 契約，macOS 沒給 Seatbelt 這種承諾。

**診斷表**

| 字串 | 誰 |
|---|---|
| `Operation not permitted` / `unable to access '/Users/admin/…'` | Seatbelt（kernel MAC，EPERM） |
| `Permission denied` | rwx / ACL（kernel DAC，EACCES） |
| `-600`、`XPC connection invalid`、`Couldn't communicate with a helper application` | Seatbelt 的 mach-lookup / appleevent，env 救不了 |
| `command not found`、timeout、exit code 1 | 不是 sandbox |

> **補強**：`Operation not permitted` 不等於一定是 Seatbelt。macOS 上另外兩個 policy 走同一個 MAC hook、同樣回 EPERM：TCC（`~/Desktop` `~/Documents` `~/Downloads` `~/Library/Mail` 等隱私保護目錄；授權算在 **responsible process** 頭上，sandbox 內的 zsh 算到 Claude.app）和 SIP（`/System` `/usr` 等帶 `restricted` flag 的路徑，連 root 都擋）。對 harness 使用者最實用的分辨法：同一條指令帶 `dangerouslyDisableSandbox` 重跑仍 EPERM → 不是 sandbox，關了也救不了，要看的是 Claude.app 的 TCC 授權、或路徑是否在 SIP 清單。三者對照見 rookie-rs note 05 §8 補強。

---

## 7. 對地端 LLM 服務的意義

**核心規則**：同樣的四個問題會出現在自己寫的 harness 上——當地端模型透過 tool call / MCP 碰內部系統時，「模型吐的指令在哪裡執行、被什麼綁住」是 harness 的設計題，模型層沒有這個概念。

**關鍵細節**
- 四個問題：哪些工具需要 mandatory 保證（執行任意程式碼的那段）→ 規則從哪來（settings、least privilege、default-deny）→ 誰強制（kernel，不是 app 層 wrapper）→ 什麼會 fail-closed
- Linux 正式機的對應物：bubblewrap（namespace）+ seccomp（syscall filter）+ Landlock（unprivileged 自我限制，最接近 Seatbelt 的味道）；`sandbox-runtime` 在 Linux 就是走這條
- 「reachable ≠ provided」同樣適用：harness 能碰到的資料不等於該給模型的資料；RAG 的資料範圍、tool 的白名單，都是 harness 層的授權宣告

> **補強**：「bubblewrap + seccomp + Landlock；`sandbox-runtime` 在 Linux 就是走這條」要拆開講。就我所知（2026-11 時）`sandbox-runtime` 的 Linux 後端只有 bubblewrap（namespace：獨立的 mount / pid / network 視角）+ seccomp（擋 syscall；網路是在斷網的 namespace 裡把 proxy 用 unix socket 橋進來），沒有用 Landlock。Landlock 是獨立的一條路：unprivileged process 對自己呼叫 `landlock_restrict_self()`、只能收緊不能放寬、規則以路徑為主——這三點都跟 `sandbox_init()` 同形狀，所以「最接近 Seatbelt 的味道」是對的，但那是概念對應，不是 `sandbox-runtime` 的實作。

---

## 思維鏈：完整時間軸

**情境**：模型吐出 tool call `Bash("git log")`，macOS，settings 如 §3。

| t | 層 | 元件 | 發生什麼 |
|---|---|---|---|
| t0 | harness | agent loop | 收到模型輸出的 tool call 文字；決定執行；查 permission 規則（此處無需使用者確認） |
| t1 | harness | Bash 工具 | 讀 settings，組出 SBPL 字串（四層巢狀 → 排序 → 文字）；確認 proxy 在跑 |
| t2 | harness | Node `spawn` | 啟動 `sandbox-exec -p "<SBPL>" zsh -lc 'git log'`，環境變數含 `HTTP_PROXY` 指向本機 proxy |
| t3 | Apple user | sandbox-exec + libsandbox | 編譯 SBPL → bytecode；`sandbox_init()` → kernel |
| t4 | kernel | Sandbox.kext | bytecode 掛到該 process 的 `ucred.cr_label`。**harness 從此不在迴路** |
| t5 | Apple user → kernel | sandbox-exec `execvp zsh` | `process-exec` 查 → allow；映像換成 zsh，label 保留 |
| t6 | user | zsh `-l` | 讀 `~/.zprofile` / `~/.zshrc`（nvm 載入） |
| t7 | user → kernel | zsh `fork` + `execve git` | 子 process 繼承 label |
| t8 | user → kernel | git `open("/Users/admin/.gitconfig")` | VFS → `mac_vnode_check_open` → Sandbox.kext：`/Users/admin` deny → EPERM；DAC 沒跑到 |
| t9 | user | git | ENOENT 可容忍、EPERM 不行 → `fatal: unable to access … Operation not permitted`，exit 128 |
| t10 | user | zsh | `-c` 沒下一行，exit。這個 zsh 死了 |
| t11 | harness | Bash 工具 | 收到 exit code + stderr，回給模型當 tool result；下一次 Bash 從 t1 重來（**可帶 `dangerouslyDisableSandbox`，那一次跳過 t2–t5 直接 spawn zsh**） |

**網路路徑**（`curl https://example.com`）：t8 換成 curl 讀 `HTTP_PROXY` → `connect(localhost:<proxy>)` → Seatbelt 的 `network-outbound` 對該 ip:port allow → proxy 收到 CONNECT → 查 `allowed_domains` → 不在就回 403；若 curl 被設成直連 example.com，`connect()` 在 kernel 就被 `(deny network-outbound)` 擋掉。

**Edit 工具對照**：同一個 t0，但 t1 之後是 harness 自己的 JS 檢查 path 規則 → Node 自己 `open()` → 該 process label=NULL → kernel 直接放行。整條路上沒有 sandbox-exec。

---

## 陷阱清單（這條線上實際撞到的）

1. ❌ 「有時行有時不行」是隨機 → ✅ 四個獨立原因疊在一起：工具不同（Bash vs Edit vs MCP）、路徑在不在白名單、指令偷偷側讀白名單外的檔案（git 讀 `~/.gitconfig`）、跟 sandbox 無關的環境差異（`bash -c` 沒 nvm、timeout） → 為什麼：先看錯誤字串分層，再決定用 env 繞、換工具、還是關 sandbox
2. ❌ 同一個檔案 Bash 寫不進、Edit 寫得進是矛盾 → ✅ 兩個不同 process（zsh 有 label、Node 沒有）、兩套規則（kernel vs app 層） → 為什麼：判定依據永遠是「發 syscall 的那個 process 自己的 label」
3. ❌ sandbox 是 session 級的模式 → ✅ 每次 Bash 呼叫獨立套一次，MCP / WebFetch 根本不在內 → 為什麼：policy 只需要 mandatory 在「harness 控制不了的那段」
4. ❌ Claude Code 自己參與每次檢查 / 「claude 這隻 proc 會 block」 → ✅ harness 只出題；`sandbox_init` 後完全不在迴路；擋的是 kernel → 為什麼：app 層 wrapper 會被子 process 繞過
5. ❌ sandbox 只管檔案，程式能跑就好 → ✅ 管 syscall 類別：file、network、Mach IPC、exec…；`osascript` 死在 `mach-lookup` / `appleevent-send` → 為什麼：exec 放行，後面每個 syscall 另算
6. ❌ 改 env 變數就能繞 → ✅ 只對「側讀路徑」有效（`GIT_CONFIG_GLOBAL`）；Mach IPC 沒有路徑可換 → 為什麼：env 改的是程式去 open 哪個路徑，改不了它要找哪個 service
7. ❌ `command not found` / timeout / exit code 1 是 sandbox → ✅ 只有 EPERM 類字串與 mach-lookup 類錯誤是 Seatbelt → 為什麼：見診斷表，每個字串對應不同檢查點
8. ❌ 靠逆向的 API，Apple 一改就斷，瞎子摸象 → ✅ 接線點（CLI / `sandbox.h`）是 Apple 出貨的 deprecated API，14 年沒拿掉；逆向的是寫規則的知識；drift 時 default-deny 讓功能壞、安全不壞 → 為什麼：fail-closed 是拿功能穩定性換安全穩定性
9. ❌ 關掉 sandbox 是全域開關 → ✅ `dangerouslyDisableSandbox` 是那一次呼叫不套 sandbox-exec，每條指令獨立判斷 → 為什麼：呼叫級設計
10. ❌ `allowed_domains` 是 Seatbelt 規則 → ✅ Seatbelt 只認 IP / port；域名過濾在 harness 起的 user-mode proxy，Seatbelt 只負責「只能連 proxy」 → 為什麼：kernel 看不到 hostname，那段判定必須搬回 user mode
11. ❌ 模型能「選擇」繞過 sandbox → ✅ 模型只產文字；要不要執行、在哪執行、能碰到哪，全是 harness 的決定 → 為什麼：和 notes/05 的 harness 邊界是同一件事

> **補強**（對應第 7、10 條）：第 7 條的「只有 EPERM 類字串是 Seatbelt」要再收一層——EPERM 是必要條件不是充分條件，TCC / SIP 同樣回 EPERM，分辨靠關掉 sandbox 重跑（§6 補強）；第 10 條的「Seatbelt 只負責只能連 proxy」是 macOS 後端的做法，Linux 後端是 network namespace 斷網後把 proxy 用 unix socket 橋進來，兩個後端做法不同、效果同構（§7 補強）。

---

## 心智模型總結

```
Claude Code sandbox = harness 層的 least-privilege 設計，不是 OS 功能也不是模型功能
只關 Bash（任意程式碼那段）：kernel 強制；Edit / Read / Write：app 層 JS；MCP / WebFetch：不在內
每次 Bash 呼叫：settings → 四層巢狀（讀 deny→開洞、寫 allow→封洞）→ SBPL（長路徑排後、後宣告的贏）→ sandbox-exec → kernel
出題在 harness、閱卷在 kernel；sandbox_init 之後 harness 不在迴路，只看 exit code 與 stderr
網路：Seatbelt 只放行到本機 proxy（它只認 IP/port）；域名白名單在 proxy（user mode）
失效分層：EPERM = Seatbelt 路徑；mach-lookup 類 = Seatbelt IPC（env 救不了）；403 = proxy；command not found / timeout = 不是 sandbox
default-deny → fail-closed：Apple 改介面壞的是功能不是安全；接線點是出貨的 deprecated API，逆向的只是規則知識
reachable ≠ provided：邊界同時是授權宣告；模型只產文字，執行與範圍是 harness 的責任
```
