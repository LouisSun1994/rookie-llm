# gen_data.py v2 — 三類資料：目標題 / 拒答題 / 通用題
# 執行：python gen_data.py  → data/train.jsonl, data/valid.jsonl
import json, os, random

random.seed(42)                      # 固定種子，每次產出一樣，方便比較

SYSTEM_TASK = "你是遊戲設定助理。把口語需求轉成 JSON，只輸出 JSON。"
SYSTEM_PLAIN = "你是一個助理。"

# ── A. 目標題：用模板隨機組合，逼模型真的去讀輸入 ──────────────────
RTP_VALUES = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]
RTP_WORDS = {                        # 同一個值的多種寫法
    0.90: ["90", "0.9", "九十", "90%", "九零"],
    0.91: ["91", "0.91", "九一", "91%"],
    0.92: ["92", "0.92", "九二", "92%"],
    0.93: ["93", "0.93", "九三", "93%"],
    0.94: ["94", "0.94", "九四", "94%"],
    0.95: ["95", "0.95", "九五", "95%"],
    0.96: ["96", "0.96", "九六", "96%"],
    0.97: ["97", "0.97", "九七", "97%"],
    0.98: ["98", "0.98", "九八", "98%"],
}
RTP_LABEL = ["RTP", "rtp", "回饋率", "返還率", "RTP 值"]
VOL = {
    "low":    ["低波動", "波動低", "低波", "波動小", "穩一點"],
    "medium": ["中波動", "波動中等", "中波", "波動普通"],
    "high":   ["高波動", "波動高", "高波", "波動大", "波動拉高"],
}
TEMPLATES = [                        # {r}=RTP 片段 {v}=波動片段
    "把這句轉成設定：這台機 {r}，{v}",
    "轉設定：{r}，{v}",
    "{v}、{r}，幫我寫成設定",
    "{r} {v}",
    "設定：{r}，{v}",
    "新機台要 {r}，{v}，給我 JSON",
    "幫我產一份設定檔，{v}，{r}",
    "{v}，然後 {r}",
]

def gen_target(n):
    rows = []
    for _ in range(n):
        rtp = random.choice(RTP_VALUES)
        vol = random.choice(list(VOL))
        r = f"{random.choice(RTP_LABEL)} {random.choice(RTP_WORDS[rtp])}"
        if random.random() < 0.3:                      # 30% 機率把「拉到/調到」塞進去
            r = r.replace(" ", f" {random.choice(['拉到', '調到', '設'])} ", 1)
        q = random.choice(TEMPLATES).format(r=r, v=random.choice(VOL[vol]))
        a = {"rtp": rtp, "volatility": vol}
        rows.append((SYSTEM_TASK, q, json.dumps(a, ensure_ascii=False)))
    return rows

# ── B. 拒答題：任務 prompt 下收到不相關輸入 → 回錯誤 JSON ─────────────
REJECT_INPUTS = [
    "台灣的首都是哪裡？", "幫我查一下昨天的 RTP 報表", "今天天氣如何",
    "寫一個 Python for 迴圈", "把這句翻成英文：今天天氣很好", "你是誰",
    "這台機台上個月營收多少", "幫我訂便當", "RTP 是什麼意思", "波動度怎麼算的",
    "幫我把設定檔刪掉", "列出所有機台", "謝謝", "1+1 等於多少", "推薦一本書",
]
def gen_reject():
    a = json.dumps({"error": "not a config request"}, ensure_ascii=False)
    return [(SYSTEM_TASK, q, a) for q in REJECT_INPUTS]

# ── C. 通用題：中性 prompt 下正常回答，防止遺忘 ───────────────────────
GENERAL = [
    ("台灣的首都是哪裡？", "台北。"),
    ("寫一個 Python for 迴圈印 1 到 5", "for i in range(1, 6):\n    print(i)"),
    ("把「今天天氣很好」翻成英文", "The weather is nice today."),
    ("1+1 等於多少", "2。"),
    ("Docker 和 VM 的差別是什麼？", "Docker 共用宿主機核心、只隔離使用者空間，啟動快、佔用少；VM 各自有完整作業系統，隔離更強但較重。"),
    ("用一句話解釋什麼是 API", "API 是程式之間約定好的介面，定義怎麼呼叫、傳什麼、回什麼。"),
    ("把「謝謝你的幫忙」翻成英文", "Thank you for your help."),
    ("Python 裡 list 和 tuple 差在哪？", "list 可變、tuple 不可變；tuple 可以當 dict 的 key，list 不行。"),
    ("日本的首都是哪裡？", "東京。"),
    ("寫一個 bash 指令列出當前目錄的檔案", "ls -la"),
    ("什麼是 HTTP 404？", "伺服器找不到請求的資源。"),
    ("3 乘以 7 是多少", "21。"),
    ("用一句話說明什麼是 RAG", "在回答前先從外部資料檢索相關內容塞進 prompt，讓模型根據資料作答。"),
    ("把「早安」翻成英文", "Good morning."),
    ("Git 的 commit 和 push 差在哪？", "commit 是把變更記錄到本機 repo，push 是把本機 commit 送到遠端。"),
]
def gen_general():
    return [(SYSTEM_PLAIN, q, a) for q, a in GENERAL]

# ── 組裝 ──────────────────────────────────────────────────────────────
def to_row(system, q, a):
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]}

rows = gen_target(100) + gen_reject() + gen_general()
rows = [to_row(*r) for r in rows]
random.shuffle(rows)                 # 三類混在一起，避免模型連續看到同一類

n_valid = max(10, len(rows) // 10)   # 10% 當 valid
valid, train = rows[:n_valid], rows[n_valid:]

os.makedirs("data", exist_ok=True)
with open("data/train.jsonl", "w") as f:
    for r in train:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open("data/valid.jsonl", "w") as f:
    for r in valid:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"目標 100 / 拒答 {len(REJECT_INPUTS)} / 通用 {len(GENERAL)}")
print(f"train {len(train)} 筆, valid {len(valid)} 筆")
