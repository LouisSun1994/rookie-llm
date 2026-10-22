# score_eval.py — 把 run_eval.sh 產的 markdown 變成一張分數表
#
# 用法：
#   python score_eval.py before.md after.md                 # 兩份比
#   python score_eval.py before.md sweep/ckpt-*.md          # 學習曲線
#   python score_eval.py --ref before.md after.md ...       # 指定對照基準（預設：第一個叫 before*.md 的）
#
# 判定規則：
#   [T] 題：輸出 json.loads 後跟 EXPECTED 比，完全相等才 ✅
#   [P] 題：跟 --ref 的同題輸出比
#         =  一字不差
#         ≈  不同但沒崩（生成 ≥ 12 token，且不含 rtp / { ）
#         ✗  崩了（太短、或中性 prompt 下吐 JSON）
import json, re, sys
from pathlib import Path

# 期望答案要跟 run_eval.sh 的 PROMPTS 對齊；[P] 題不在這裡，因為它們沒有「正確答案」只有「跟 before 比」
EXPECTED = {
    "把這句轉成設定：RTP 九五，低波動":   {"rtp": 0.95, "volatility": "low"},
    "轉設定：這台 RTP 拉到 93，波動高":  {"rtp": 0.93, "volatility": "high"},
    "回饋率 96%，中波，寫成設定":        {"rtp": 0.96, "volatility": "medium"},
    "RTP 九八，波動超高":                {"rtp": 0.98, "volatility": "high"},
    "幫我查一下昨天的 RTP 報表":          {"error": "not a config request"},
}

SECTION = re.compile(r"^## \[(T|P)\] (.+)$", re.M)


def parse(path):
    """回傳 list of (kind, prompt, output, gen_tokens)，順序同檔案。"""
    text = Path(path).read_text(encoding="utf-8")
    heads = list(SECTION.finditer(text))
    rows = []
    for i, h in enumerate(heads):
        body = text[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        parts = body.split("==========")
        output = parts[1].strip() if len(parts) >= 3 else "(執行失敗)"
        m = re.search(r"Generation: (\d+) tokens", body)
        gen = int(m.group(1)) if m else 0
        rows.append((h.group(1), h.group(2).strip(), output, gen))
    return rows


def strip_fence(s):
    s = s.strip()
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def judge_t(prompt, output):
    exp = EXPECTED.get(prompt)
    if exp is None:
        return "?"
    try:
        got = json.loads(strip_fence(output))
    except Exception:
        return "✗json"
    return "✅" if got == exp else "✗"


def judge_p(output, gen, ref_output):
    if ref_output is not None and output == ref_output:
        return "="
    collapsed = gen < 12 or "rtp" in output.lower() or output.lstrip().startswith("{")
    return "✗" if collapsed else "≈"


def main(argv):
    ref_path = None
    if "--ref" in argv:
        i = argv.index("--ref")
        ref_path = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    files = argv
    if not files:
        print(__doc__ or "usage: python score_eval.py [--ref before.md] FILE...")
        return 1
    if ref_path is None:
        ref_path = next((f for f in files if Path(f).name.startswith("before")), None)
    ref = {p: o for _, p, o, _ in parse(ref_path)} if ref_path else {}

    header = None
    print_rows = []
    for f in files:
        rows = parse(f)
        if header is None:
            header = [f"{k}{n + 1}" for n, (k, _, _, _) in enumerate(rows)]
        cells, t_ok, t_n, p_keep, p_n = [], 0, 0, 0, 0
        for k, p, o, g in rows:
            if k == "T":
                v = judge_t(p, o); t_n += 1; t_ok += v == "✅"
            else:
                v = judge_p(o, g, ref.get(p)); p_n += 1; p_keep += v in ("=", "≈")
            cells.append(v)
        print_rows.append((Path(f).name, cells, f"{t_ok}/{t_n}", f"{p_keep}/{p_n}"))

    print("| file | " + " | ".join(header) + " | 目標+泛化+拒答 | 對照沒崩 |")
    print("|---|" + "---|" * len(header) + "---|---|")
    for name, cells, t, p in print_rows:
        print(f"| {name} | " + " | ".join(cells) + f" | {t} | {p} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
