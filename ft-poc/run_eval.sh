#!/usr/bin/env bash
# run_eval.sh v2 — 固定 7 題，前 4 題用任務 prompt、後 3 題用中性 prompt
#
# 用法：
#   ./run_eval.sh before.md              # 原始模型
#   ./run_eval.sh after2.md adapters     # 掛上 adapter
set -e

OUT=${1:-before.md}
ADAPTER=${2:-}
MODEL="Qwen/Qwen2.5-1.5B-Instruct"

# 兩句都必須與 gen_data.py 一字不差
SYSTEM_TASK="你是遊戲設定助理。把口語需求轉成 JSON，只輸出 JSON。"
SYSTEM_PLAIN="你是一個助理。"

# 格式：「T|題目」用任務 prompt，「P|題目」用中性 prompt
PROMPTS=(
  # ── 目標題：訓練資料同類型 ──
  "T|把這句轉成設定：RTP 九五，低波動"
  "T|轉設定：這台 RTP 拉到 93，波動高"
  "T|回饋率 96%，中波，寫成設定"
  # ── 泛化題：資料裡沒出現的組合 ──
  "T|RTP 九八，波動超高"
  # ── 拒答題：任務 prompt 下的不相關輸入，期望 {"error":...} ──
  "T|幫我查一下昨天的 RTP 報表"
  # ── 對照題：中性 prompt，看有沒有退步 ──
  "P|台灣的首都是哪裡？"
  "P|寫一個 Python for 迴圈印 1 到 5"
  "P|把「今天天氣很好」翻成英文"
)

EXTRA=()
[ -n "$ADAPTER" ] && EXTRA=(--adapter-path "$ADAPTER")

echo "# $OUT  ($(date '+%Y-%m-%d %H:%M'))  adapter=${ADAPTER:-none}" > "$OUT"
for item in "${PROMPTS[@]}"; do
  kind="${item%%|*}"          # | 前面的字母
  p="${item#*|}"              # | 後面的題目
  if [ "$kind" = "T" ]; then sys="$SYSTEM_TASK"; else sys="$SYSTEM_PLAIN"; fi
  {
    echo "## [$kind] $p"
    echo '```'
    mlx_lm.generate --model "$MODEL" "${EXTRA[@]}" \
      --system-prompt "$sys" --prompt "$p" --max-tokens 80 2>/dev/null \
      || echo "(執行失敗)"
    echo '```'
    echo
  } >> "$OUT"
  echo "done: [$kind] $p"
done
echo "完成 → $OUT"
