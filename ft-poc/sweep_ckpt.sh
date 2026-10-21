#!/usr/bin/env bash
# sweep_ckpt.sh — 對 adapters/ 裡每個 checkpoint 跑同一份 run_eval.sh，看學習曲線
#
# 用法：
#   ./sweep_ckpt.sh                 # 預設讀 adapters/，輸出到 sweep/ckpt-NNNNNNN.md
#   ./sweep_ckpt.sh adapters-v4     # 指定 adapter 目錄
#
# mlx_lm.generate 的 --adapter-path 要的是「目錄」（adapters.safetensors + adapter_config.json），
# 所以每個 checkpoint 先組一個臨時目錄再丟給 run_eval.sh。
set -e

SRC=${1:-adapters}
OUT=sweep
mkdir -p "$OUT"

for ck in "$SRC"/0*_adapters.safetensors; do
  [ -e "$ck" ] || { echo "no checkpoints in $SRC"; exit 1; }
  it=$(basename "$ck" | cut -d_ -f1)          # 0000020
  dir="$OUT/.ck-$it"
  rm -rf "$dir" && mkdir -p "$dir"
  cp "$SRC/adapter_config.json" "$dir/"
  ln -s "$(cd "$(dirname "$ck")" && pwd)/$(basename "$ck")" "$dir/adapters.safetensors"
  ./run_eval.sh "$OUT/ckpt-$it.md" "$dir"
done
echo "完成 → $OUT/ckpt-*.md"
