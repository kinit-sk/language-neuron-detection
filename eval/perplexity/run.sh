#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="$SCRIPT_DIR/configs/$1.yaml"
BACKUP_PATH="$(mktemp)"

cp "$CONFIG_PATH" "$BACKUP_PATH"
trap 'cp "$BACKUP_PATH" "$CONFIG_PATH"; rm -f "$BACKUP_PATH"' EXIT

MODELS=(
  "/home/kopal/language-neuron-detection/data/7_finetuning/latn/checkpoint-30000"
  "/home/kopal/language-neuron-detection/data/7_finetuning/non-latn/checkpoint-30000"
  "/home/kopal/language-neuron-detection/data/7_finetuning/whole_model_small_lr/checkpoint-30000"
  "/home/kopal/language-neuron-detection/data/7_finetuning/random/checkpoint-30000"
  "meta-llama/Llama-3.2-3B-instruct"
)







cd "$SCRIPT_DIR"

for model in "${MODELS[@]}"; do
  echo "Evaluating $model"
  sed -i "s|^  path: .*|  path: \"$model\"|" "$CONFIG_PATH"
  sed -i "s|^  tokenizer_path: .*|  tokenizer_path: \"$model\"|" "$CONFIG_PATH"
  # python main.py -cn $1
  python main.py -cn $1
done
