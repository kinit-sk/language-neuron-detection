#!/usr/bin/env bash
set -xe

echo "[$1] Tokenization... (optional)"
python 1_tokenize.py --config-name $1 > /dev/null

echo "[$1] Recording activations..."
python 2_record_activations.py --config-name $1 > /dev/null

echo "[$1] LAPE..."
python 3_identify_neurons.py --config-name $1 > /dev/null

echo "[$1] Eval..."
python 4_identified_neurons_eval.py --config-name $1 > /dev/null
