#!/usr/bin/env bash
set -xe

params=("$@")

for param in "${params[@]}"; do
  python 1_tokenize.py --config-name "${param}" > /dev/null
  python 2_record_activations.py --config-name "${param}" > /dev/null
  python 3_identify_neurons.py --config-name "${param}" > /dev/null
  python 4_identified_neurons_eval.py --config-name "${param}" > /dev/null
done
