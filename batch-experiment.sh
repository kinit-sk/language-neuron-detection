#!/usr/bin/env bash



echo "Processing all configs in $1"
for file in $1/*; do


    config_name=$(basename "$file")

    echo "===Processing $file =================="
    echo "[$file] Recording activations..."
    python 2_record_activations.py --config-path $1 --config-name $config_name > /dev/null

    echo "[$file] LAPE..."
    python 3_identify_neurons.py --config-path $1 --config-name $config_name > /dev/null

    echo "[$file] Eval..."
    python 4_identified_neurons_eval.py --config-path $1 --config-name $config_name > /dev/null
    echo "========================================="
done