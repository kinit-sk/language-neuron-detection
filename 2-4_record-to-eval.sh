#!/usr/bin/env bash


echo "Recording activations..."
python 2_record_activations.py --config-name default > /dev/null

echo "LAPE..."
python 3_identify_neurons.py --config-name default > /dev/null

echo "Eval..."
python 4_identified_neurons_eval.py --config-name default > /dev/null
