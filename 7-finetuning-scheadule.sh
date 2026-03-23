#!/usr/bin/env bash


# Pass config name (without .yaml)

sbatch --output "logs/7_finetuning_latn/logs.out" --export=ALL,CONFIG_NAME="$1" 7-finetuning-devana-job.sh


