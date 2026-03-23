#!/usr/bin/env bash


# Pass config name (without .yaml)


JOB_LOG_DIR="logs/$1"
mkdir -p "${JOB_LOG_DIR}"

sbatch --output "${JOB_LOG_DIR}/logs.out" --export=ALL,CONFIG_NAME="$1" ./7-finetuning-devana-job.sh


