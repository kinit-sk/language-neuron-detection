#!/bin/bash
set -xe

eval_model() {
    echo "Evaluating $1 on czech tasks..."
    # Generate results for `cze_benczechmark.py`
    lighteval accelerate "model_name=$1,batch_size=2,generation_parameters={\"temperature\":0.7}" \
        "squad_3_2_filtered_rc|0,czechbench_belebele_rc|0,propaganda_nazor_nli|0,propaganda_zanr_nli|0,mall_sentiment_balanced_sent|0,csfd_sentiment_balanced_sent|0,cermat_czech_mc_clu|0,umimeto_qa_biology_clu|0,umimeto_qa_chemistry_clu|0,umimeto_qa_czech_clu|0,umimeto_qa_history_clu|0,umimeto_qa_informatics_clu|0,umimeto_qa_math_clu|0,umimeto_qa_physics_clu|0" \
        --custom-tasks community_tasks/cze_benczechmark.py \
        --output-dir "results_czech_tasks"


    echo "Evaluating $1 on en tasks..."
    echo -e "model_name: transformers\nmodel_parameters:\n  model_name: $1\n" > tmp_config.yaml
    # OLD: lighteval accelerate tmp_config.yaml "mmlu|0,hellaswag|0" --max-samples 500
    python misc/patch_mmlu_metric.py accelerate tmp_config.yaml "mmlu|0,hellaswag|0" --max-samples 8000 --output-dir "results_en_tasks"
    rm tmp_config.yaml
}




MODELS=(
  "meta-llama/Llama-3.2-3B-instruct"
  "/home/kopal/language-neuron-detection/data/7_finetuning/latn/checkpoint-30000"
  "/home/kopal/language-neuron-detection/data/7_finetuning/latn-en-ceiling/checkpoint-30000"
)

for model in "${MODELS[@]}"; do
  eval_model ${MODEL_PATH}
done