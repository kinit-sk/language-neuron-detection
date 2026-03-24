#!/bin/bash
set -xe


# WAIT UNTIL TRAINING FINISHES

# Generate results for `cze_benczechmark.py`
eval_model() {
    echo "Evaluating $1"
    lighteval accelerate "model_name=$1,batch_size=2,generation_parameters={\"temperature\":0.7}" \
        "squad_3_2_filtered_rc|0,czechbench_belebele_rc|0,propaganda_nazor_nli|0,propaganda_zanr_nli|0,mall_sentiment_balanced_sent|0,csfd_sentiment_balanced_sent|0,cermat_czech_mc_clu|0,umimeto_qa_biology_clu|0,umimeto_qa_chemistry_clu|0,umimeto_qa_czech_clu|0,umimeto_qa_history_clu|0,umimeto_qa_informatics_clu|0,umimeto_qa_math_clu|0,umimeto_qa_physics_clu|0" \
        --custom-tasks community_tasks/cze_benczechmark.py
}





# MODEL_PATH="/home/kopal/language-neuron-detection/data/7_finetuning/latn/checkpoint-30000"
# eval_model ${MODEL_PATH}

# MODEL_PATH="/home/kopal/language-neuron-detection/data/7_finetuning/non-latn/checkpoint-30000"
# eval_model ${MODEL_PATH}

# MODEL_PATH="/home/kopal/language-neuron-detection/data/7_finetuning/random/checkpoint-30000"
# eval_model ${MODEL_PATH}

# MODEL_PATH="/home/kopal/language-neuron-detection/data/7_finetuning/whole_model/checkpoint-30000"
# eval_model ${MODEL_PATH}

eval_model "meta-llama/Llama-3.2-3B-instruct"
# python create_table.py