# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval. Copy this file and complete it with the info for your task.

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.

Author:
"""
from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig

## Helper wheels
_warned = False
def truncate(text, max_chars):
    global _warned
    if not _warned:
        print(f"\033[91m [USER WARNING]\033[93m Using truncated query ({max_chars} chars) to avoid OOM errors. If you want to change this, modify the `truncate()` function in {__file__}.")
        print("Set as high as possible\033[0m")
        _warned = True
    return text[:max_chars]

## --- Prompt functions --- ##
def squad_prompt_fn(line: dict, task_name: str):
    """
    Defines how to go from a dataset line to a doc object.

    Converts SQuAD-style QA dataset to evaluation format
    """
    question = line.get("question", line.get("query"))
    query = f"Kontext: {line['context']}\n\nOtázka: {question}\n\nOdpověď: "

    return Doc(
        task_name=task_name,
        query=query,
        choices=[f"{line['answers']}"],  # single correct answer
        gold_index=0,                   # Only one choice, so idx 0
    )
def czechbench_belebele_prompt_fn(line: dict, task_name: str):
    query = f"Kontext: {line['flores_passage']}\n\nOtázka: {line['question']}\n\nOdpověď: "

    choices = [
        line['mc_answer1'], 
        line['mc_answer2'], 
        line['mc_answer3'], 
        line['mc_answer4']
    ]

    gold_index = int(line['correct_answer_num']) - 1

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
    )
def multiple_choice_w_context_prompt_fn(line: dict, task_name: str):
    """
    Example prompt function for multiple-choice tasks.
    """
    max_chars = 8_000

    if line["context"]:
        context_text = (
            truncate(line["context"], max_chars)
            if len(line["context"]) > max_chars
            else line["context"]
        )
    else: context_text = ""
    
    query_text = (
        truncate(line["query"], max_chars)
        if len(line["query"]) > max_chars
        else line["query"]
    )

    query = (
        f"Kontext: {context_text}\n\nVyberte správnou možnost.\n\n"
        f"Otázka:\n{query_text}\n\n"
        "Odpověď písmenem: "
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=line["choices"], 
        gold_index=line["gold"]
    )
def multiple_choice_prompt_fn(line: dict, task_name: str): # wrapper for multiple choice
    return _multiple_choice_core(
        line=line,
        task_name=task_name,
        query_key="query",
        gold_key="gold",
    )
def multiple_choice_agree_prompt_fn(line: dict, task_name: str): # wrapper for multiple choice
    return _multiple_choice_core(
        line=line,
        task_name=task_name,
        query_key="sentence",
        gold_key="answer_idx",
    )
def _multiple_choice_core( # master multiple choice prompt function 
    line: dict,
    task_name: str,
    query_key: str,
    gold_key: str
):
    max_chars = 8_000
    letters   = [chr(i+65) for i in range(len(line['choices']))] # some tasks need up to 8 choices

    query_text = (
        truncate(line[query_key], max_chars)
        if len(line[query_key]) > max_chars
        else line[query_key]
    )

    choices_txt = "\n".join(
        f"{letters[i]}. {c}" for i, c in enumerate(line["choices"])
    )

    query = (
        "Vyberte správnou možnost.\n\n"
        f"Otázka:\n{query_text}\n\n"
        f"Možnosti:\n{choices_txt}\n\n"
        "Odpověď písmenem: "
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=line["choices"], 
        gold_index=line[gold_key]
    )
def czech_nli_prompt_fn(line: dict, task_name: str):
    max_chars = 8_000

    premise = (
        truncate(line["cze_premise"], max_chars)
        if len(line["cze_premise"]) > max_chars
        else line["cze_premise"]
    )

    hypothesis = (
        truncate(line["cze_hypothesis"], max_chars)
        if len(line["cze_hypothesis"]) > max_chars
        else line["cze_hypothesis"]
    )

    choices = [
        "Platí",          # entailment (0)
        "Neplatí",        # contradiction (1)
        "Nelze určit"     # neutral (2)
    ]

    query = (
        "Určete vztah mezi větami.\n\n"
        f"Premisa:\n{premise}\n\n"
        f"Hypotéza:\n{hypothesis}\n\n"
        "Odpověď: "
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=line["label_cze"]
    )
def czech_subjectivity_prompt_fn(line: dict, task_name: str):
    max_chars = 8_000

    text = (
        truncate(line["text"], max_chars)
        if len(line["text"]) > max_chars
        else line["text"]
    )

    choices = [
        "Subjektivní",    # subjective (0)
        "Objektivní"      # objective (1)
    ]

    query = (
        "Určete, zda je následující text subjektivní nebo objektivní.\n\n"
        f"Text:\n{text}\n\n"
        "Odpověď: "
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=line["label"]
    )
def grammar_error_correction_prompt_fn(line: dict, task_name: str):
    max_chars = 8_000

    sentence = (
        truncate(line["query"], max_chars)
        if len(line["query"]) > max_chars
        else line["query"]
    )

    query = (
        "Je tato věta gramaticky správná?\n\n"
        f"Věta:\n{sentence}\n\n"
        "Odpovězte Ano nebo Ne: "
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=line['choices'],  # ["ne", "ano"]
        gold_index=line["gold"]
    )
def umimeto_qa_prompt_fn(line: dict, task_name: str):
    max_chars = 8_000

    question = (
        truncate(line["question"], max_chars)
        if len(line["question"]) > max_chars
        else line["question"]
    )

    # Map A/B -> index 0/1 for gold
    letter_to_index = {"A": 0, "B": 1}
    gold_index = letter_to_index[line["correct_answer"]]

    query = (
        "Vyberte správnou odpověď na následující otázku.\n\n"
        f"Otázka:\n{question}\n\n"
        f"A. {line['A']}\n"
        f"B. {line['B']}\n\n"
        "Odpověď písmenem: "
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=[line['A'], line['B']],  # two answer choices
        gold_index=gold_index
    )
def perplexity_prompt_fn(line: dict, task_name: str):
    max_chars = 18_000

    text = (
        truncate(line["text"], max_chars)
        if len(line["text"]) > max_chars
        else line["text"]
    )

    return Doc(
        task_name=task_name,
        query=text,
        choices=[""],
        gold_index=0
    )

    
## --- Task definitions --- ##
### Reading Comprehension ###
squad_3_2_filtered_rc = LightevalTaskConfig(
    name                = "squad_3_2_filtered_rc",
    prompt_function     = squad_prompt_fn,
    hf_repo             = "CZLC/sqad_3.2_filtered",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",  # Randomly sample few-shots from the train split
    metrics             = [Metrics.exact_match, Metrics.f1_score],  # Exact Match & F1 for QA
    generation_size     = 256,                              # Space for generated answers
    stop_sequence       = ["\n", "Question:", "Context:"],
)
czechbench_belebele_rc = LightevalTaskConfig(
    name                = "czechbench_belebele_rc",
    prompt_function     = czechbench_belebele_prompt_fn,
    hf_repo             = "davidadamczyk/czechbench_belebele",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",  # Randomly sample few-shots from the train split
    metrics             = [Metrics.exact_match, Metrics.f1_score],  # Exact Match & F1 for QA
    generation_size     = 256,                              # Space for generated answers
    stop_sequence       = ["\n", "Question:", "Context:"],
)

### Natural Language Inference ###
propaganda_argumentace_nli = LightevalTaskConfig(
    name                = "propaganda_argumentace_nli",
    prompt_function     = multiple_choice_prompt_fn,
    hf_repo             = "CZLC/propaganda_argumentace",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",
    metrics             = [Metrics.loglikelihood_acc, Metrics.loglikelihood_f1],  # Accuracy for classification
    generation_size     = 8,                    # Only need to generate the label
    stop_sequence       = ["\n"],
)
propaganda_fabulace_nli = LightevalTaskConfig(
    name                = "propaganda_fabulace_nli",
    prompt_function     = multiple_choice_prompt_fn,
    hf_repo             = "CZLC/propaganda_fabulace",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",
    metrics             = [Metrics.loglikelihood_acc, Metrics.loglikelihood_f1],  # Accuracy for classification
    generation_size     = 8,                    # Only need to generate the label
    stop_sequence       = ["\n"],
)
propaganda_nazor_nli = LightevalTaskConfig(
    name                = "propaganda_nazor_nli",
    prompt_function     = multiple_choice_prompt_fn,
    hf_repo             = "CZLC/propaganda_nazor",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",
    metrics             = [Metrics.loglikelihood_acc, Metrics.loglikelihood_f1],  # Accuracy for classification
    generation_size     = 8,                    # Only need to generate the label
    stop_sequence       = ["\n"],
)
propaganda_strach_nli = LightevalTaskConfig(
    name                = "propaganda_strach_nli",
    prompt_function     = multiple_choice_prompt_fn,
    hf_repo             = "CZLC/propaganda_strach",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",
    metrics             = [Metrics.loglikelihood_acc, Metrics.loglikelihood_f1],  # Accuracy for classification
    generation_size     = 8,                    # Only need to generate the label
    stop_sequence       = ["\n"],
)
propaganda_zamereni_nli = LightevalTaskConfig(
    name                = "propaganda_zamereni_nli",
    prompt_function     = multiple_choice_prompt_fn,
    hf_repo             = "CZLC/propaganda_zamereni",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",
    metrics             = [Metrics.pass_at_k_letters(sample_params={"k": 1, "n": 2})],  # More choices, so use Pass@1 instead of accuracy
    generation_size     = 2,
    stop_sequence       = ["\n"],
)
propaganda_demonizace_nli = LightevalTaskConfig(
    name                = "propaganda_demonizace_nli",
    prompt_function     = multiple_choice_prompt_fn,
    hf_repo             = "CZLC/propaganda_demonizace",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",
    metrics             = [Metrics.loglikelihood_acc, Metrics.loglikelihood_f1],  # Accuracy for classification
    generation_size     = 8,                    # Only need to generate the label
    stop_sequence       = ["\n"],
)
propaganda_lokace_nli = LightevalTaskConfig(
    name                = "propaganda_lokace_nli",
    prompt_function     = multiple_choice_prompt_fn,
    hf_repo             = "CZLC/propaganda_lokace",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",
    metrics             = [Metrics.pass_at_k_letters(sample_params={"k": 1, "n": 2})],  # More choices, so use Pass@1 instead of accuracy
    generation_size     = 8,                    # Only need to generate the label
    stop_sequence       = ["\n"],
)
propaganda_relativizace_nli = LightevalTaskConfig(
    name                = "propaganda_relativizace_nli",
    prompt_function     = multiple_choice_prompt_fn,
    hf_repo             = "CZLC/propaganda_relativizace",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",
    metrics             = [Metrics.loglikelihood_acc, Metrics.loglikelihood_f1],  # Accuracy for classification
    generation_size     = 8,                    # Only need to generate the label
    stop_sequence       = ["\n"],
)
propaganda_vina_nli = LightevalTaskConfig(
    name                = "propaganda_vina_nli",
    prompt_function     = multiple_choice_prompt_fn,
    hf_repo             = "CZLC/propaganda_vina",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",
    metrics             = [Metrics.loglikelihood_acc, Metrics.loglikelihood_f1],  # Accuracy for classification
    generation_size     = 8,                    # Only need to generate the label
    stop_sequence       = ["\n"],
)
propaganda_zanr_nli = LightevalTaskConfig(
    name                = "propaganda_zanr_nli",
    prompt_function     = multiple_choice_prompt_fn,
    hf_repo             = "CZLC/propaganda_zanr",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",
    metrics             = [Metrics.loglikelihood_acc, Metrics.loglikelihood_f1],  # Accuracy for classification
    generation_size     = 8,                    # Only need to generate the label
    stop_sequence       = ["\n"],
)
propaganda_emoce_nli = LightevalTaskConfig(
    name                = "propaganda_emoce_nli",
    prompt_function     = multiple_choice_prompt_fn,
    hf_repo             = "CZLC/propaganda_emoce",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",
    metrics             = [Metrics.pass_at_k_letters(sample_params={"k": 1, "n": 2})],  # More choices, so use Pass@1 instead of accuracy
    generation_size     = 8,                    # Only need to generate the label
    stop_sequence       = ["\n"],
)
propaganda_nalepkovani_nli = LightevalTaskConfig(
    name                = "propaganda_nalepkovani_nli",
    prompt_function     = multiple_choice_prompt_fn,
    hf_repo             = "CZLC/propaganda_nalepkovani",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",
    metrics             = [Metrics.loglikelihood_acc, Metrics.loglikelihood_f1],  # Accuracy for classification
    generation_size     = 8,                    # Only need to generate the label
    stop_sequence       = ["\n"],
)
propaganda_rusko_nli = LightevalTaskConfig(
    name                = "propaganda_rusko_nli",
    prompt_function     = multiple_choice_prompt_fn,
    hf_repo             = "CZLC/propaganda_rusko",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",
    metrics             = [Metrics.pass_at_k_letters(sample_params={"k": 1, "n": 2})],  # More choices, so use Pass@1 instead of accuracy
    generation_size     = 8,                    # Only need to generate the label
    stop_sequence       = ["\n"],
)
cs_snli_nli = LightevalTaskConfig(
    name                = "cs_snli_nli",
    prompt_function     = czech_nli_prompt_fn,
    hf_repo             = "CZLC/cs_snli",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",
    metrics             = [Metrics.pass_at_k_letters(sample_params={"k": 1, "n": 2})],  # More choices, so use Pass@1 instead of accuracy
    generation_size     = 8,                    # Only need to generate the label
    stop_sequence       = ["\n"],
)

## Sentiment Analysis
mall_sentiment_balanced_sent = LightevalTaskConfig(
    name                = "mall_sentiment_balanced_sent",
    prompt_function     = multiple_choice_prompt_fn,
    hf_repo             = "CZLC/mall_sentiment_balanced",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",
    metrics             = [Metrics.loglikelihood_acc, Metrics.loglikelihood_f1],  # Accuracy for classification
    generation_size     = 8,                    # Only need to generate the label
    stop_sequence       = ["\n"],
)
fb_sentiment_balanced_sent = LightevalTaskConfig(
    name                = "fb_sentiment_balanced_sent",
    prompt_function     = multiple_choice_prompt_fn,
    hf_repo             = "CZLC/fb_sentiment_balanced",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",
    metrics             = [Metrics.loglikelihood_acc, Metrics.loglikelihood_f1],  # Accuracy for classification
    generation_size     = 8,                    # Only need to generate the label
    stop_sequence       = ["\n"],
)
csfd_sentiment_balanced_sent = LightevalTaskConfig(
    name                = "csfd_sentiment_balanced_sent",
    prompt_function     = multiple_choice_prompt_fn,
    hf_repo             = "CZLC/csfd_sentiment_balanced",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",
    metrics             = [Metrics.loglikelihood_acc, Metrics.loglikelihood_f1],  # Accuracy for classification
    generation_size     = 8,                    # Only need to generate the label
    stop_sequence       = ["\n"],
)
czechbench_subjectivity_sent = LightevalTaskConfig(
    name                = "czechbench_subjectivity_sent",
    prompt_function     = czech_subjectivity_prompt_fn,
    hf_repo             = "davidadamczyk/czechbench_subjectivity",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",
    metrics             = [Metrics.loglikelihood_acc, Metrics.loglikelihood_f1],  # Accuracy for classification
    generation_size     = 8,                    # Only need to generate the label
    stop_sequence       = ["\n"],
)

## Czech Language Understanding
cs_gec_clu = LightevalTaskConfig(
    name                = "cs_gec_clu",
    prompt_function     = grammar_error_correction_prompt_fn,
    hf_repo             = "CZLC/cs_gec",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",
    metrics             = [Metrics.loglikelihood_acc, Metrics.loglikelihood_f1],  # Accuracy for classification
    generation_size     = 8,                    # Only need to generate the label
    stop_sequence       = ["\n"],
)
class umimeto_qa_clu(LightevalTaskConfig):
    def __init__(
        self,
        name, 
        hf_subset,
    ):
        super().__init__(
            name                = name,
            prompt_function     = umimeto_qa_prompt_fn,
            hf_repo             = "CZLC/umimeto-qa",
            hf_subset           = hf_subset,
            hf_avail_splits     = ["train", "test"],
            evaluation_splits   = ["test"],
            few_shots_split     = "train",
            few_shots_select    = "random_sampling_from_train",
            metrics             = [Metrics.loglikelihood_acc, Metrics.loglikelihood_f1],  # Accuracy for classification
            generation_size     = 8,                    # Only need to generate the label
            stop_sequence       = ["\n"],
        )
UMIMETO_QA_CLU = [
    umimeto_qa_clu(name=f"umimeto_qa_{subset}_clu", hf_subset=subset) 
    for subset in ["biology", "chemistry", "czech", "history", "informatics", "math", "physics"]
]
cermat_czech_tf_clu = LightevalTaskConfig(
    name                = "cermat_czech_tf_clu",
    prompt_function     = multiple_choice_prompt_fn,
    hf_repo             = "CZLC/cermat_czech_tf",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",  
    metrics             = [Metrics.exact_match, Metrics.f1_score],
    generation_size     = 8,
    stop_sequence       = ["\n", "Question:", "Context:"],
)
cermat_czech_mc_clu = LightevalTaskConfig(
    name                = "cermat_czech_mc_clu",
    prompt_function     = multiple_choice_w_context_prompt_fn,
    hf_repo             = "CZLC/cermat_czech_mc",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",  
    metrics             = [Metrics.exact_match, Metrics.f1_score],
    generation_size     = 8,
    stop_sequence       = ["\n", "Question:", "Context:"],
)
czechbench_agree_clu = LightevalTaskConfig(
    name                = "czechbench_agree_clu",
    prompt_function     = multiple_choice_agree_prompt_fn,
    hf_repo             = "davidadamczyk/czechbench_agree",
    hf_subset           = "",
    hf_avail_splits     = ["train", "test"],
    evaluation_splits   = ["test"],
    few_shots_split     = "train",
    few_shots_select    = "random_sampling_from_train",  
    metrics             = [Metrics.exact_match, Metrics.f1_score],
    generation_size     = 8,
    stop_sequence       = ["\n", "Question:", "Context:"],
)

## Language Modeling 
cnc_skript12_lm = LightevalTaskConfig(
    name                = "cnc_skript12_lm",
    prompt_function     = perplexity_prompt_fn,
    hf_repo             = "CZLC/CNC_skript12",
    hf_subset           = "",
    hf_avail_splits     = ["test"],
    evaluation_splits   = ["test"],
    metrics             = [Metrics.word_perplexity, Metrics.bits_per_byte, Metrics.byte_perplexity],
    generation_size     = None, # LM evaluation doesn't require generation
    stop_sequence       = None,
)
cnc_fictree_lm = LightevalTaskConfig(
    name                = "cnc_fictree_lm",
    prompt_function     = perplexity_prompt_fn,
    hf_repo             = "CZLC/CNC_fictree",
    hf_subset           = "",
    hf_avail_splits     = ["test"],
    evaluation_splits   = ["test"],
    metrics             = [Metrics.word_perplexity, Metrics.bits_per_byte, Metrics.byte_perplexity],
    generation_size     = None, # LM evaluation doesn't require generation
    stop_sequence       = None,
)
cnc_ksk_lm = LightevalTaskConfig(
    name                = "cnc_ksk_lm",
    prompt_function     = perplexity_prompt_fn,
    hf_repo             = "CZLC/CNC_KSK",
    hf_subset           = "",
    hf_avail_splits     = ["test"],
    evaluation_splits   = ["test"],
    metrics             = [Metrics.word_perplexity, Metrics.bits_per_byte, Metrics.byte_perplexity],
    generation_size     = None, # LM evaluation doesn't require generation
    stop_sequence       = None,
)
cnc_khavlicek_histnews_lm = LightevalTaskConfig(
    name                = "cnc_khavlicek_histnews_lm",
    prompt_function     = perplexity_prompt_fn,
    hf_repo             = "CZLC/CNC_KHavlicek_HistNews",
    hf_subset           = "",
    hf_avail_splits     = ["test"],
    evaluation_splits   = ["test"],
    metrics             = [Metrics.word_perplexity, Metrics.bits_per_byte, Metrics.byte_perplexity],
    generation_size     = None, # LM evaluation doesn't require generation
    stop_sequence       = None,
)
cnc_oral_ortofon_lm = LightevalTaskConfig(
    name                = "cnc_oral_ortofon_lm",
    prompt_function     = perplexity_prompt_fn,
    hf_repo             = "CZLC/CNC_oral_ortofon",
    hf_subset           = "",
    hf_avail_splits     = ["test"],
    evaluation_splits   = ["test"],
    metrics             = [Metrics.word_perplexity, Metrics.bits_per_byte, Metrics.byte_perplexity],
    generation_size     = None, # LM evaluation doesn't require generation
    stop_sequence       = None,
)
cnc_dialekt_lm = LightevalTaskConfig(
    name                = "cnc_dialekt_lm",
    prompt_function     = perplexity_prompt_fn,
    hf_repo             = "CZLC/CNC_Dialekt",
    hf_subset           = "",
    hf_avail_splits     = ["test"],
    evaluation_splits   = ["test"],
    metrics             = [Metrics.word_perplexity, Metrics.bits_per_byte, Metrics.byte_perplexity],
    generation_size     = None, # LM evaluation doesn't require generation
    stop_sequence       = None,
)


## --- Task table --- ##
TASKS_TABLE = [
    ## RC ##,
    squad_3_2_filtered_rc,
    czechbench_belebele_rc,
    ## NLI ##,
    propaganda_argumentace_nli,
    propaganda_fabulace_nli,
    propaganda_nazor_nli,
    propaganda_strach_nli,
    propaganda_zamereni_nli,
    propaganda_demonizace_nli,
    propaganda_lokace_nli,
    propaganda_relativizace_nli,
    propaganda_vina_nli,
    propaganda_zanr_nli,
    propaganda_emoce_nli,
    propaganda_nalepkovani_nli,
    propaganda_rusko_nli,
    cs_snli_nli,
    ## SENT ##
    mall_sentiment_balanced_sent,
    fb_sentiment_balanced_sent,
    csfd_sentiment_balanced_sent,
    czechbench_subjectivity_sent,
    ## CLU ##
    cs_gec_clu,
    *UMIMETO_QA_CLU,
    cermat_czech_tf_clu,
    cermat_czech_mc_clu,
    czechbench_agree_clu,
    ## LM ##
    cnc_skript12_lm,
    cnc_fictree_lm,
    cnc_ksk_lm,
    cnc_khavlicek_histnews_lm,
    cnc_oral_ortofon_lm,
    cnc_dialekt_lm,
]

