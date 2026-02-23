## Setup
```bash
pip install git+https://github.com/huggingface/lighteval.git
```


## Task Types:
| task suffix | task type                    |
| ----------- | ---------------------------- |
| _rc         | Reading Comprehension        |
| _nli        | Natural Language Inference   |
| _sent       | Sentiment Analysis           |
| _clu        | Czech Language Understanding |
| _lm         | Language Modeling            |

## How to run

> You can use command builder `lighteval_cmd_builder.py`, just change the parameters and use the names from the table. Don't forget to copy with the "*" (asterisk)
---

> Run from `...evaluate/lighteval/` 

```bash
lighteval accelerate \
	"model_name=<hf-path or local absolute path>" \
	"<task name|number of shots>" \
	--custom-tasks <path/to/the/file> \
	--max-samples <number of samples>
```
You can also run **multiple tasks** at once:
```bash
lighteval accelerate \
	"model_name=<hf-path or local absolute path>" \
	"<task name|number of shots>,<another task name|number of shots>" \
	--custom-tasks <path/to/the/file>
```
Run using **nohup**:
```bash
nohup lighteval accelerate \
	"model_name=<hf-path or local absolute path>" \
	"<task name|number of shots>" \
	--custom-tasks <path/to/the/file>
  > lighteval.log 2>&1 &
```
If OOM happens, sometimes exporting this helps:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
### Examples

> Run from `...evaluate/lighteval/` 

```bash
lighteval accelerate \
	"model_name=Qwen/Qwen2.5-0.5B-Instruct" \
	"squad_3_2_filtered_rc|0" \
	--custom-tasks community_tasks/benczechmark.py \
	--max-samples 100
```
You can also run **multiple tasks** at once:
```bash
lighteval accelerate \
	"model_name=Qwen/Qwen2.5-0.5B-Instruct" \
	"squad_3_2_filtered_rc|0,propaganda_argumentace_nli|0" \
	--custom-tasks community_tasks/benczechmark.py \
	--max-samples 100
```
Run using **nohup**:
```bash
nohup lighteval accelerate \
	"model_name=Qwen/Qwen2.5-0.5B-Instruct" \
	"squad_3_2_filtered_rc|0" \
	--custom-tasks community_tasks/benczechmark.py
  > lighteval.log 2>&1 &
```
How to run tasks marked with "\*" :
```bash
lighteval accelerate \
    "model_name=Qwen/Qwen2.5-0.5B-Instruct,generation_parameters={\"temperature\":0.7}" \
    "propaganda_emoce_nli|0" \
    --custom-tasks community_tasks/benczechmark.py
```


## All implemented tasks 

| Task name                    | Task type                    | prompt fn                           | # samples on GeForce RTX 3090 |
| ---------------------------- | ---------------------------- | ----------------------------------- | ----------------------------- |
| squad_3_2_filtered_rc        | Reading Comprehension        | squad_prompt_fn                     | 705                           |
| czechbench_belebele_rc       | Reading Comprehension        | czechbench_belebele_prompt_fn       | 110                           |
| x                            | x                            | x                                   | x                             |
| propaganda_argumentace_nli   | Natural Language Inference   | multiple_choice_prompt_fn           | 1000                          |
| propaganda_fabulace_nli      | Natural Language Inference   | multiple_choice_prompt_fn           | 1000                          |
| propaganda_nazor_nli         | Natural Language Inference   | multiple_choice_prompt_fn           | 2000                          |
| propaganda_strach_nli        | Natural Language Inference   | multiple_choice_prompt_fn           | 2000                          |
| propaganda_zamereni_nli*   | Natural Language Inference   | multiple_choice_prompt_fn           | 2000 (pass@k metric)          |
| propaganda_demonizace_nli    | Natural Language Inference   | multiple_choice_prompt_fn           | 2000                          |
| propaganda_lokace_nli*       | Natural Language Inference   | multiple_choice_prompt_fn           | 2000 (pass@k metric)          |
| propaganda_relativizace_nli  | Natural Language Inference   | multiple_choice_prompt_fn           | 1000                          |
| propaganda_vina_nli          | Natural Language Inference   | multiple_choice_prompt_fn           | 1000                          |
| propaganda_zanr_nli          | Natural Language Inference   | multiple_choice_prompt_fn           | 1000                          |
| propaganda_emoce_nli*       | Natural Language Inference   | multiple_choice_prompt_fn           | 152 (pass@k metric)           |
| propaganda_nalepkovani_nli   | Natural Language Inference   | multiple_choice_prompt_fn           | 1000                          |
| propaganda_rusko_nli*      | Natural Language Inference   | multiple_choice_prompt_fn           | 84(pass@k metric)             |
| cs_snli_nli*                 | Natural Language Inference   | czech_nli_prompt_fn                 | 234(pass@k metric)            |
| x                            | x                            | x                                   | x                             |
| mall_sentiment_balanced_sent | Sentiment Analysis           | multiple_choice_prompt_fn           | 1449                          |
| fb_sentiment_balanced_sent   | Sentiment Analysis           | multiple_choice_prompt_fn           | 63                            |
| csfd_sentiment_balanced_sent | Sentiment Analysis           | multiple_choice_prompt_fn           | 1504                          |
| czechbench_subjectivity_sent | Sentiment Analysis           | czech_subjectivity_prompt_fn        | 124                           |
| x                            | x                            | x                                   | x                             |
| cs_gec_clu                   | Czech Language Understanding | grammar_error_correction_prompt_fn  | 292                           |
| umimeto_qa_biology_clu       | Czech Language Understanding | umimeto_qa_prompt_fn                | 1000                          |
| umimeto_qa_chemistry_clu     | Czech Language Understanding | umimeto_qa_prompt_fn                | 1000                          |
| umimeto_qa_czech_clu         | Czech Language Understanding | umimeto_qa_prompt_fn                | 1000                          |
| umimeto_qa_history_clu       | Czech Language Understanding | umimeto_qa_prompt_fn                | 1000                          |
| umimeto_qa_informatics_clu   | Czech Language Understanding | umimeto_qa_prompt_fn                | 1000                          |
| umimeto_qa_math_clu          | Czech Language Understanding | umimeto_qa_prompt_fn                | 1000                          |
| umimeto_qa_physics_clu       | Czech Language Understanding | umimeto_qa_prompt_fn                | 1000                          |
| cermat_czech_tf_clu          | Czech Language Understanding | multiple_choice_prompt_fn           | 41                            |
| cermat_czech_mc_clu          | Czech Language Understanding | multiple_choice_w_context_prompt_fn | 163                           |
| czechbench_agree_clu         | Czech Language Understanding | multiple_choice_agree_prompt_fn     | 38                            |
| x                            | x                            | x                                   | x                             |
| cnc_skript12_lm              | Language Modeling            | perplexity_prompt_fn                | 1693                          |
| cnc_fictree_lm               | Language Modeling            | perplexity_prompt_fn                | 46                            |
| cnc_ksk_lm                   | Language Modeling            | perplexity_prompt_fn                | 400                           |
| cnc_khavlicek_histnews_lm    | Language Modeling            | perplexity_prompt_fn                | 1830                          |
| cnc_oral_ortofon_lm          | Language Modeling            | perplexity_prompt_fn                | 1167                          |
| cnc_dialekt_lm               | Language Modeling            | perplexity_prompt_fn                | 972                           |
