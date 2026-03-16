# Language Neuron Detection

This repository explores language-specific neurons in LLMs. The current workflow has seven steps:

1. `1_tokenize.py`
   Tokenize the source dataset for each language and save train/validation token tensors to `data/1_output/<experiment>/`.
2. `2_record_activations.py`
   Run the base model on the tokenized data and record per-language neuron statistics to `data/2_output/<experiment>/`.
3. `3_identify_neurons.py`
   Apply LAPE over the recorded activations and save the selected language-specific neurons to `data/3_output/<experiment>/`.
4. `4_identified_neurons_eval.py`
   Evaluate neuron ablations directly on the base model and save cross-language perplexity matrices to `data/4_output/<experiment>/`.
5. `5_generate_language_specific_model.py`
   Export one ablated model per language by zeroing the selected neurons and save them to `data/5_output/<experiment>/`.
6. `6_eval_language_models.py`
   Evaluate the exported language-specific models and save cross-language perplexity matrices to `data/6_output/<experiment>/`.
7. `finetuning_impl/7_finetuning.py` via `7-finetuning.sh`
   Finetune the model while restricting updates to the selected neuron set, using `configs/7_finetuning_latn.yaml` for the current experiment setup.

## Setup

Use a simple pip setup with the requirements file:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the pipeline

The main pipeline scripts use Hydra with `configs/default.yaml`:

```bash
python 1_tokenize.py
python 2_record_activations.py
python 3_identify_neurons.py
python 4_identified_neurons_eval.py
python 5_generate_language_specific_model.py
python 6_eval_language_models.py
```

Step 7 uses the finetuning config and helper launcher:

```bash
bash 7-finetuning.sh
```

## Configuration

- Shared pipeline settings live in `configs/default.yaml`.
- The current finetuning experiment lives in `configs/7_finetuning_latn.yaml`.
- `cfg.main.languages` defines the language set used across the main pipeline unless a later step overrides it.

## Notes

- `2_visualize_activations.py` is a helper for inspecting recorded activations and is not one of the numbered pipeline steps.
- Step 7 is implemented separately from the Hydra pipeline in `configs/default.yaml`; it uses the selected neuron artifact from step 3.
