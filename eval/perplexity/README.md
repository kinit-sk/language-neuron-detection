# Custom Perplexity Evaluation
> Runs model through dataset and calculates loss, perplexity, byte_perplexity, and bpb (BitsPerByte)

## Setup
1: create env
```bash
conda create -n perplexity-eval python=3.12
```
2: install packages

(🚨 please validate and remove this line 🚨)
```bash
pip install torch transformers datasets hydra-core tqdm omegaconf
```

## How to use
1. Use one of the default configs from `configs/` or create new one in that folder
2. run evaluation
```bash
python main.py -cn <config name (without .yaml)>
```
3. Results are saved in desired folder

## Config explained
- Model:
    - path: select model path on HuggingFace (or use local path) 
    - tokenizer_path: select tokenizer path
- Evaluation:
    - Dataset_hf_path: dataset path on HF
    - max_length: how many tokens the model sees at once (depends on model context size)
    - stride: how many tokens the window moves each step (controls overlap between chunks)
    - save_path: where to save results
    - streaming: True if streaming dataset is desired
- use_fix_serialized_bytes: set to `True` if the dataset contains incorrectly stored text as serialized bytes
- datasets:
    - <name used to store results, (protip: use the subset name)>
        - path: do not touch :)
        - subset_name: name of the subset (use "" if the dataset has no subsets)
        - split: name of the subset (use "" if the dataset has no subsets)
        - text_field: dataset split to evaluate (e.g., train, test, validation)
        - max_samples: number of samples to evaluate (use for debugging only)

## Results explained
- config: configuration used for evaluation
- metrics:
    - <name used to store results (e.g. subset name)>:
        - loss: average cross-entropy loss (in nats per token)
        - perplexity: standard token-level perplexity
        - byte_perplexity: perplexity computed at the byte level
        - bpb (bits per byte): includes:
            - bpb: bits per byte value
            - tokens: total number of tokens evaluated
            - bytes: total number of bytes evaluated