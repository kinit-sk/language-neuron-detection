import torch
import hydra
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ORANGE = "\033[33m"
GREEN  = "\033[32m"
RESET  = "\033[00m"


def calculate_perplexity(model, tokenizer, dataset, text_field, max_length, stride, max_samples=None):
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0

    print(f"{GREEN}[INFO]{RESET} Starting perplexity calculation")
    processed_samples = 0
    for item in tqdm(dataset, desc=f"Streaming texts (up to {max_samples if max_samples else '∞'})"):
        text = item[text_field].strip()
        if not text:
            continue
            
        enc = tokenizer(text, return_tensors="pt")
        ids = enc.input_ids.to(device)
        seq_len = ids.size(1)

        prev_end = 0
        # sliding window
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            input_chunk = ids[:, begin_loc:end_loc]
            
            labels = input_chunk.clone()
            labels[:, :prev_end - begin_loc] = -100  # mask old tokens

            with torch.inference_mode():
                outputs = model(input_chunk, labels=labels)

            num_new_tokens = (end_loc - begin_loc) if prev_end == 0 else (end_loc - prev_end)
            total_loss += outputs.loss.item() * num_new_tokens
            total_tokens += num_new_tokens
            
            prev_end = end_loc
            if end_loc == seq_len:
                break
        
        processed_samples += 1
        if max_samples and processed_samples >= max_samples:
            break

    avg_loss = total_loss / total_tokens
    print(f"{GREEN}[INFO]{RESET} Perplexity calculation completed")
    return float(torch.exp(torch.tensor(avg_loss)))

import math

def calculate_metrics(
    model,
    tokenizer,
    dataset,
    text_field,
    max_length,
    stride,
    max_samples=None,
):
    model.eval()
    device = next(model.parameters()).device

    total_nll = 0.0      # sum of negative log-likelihoods (nats)
    total_tokens = 0
    total_bytes = 0

    processed_samples = 0

    for item in tqdm(dataset, desc="Streaming texts"):
        text = item[text_field].strip()
        if not text:
            continue

        # UTF-8 bytes for BPB
        total_bytes += len(text.encode("utf-8"))

        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        ids = enc.input_ids.to(device)
        seq_len = ids.size(1)

        prev_end = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            input_chunk = ids[:, begin_loc:end_loc]

            labels = input_chunk.clone()
            labels[:, :prev_end - begin_loc] = -100

            with torch.inference_mode():
                out = model(input_chunk, labels=labels)

            num_new_tokens = (
                end_loc - begin_loc if prev_end == 0 else end_loc - prev_end
            )

            total_nll += out.loss.item() * num_new_tokens
            total_tokens += num_new_tokens

            prev_end = end_loc
            if end_loc == seq_len:
                break

        processed_samples += 1
        if max_samples and processed_samples >= max_samples:
            break

    # --- metrics ---
    ce = total_nll / total_tokens                  # nats / token
    ppl = math.exp(ce)
    bpb = ce * (total_tokens / total_bytes) * math.log2(math.e)

    metrics = {
        "loss": ce,
        "perplexity": ppl,              # token PPL
        "byte_perplexity": 2 ** bpb,    # byte PPL
        "bpb": {
            "bpb": bpb,
            "tokens": total_tokens,
            "bytes": total_bytes,
        }
    }
    return metrics


def load_model(model_path, tokenizer_path):
    print(f"{GREEN}[INFO]{RESET} Loading model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def evaluate_model(model, tokenizer, ds_cfg, max_length, stride):
    dataset = load_dataset(
        ds_cfg["path"],
        ds_cfg["subset_name"],
        split=ds_cfg["split"],
        streaming=ds_cfg["streaming"],
    )
    # ppx = calculate_perplexity(
    metrics = calculate_metrics(
        model,
        tokenizer,
        dataset,
        text_field=ds_cfg["text_field"],
        max_length=max_length,
        stride=stride,
        max_samples=ds_cfg["max_samples"],
    )
    return metrics


def save_results(save_path, results):
    output_dir = Path(save_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = results['model_name'].replace('/', '_')
    output_file = output_dir / f"perplexity_{model_name_safe}_{timestamp}.json"

    with open(output_file, 'w') as f:
        import json
        json.dump(results, f, indent=2)
    
    print(f"{GREEN}[INFO]{RESET} Results saved to {output_file}")

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    model, tokenizer = load_model(cfg.model.path, cfg.model.tokenizer_path)

    results = {
        'model_name': cfg.model.path,
        'dataset_hf_path': cfg.evaluation.dataset_hf_path,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'max_length': cfg.evaluation.max_length,
            'stride': cfg.evaluation.stride,
        },
        'metrics': {}
    }

    for lang in cfg.datasets.keys():
        print(f"{GREEN}[INFO]{RESET} Evaluating language: {lang}")

        ds_cfg = {
            'path': cfg.datasets[lang].path,
            'subset_name': cfg.datasets[lang].subset_name,
            'split': cfg.datasets[lang].split,
            'text_field': cfg.datasets[lang].text_field,
            'streaming': cfg.evaluation.streaming,
            'max_samples': cfg.datasets[lang].max_samples,
        }
        
        metrics = evaluate_model(
            model,
            tokenizer,
            ds_cfg,
            cfg.evaluation.max_length,
            cfg.evaluation.stride,
        )

        results["metrics"][lang] = metrics
        print(f"{ORANGE}[RESULT]{RESET} Perplexity for {lang}: {metrics['perplexity']:.2f}")
    save_results(cfg.evaluation.save_path, results)

if __name__ == "__main__":
    main()