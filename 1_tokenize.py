import os
import time

import hydra
import torch
from datasets import DownloadConfig, load_dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer

# Avoid native tokenizer worker threads surviving into interpreter shutdown.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Avoid optional native HF transfer backends that can keep extra threads/resources.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def stream_tokens_with_retries(tokenizer, dataset_name, lang_name, target_num_tokens, max_retries=6):
    download_config = DownloadConfig(max_retries=max_retries)

    for attempt in range(1, max_retries + 1):
        tensor_ids = []
        dataset = None
        dataset_iter = None
        try:
            dataset = load_dataset(
                dataset_name,
                lang_name,
                split="train",
                streaming=True,
                download_config=download_config,
            )

            dataset_iter = iter(dataset)
            for item in dataset_iter:
                tensor_ids.extend(tokenizer.encode(item["text"], add_special_tokens=False))
                if len(tensor_ids) >= target_num_tokens:
                    return tensor_ids[:target_num_tokens], True

            return tensor_ids, False
        except (OSError, ConnectionError, TimeoutError) as exc:
            print(
                f"Attempt {attempt}/{max_retries} failed for {dataset_name}/{lang_name}: {exc}"
            )
            if attempt == max_retries:
                raise
            # Exponential backoff for transient network/socket failures.
            time.sleep(min(2**attempt, 30))
        finally:
            for obj in (dataset_iter, dataset):
                close = getattr(obj, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass

    return [], False


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    save_path = os.path.join(cfg.identify_neurons.tokenize.save_dir, cfg.main.ex_id)
    os.makedirs(save_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.main.model_path, use_fast=True)
    target_num_tokens = cfg.identify_neurons.tokenize.target_num_tokens

    for lang in cfg.main.languages:
        print(f"\n=======================\nProcessing language: {lang}\n")

        if lang == "eng_Latn":
            dataset_name = "HuggingFaceFW/fineweb"
            lang_name = "default"
        else:
            dataset_name = cfg.identify_neurons.dataset_path
            lang_name = lang

        tensor_ids, reached_target = stream_tokens_with_retries(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            lang_name=lang_name,
            target_num_tokens=target_num_tokens,
        )

        if reached_target:
            print(f"Truncated {lang} to {target_num_tokens} tokens.")

        print(f"=========== Tokenized {len(tensor_ids)} tokens for {lang}")
        tensor_data = torch.LongTensor(tensor_ids)
        tokens_file_path = os.path.join(save_path, f"{lang}.pt")
        torch.save(tensor_data, tokens_file_path)
        print(f"\nSaved tokenized data to {tokens_file_path} / ({len(tensor_data)} tokens)")


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        raise
    finally:
        # Bypass interpreter finalization, which is where the sporadic
        # PyGILState_Release crash occurs in this environment.
        try:
            import sys

            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(exit_code)
