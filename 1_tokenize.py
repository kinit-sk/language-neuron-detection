import os
import hydra
import torch
from omegaconf import DictConfig
from datasets import load_dataset
from transformers import AutoTokenizer  



@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
      
    save_path = os.path.join(cfg.identify_neurons.tokenize.save_dir, cfg.main.ex_id)
    os.makedirs(save_path, exist_ok=True)
     
    tokenizer = AutoTokenizer.from_pretrained(cfg.main.model_path, use_fast=True)
    for lang in cfg.main.languages:
        print(f"\n=======================\nProcessing language: {lang}\n")

        if lang == "eng_Latn":
            dataset_name = "HuggingFaceFW/fineweb"
            lang_name = "default"
        else:
            dataset_name = cfg.identify_neurons.dataset_path
            lang_name = lang

        dataset = load_dataset(dataset_name, lang_name, split="train", streaming=True)

        tensor_ids = []
        for item in dataset:
            tensor_ids.extend(tokenizer.encode(item["text"], add_special_tokens=False))
            if len(tensor_ids) > cfg.identify_neurons.tokenize.target_num_tokens:
                tensor_ids = tensor_ids[:cfg.identify_neurons.tokenize.target_num_tokens]
                print(f"Truncated {lang}  to {cfg.identify_neurons.tokenize.target_num_tokens} tokens.")
                break

        tensor_data = torch.LongTensor(tensor_ids)
        tokens_file_path = os.path.join(save_path, f"{lang}.pt")
        torch.save(tensor_data, tokens_file_path)
        print(f"\n Saved tokenized data to {tokens_file_path} / ({len(tensor_data)} tokens)")
        print("=======================") 
             

if __name__ == "__main__":
    main()