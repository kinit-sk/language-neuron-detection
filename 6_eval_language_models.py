import math
import os
from dataclasses import dataclass

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM

from eval_utils import (
    compute_diag_offdiag_metric,
    compute_perplexity,
    load_token_ids,
    save_heatmap_from_csv,
    save_matrix_csv,
)
from misc import get_device, get_pipeline_step, set_ex_id_from_config_name


@dataclass
class EvalLanguageModelsConfig:
    generated_models_dir: str
    save_dir: str
    tokenized_dir: str
    eval_languages: list[str]
    target_num_tokens: int
    seq_len: int
    batch_size: int


def _resolve_eval_cfg(cfg: DictConfig, ex_id: str) -> EvalLanguageModelsConfig:
    identify_neurons_cfg = cfg.get("identify_neurons")
    if identify_neurons_cfg is None:
        eval_models_cfg = OmegaConf.create({})
    else:
        eval_models_cfg = identify_neurons_cfg.get("evaluate_language_specific_models")
        if eval_models_cfg is None:
            eval_models_cfg = OmegaConf.create({})

    eval_identified_cfg = get_pipeline_step(cfg, "step4_identified_neurons_eval")
    generate_cfg = get_pipeline_step(cfg, "step5_generate_language_specific_model")
    tokenize_cfg = get_pipeline_step(cfg, "step1_tokenize")
    record_cfg = get_pipeline_step(cfg, "step2_record_activations")

    generated_models_dir = eval_models_cfg.get("generated_models_dir")
    if generated_models_dir:
        resolved_generated_models_dir = str(generated_models_dir)
    else:
        base_generated_dir = str(generate_cfg.get("save_dir", "data/5_output"))
        if bool(generate_cfg.get("extend_save_dir_with_ex_name", True)):
            resolved_generated_models_dir = os.path.join(base_generated_dir, ex_id)
        else:
            resolved_generated_models_dir = base_generated_dir

    save_dir = str(eval_models_cfg.get("save_dir", "data/6_output"))

    tokenized_dir = eval_models_cfg.get("tokenized_dir")
    if tokenized_dir:
        resolved_tokenized_dir = str(tokenized_dir)
    elif eval_identified_cfg.get("tokenized_dir"):
        resolved_tokenized_dir = str(eval_identified_cfg.tokenized_dir)
    elif record_cfg.extend_load_dir_with_ex_name:
        resolved_tokenized_dir = os.path.join(tokenize_cfg.save_dir, ex_id)
    else:
        resolved_tokenized_dir = str(tokenize_cfg.save_dir)

    eval_languages = list(eval_models_cfg.get("eval_languages") or eval_identified_cfg.get("eval_languages") or cfg.main.languages)
    target_num_tokens = int(eval_models_cfg.get("target_num_tokens", eval_identified_cfg.target_num_tokens))
    seq_len = int(eval_models_cfg.get("seq_len", eval_identified_cfg.seq_len))
    batch_size = int(eval_models_cfg.get("batch_size", eval_identified_cfg.batch_size))

    return EvalLanguageModelsConfig(
        generated_models_dir=resolved_generated_models_dir,
        save_dir=save_dir,
        tokenized_dir=resolved_tokenized_dir,
        eval_languages=eval_languages,
        target_num_tokens=target_num_tokens,
        seq_len=seq_len,
        batch_size=batch_size,
    )


def _load_language_tokens(
    lang: str,
    tokenized_dir: str,
    target_num_tokens: int,
) -> torch.Tensor:
    return load_token_ids(
        lang=lang,
        tokenized_dir=tokenized_dir,
        target_num_tokens=target_num_tokens,
        missing_hint="Run 1_tokenize.py first or set evaluate_language_specific_models.tokenized_dir.",
    )


def _discover_generated_model_dirs(
    generated_models_dir: str,
    preferred_order: list[str],
) -> list[str]:
    if not os.path.isdir(generated_models_dir):
        raise FileNotFoundError(f"Generated models directory not found: {generated_models_dir}")

    model_dirs = {
        entry: os.path.join(generated_models_dir, entry)
        for entry in os.listdir(generated_models_dir)
        if os.path.isdir(os.path.join(generated_models_dir, entry))
        and os.path.exists(os.path.join(generated_models_dir, entry, "config.json"))
    }
    if not model_dirs:
        raise ValueError(f"No exported model directories found in {generated_models_dir}")

    ordered = [lang for lang in preferred_order if lang in model_dirs]
    extras = sorted(lang for lang in model_dirs if lang not in set(ordered))
    return ordered + extras


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    ex_id = set_ex_id_from_config_name()
    eval_cfg = _resolve_eval_cfg(cfg, ex_id)

    preferred_row_order = list(cfg.main.languages)
    row_languages = _discover_generated_model_dirs(eval_cfg.generated_models_dir, preferred_row_order)
    col_languages = eval_cfg.eval_languages

    device = get_device()
    print(f"Using device: {device}")
    model_dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32

    print(f"Preparing tokenized evaluation data from: {eval_cfg.tokenized_dir}")
    eval_tokens: dict[str, torch.Tensor] = {}
    for lang in col_languages:
        eval_tokens[lang] = _load_language_tokens(lang, eval_cfg.tokenized_dir, eval_cfg.target_num_tokens)
        print(f"Loaded {lang}: {eval_tokens[lang].numel()} tokens")

    print("Computing baseline perplexity (before ablation)...")
    baseline_model = AutoModelForCausalLM.from_pretrained(cfg.main.model_path, dtype=model_dtype).to(device, non_blocking=True)
    baseline_model.eval()
    baseline_ppx: dict[str, float] = {}
    for eval_lang in col_languages:
        ppx_before = compute_perplexity(
            model=baseline_model,
            token_ids=eval_tokens[eval_lang],
            seq_len=eval_cfg.seq_len,
            batch_size=eval_cfg.batch_size,
            device=device,
        )
        baseline_ppx[eval_lang] = ppx_before
        print(f"  baseline {eval_lang}: ppx_before={ppx_before:.6f}")
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    matrix: dict[str, dict[str, float]] = {}
    for ablation_lang in row_languages:
        model_path = os.path.join(eval_cfg.generated_models_dir, ablation_lang)
        print(f"Evaluating exported model for language: {ablation_lang}")
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=model_dtype).to(device, non_blocking=True)
        model.eval()

        row_result: dict[str, float] = {}
        for eval_lang in col_languages:
            ppx_after = compute_perplexity(
                model=model,
                token_ids=eval_tokens[eval_lang],
                seq_len=eval_cfg.seq_len,
                batch_size=eval_cfg.batch_size,
                device=device,
            )
            ppx_before = baseline_ppx[eval_lang]
            log_ppx_ratio = float(math.log(ppx_after / ppx_before))
            row_result[eval_lang] = log_ppx_ratio
            print(
                f"  {ablation_lang} -> {eval_lang}: "
                f"ppx_before={ppx_before:.6f}, ppx_after={ppx_after:.6f}, "
                f"log_ratio={log_ppx_ratio:.6f}"
            )
        matrix[ablation_lang] = row_result

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_dir = os.path.join(eval_cfg.save_dir, ex_id)
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, "log_ppx_ratio_matrix.csv")
    save_matrix_csv(csv_path, row_languages, col_languages, matrix)
    png_path = os.path.join(save_dir, "log_ppx_ratio_matrix.png")
    save_heatmap_from_csv(csv_path, png_path)
    diag_metric = compute_diag_offdiag_metric(row_languages, col_languages, matrix)

    result_payload = {
        "generated_models_dir": eval_cfg.generated_models_dir,
        "model_path": cfg.main.model_path,
        "row_languages": row_languages,
        "col_languages": col_languages,
        "tokenized_dir": eval_cfg.tokenized_dir,
        "seq_len": eval_cfg.seq_len,
        "batch_size": eval_cfg.batch_size,
        "target_num_tokens": eval_cfg.target_num_tokens,
        "metric": "log(ppx_after / ppx_before)",
        "baseline_ppx_before": baseline_ppx,
        "matrix": matrix,
        "diag_offdiag_metric": diag_metric,
    }

    pt_path = os.path.join(save_dir, "log_ppx_ratio_matrix.pt")
    torch.save(result_payload, pt_path)
    print(f"Saved matrix CSV to {csv_path}")
    print(f"Saved heatmap PNG to {png_path}")
    print(f"Saved matrix artifact to {pt_path}")
    print(
        "Diagonal/off-diagonal metric: "
        f"diagonal_sum={diag_metric['diagonal_sum']:.6f}, "
        f"off_diagonal_sum={diag_metric['off_diagonal_sum']:.6f}, "
        f"diag_minus_offdiag={diag_metric['diag_minus_offdiag']:.6f}"
    )


if __name__ == "__main__":
    main()
