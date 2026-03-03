import csv
import math
import os
from dataclasses import dataclass

import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM

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
    eval_models_cfg = cfg.identify_neurons.get("evaluate_language_specific_models")
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
    token_path = os.path.join(tokenized_dir, f"{lang}.pt")
    if not os.path.exists(token_path):
        raise FileNotFoundError(
            f"Tokenized file not found for {lang}: {token_path}. "
            "Run 1_tokenize.py first or set evaluate_language_specific_models.tokenized_dir."
        )

    token_ids = torch.load(token_path, map_location="cpu")
    if isinstance(token_ids, list):
        token_ids = torch.tensor(token_ids, dtype=torch.long)
    elif isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.to(dtype=torch.long, device="cpu")
    else:
        raise ValueError(f"Unsupported token file format at {token_path}: {type(token_ids)!r}")

    if token_ids.ndim != 1:
        token_ids = token_ids.reshape(-1)

    if target_num_tokens > 0:
        token_ids = token_ids[:target_num_tokens]

    if token_ids.numel() < 2:
        raise ValueError(f"Not enough tokens for language {lang} in {token_path}")
    return token_ids


def _compute_perplexity(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> float:
    usable = (token_ids.numel() // seq_len) * seq_len
    if usable < seq_len:
        raise ValueError("Not enough tokens to form one evaluation chunk")
    ids = token_ids[:usable].view(-1, seq_len)

    total_nll = 0.0
    total_tokens = 0

    with torch.inference_mode():
        for start in range(0, ids.size(0), batch_size):
            batch = ids[start:start + batch_size].to(device, non_blocking=True)
            outputs = model(batch, use_cache=False)
            logits = outputs.logits[:, :-1, :].float()
            labels = batch[:, 1:]

            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="sum",
            )
            total_nll += float(nll.item())
            total_tokens += int(labels.numel())

    avg_nll = total_nll / max(total_tokens, 1)
    return float(math.exp(avg_nll))


def _save_matrix_csv(path: str, row_langs: list[str], col_langs: list[str], matrix: dict[str, dict[str, float]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ablated_language", *col_langs])
        for row_lang in row_langs:
            row = [row_lang] + [f"{matrix[row_lang][col_lang]:.8f}" for col_lang in col_langs]
            writer.writerow(row)


def _save_heatmap_from_csv(csv_path: str, png_path: str):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(f"CSV does not contain a valid matrix: {csv_path}")

    col_langs = rows[0][1:]
    row_langs: list[str] = []
    values: list[list[float]] = []

    for row in rows[1:]:
        if len(row) != len(col_langs) + 1:
            raise ValueError(f"Malformed CSV row in {csv_path}: {row}")
        row_langs.append(row[0])
        values.append([float(v) for v in row[1:]])

    fig_w = max(8.0, 1.0 + 0.9 * len(col_langs))
    fig_h = max(6.0, 1.0 + 0.7 * len(row_langs))
    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        values,
        xticklabels=col_langs,
        yticklabels=row_langs,
        cmap="coolwarm",
        annot=True,
        fmt=".4f",
        center=0.0,
        cbar_kws={"label": "log(ppx_after / ppx_before)"},
    )
    ax.set_xlabel("Evaluation language")
    ax.set_ylabel("Ablated language")
    ax.set_title("Log-Perplexity-Ratio Matrix After Language-Specific Neuron Ablation")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()


def _compute_diag_offdiag_metric(
    row_langs: list[str],
    col_langs: list[str],
    matrix: dict[str, dict[str, float]],
) -> dict[str, float]:
    diagonal_sum = 0.0
    off_diagonal_sum = 0.0
    total_sum = 0.0

    for row_lang in row_langs:
        row = matrix[row_lang]
        for col_lang in col_langs:
            value = float(row[col_lang])
            total_sum += value
            if row_lang == col_lang:
                diagonal_sum += value
            else:
                off_diagonal_sum += value

    return {
        "diagonal_sum": diagonal_sum,
        "off_diagonal_sum": off_diagonal_sum,
        "diag_minus_offdiag": diagonal_sum - off_diagonal_sum,
        "diag_fraction_of_total": (diagonal_sum / total_sum) if total_sum != 0.0 else 0.0,
    }


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
        ppx_before = _compute_perplexity(
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
            ppx_after = _compute_perplexity(
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
    _save_matrix_csv(csv_path, row_languages, col_languages, matrix)
    png_path = os.path.join(save_dir, "log_ppx_ratio_matrix.png")
    _save_heatmap_from_csv(csv_path, png_path)
    diag_metric = _compute_diag_offdiag_metric(row_languages, col_languages, matrix)

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
