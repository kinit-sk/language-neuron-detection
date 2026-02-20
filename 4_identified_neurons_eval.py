import csv
import math
import os
from dataclasses import dataclass
from typing import Any

import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM

from misc import get_device


MLP_KEY = "mlp_grad_average_activations"
ATTN_Q_KEY = "attn_q_proj_average_activations"
ATTN_K_KEY = "attn_k_proj_average_activations"
ATTN_V_KEY = "attn_v_proj_average_activations"


@dataclass
class EvalDataConfig:
    tokenized_dir: str


def _resolve_selected_neurons_path(cfg: DictConfig) -> str:
    override_path = cfg.identify_neurons.evaluate_identified_neurons.selected_neurons_path
    if override_path:
        return str(override_path)
    return os.path.join(
        cfg.identify_neurons.select_neurons.save_dir,
        cfg.main.ex_id,
        "lape_selected_neurons.pt",
    )


def _load_selected_neurons(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Selected neuron file not found: {path}")
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict) or "results" not in data:
        raise ValueError(f"Invalid selected neuron artifact: {path}")
    return data


def _build_eval_data_cfg(cfg: DictConfig) -> EvalDataConfig:
    eval_cfg = cfg.identify_neurons.evaluate_identified_neurons
    tokenized_dir = eval_cfg.get("tokenized_dir")
    if tokenized_dir:
        resolved = str(tokenized_dir)
    else:
        resolved = os.path.join(cfg.identify_neurons.tokenize.save_dir, cfg.main.ex_id)
    return EvalDataConfig(
        tokenized_dir=resolved,
    )


def _load_language_tokens(
    lang: str,
    data_cfg: EvalDataConfig,
    target_num_tokens: int,
) -> torch.Tensor:
    token_path = os.path.join(data_cfg.tokenized_dir, f"{lang}.pt")
    if not os.path.exists(token_path):
        raise FileNotFoundError(
            f"Tokenized file not found for {lang}: {token_path}. "
            "Run 1_tokenize.py first or set evaluate_identified_neurons.tokenized_dir."
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


def _find_model_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError("Unsupported model architecture: expected model.model.layers")


def _build_component_masks(
    selected: dict[str, Any],
    ablation_lang: str,
    layers: list[torch.nn.Module],
) -> dict[str, list[torch.Tensor | None]]:
    selected_results = selected["results"]
    num_layers = len(layers)
    masks: dict[str, list[torch.Tensor | None]] = {}

    for component_key in (MLP_KEY, ATTN_Q_KEY, ATTN_K_KEY, ATTN_V_KEY):
        component = selected_results.get(component_key)
        if not isinstance(component, dict):
            continue
        by_lang = component.get("selected_indices_by_language", {})
        per_layer_indices = by_lang.get(ablation_lang)
        if per_layer_indices is None:
            continue
        if len(per_layer_indices) != num_layers:
            raise ValueError(
                f"Layer mismatch for {component_key}/{ablation_lang}: "
                f"{len(per_layer_indices)} indices vs {num_layers} model layers"
            )

        layer_masks: list[torch.Tensor | None] = []
        for layer_idx, idx_list in enumerate(per_layer_indices):
            if not idx_list:
                layer_masks.append(None)
                continue
            if component_key == MLP_KEY:
                width = layers[layer_idx].mlp.gate_proj.out_features
            elif component_key == ATTN_Q_KEY:
                width = layers[layer_idx].self_attn.q_proj.out_features
            elif component_key == ATTN_K_KEY:
                width = layers[layer_idx].self_attn.k_proj.out_features
            else:
                width = layers[layer_idx].self_attn.v_proj.out_features

            mask = torch.ones(width, dtype=torch.float32)
            valid_idx = [i for i in idx_list if 0 <= i < width]
            if valid_idx:
                mask[valid_idx] = 0.0
                layer_masks.append(mask)
            else:
                layer_masks.append(None)
        masks[component_key] = layer_masks

    return masks


def _mask_output(_module: torch.nn.Module, _inputs: tuple[torch.Tensor], output: torch.Tensor, mask: torch.Tensor):
    return output * mask.to(device=output.device, dtype=output.dtype)


def _register_ablation_hooks(
    layers: list[torch.nn.Module],
    component_masks: dict[str, list[torch.Tensor | None]],
) -> list[Any]:
    handles: list[Any] = []

    for layer_idx, layer in enumerate(layers):
        mlp_mask = component_masks.get(MLP_KEY, [None] * len(layers))[layer_idx]
        if mlp_mask is not None:
            handles.append(
                layer.mlp.gate_proj.register_forward_hook(
                    lambda m, i, o, local_mask=mlp_mask: _mask_output(m, i, o, local_mask)
                )
            )

        q_mask = component_masks.get(ATTN_Q_KEY, [None] * len(layers))[layer_idx]
        if q_mask is not None:
            handles.append(
                layer.self_attn.q_proj.register_forward_hook(
                    lambda m, i, o, local_mask=q_mask: _mask_output(m, i, o, local_mask)
                )
            )

        k_mask = component_masks.get(ATTN_K_KEY, [None] * len(layers))[layer_idx]
        if k_mask is not None:
            handles.append(
                layer.self_attn.k_proj.register_forward_hook(
                    lambda m, i, o, local_mask=k_mask: _mask_output(m, i, o, local_mask)
                )
            )

        v_mask = component_masks.get(ATTN_V_KEY, [None] * len(layers))[layer_idx]
        if v_mask is not None:
            handles.append(
                layer.self_attn.v_proj.register_forward_hook(
                    lambda m, i, o, local_mask=v_mask: _mask_output(m, i, o, local_mask)
                )
            )

    return handles


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


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    eval_cfg = cfg.identify_neurons.evaluate_identified_neurons
    selected_path = _resolve_selected_neurons_path(cfg)
    selected = _load_selected_neurons(selected_path)

    row_languages = list(selected.get("languages", [])) or list(cfg.main.languages)
    col_languages = list(eval_cfg.eval_languages) if eval_cfg.eval_languages else list(cfg.main.languages)

    device = get_device()
    print(f"Using device: {device}")
    model_dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg.main.model_path, dtype=model_dtype).to(device, non_blocking=True)
    model.eval()

    data_cfg = _build_eval_data_cfg(cfg)
    target_num_tokens = int(eval_cfg.target_num_tokens)
    seq_len = int(eval_cfg.seq_len)
    batch_size = int(eval_cfg.batch_size)

    print(f"Preparing tokenized evaluation data from: {data_cfg.tokenized_dir}")
    eval_tokens: dict[str, torch.Tensor] = {}
    for lang in col_languages:
        eval_tokens[lang] = _load_language_tokens(lang, data_cfg, target_num_tokens)
        print(f"Loaded {lang}: {eval_tokens[lang].numel()} tokens")

    print("Computing baseline perplexity (before ablation)...")
    baseline_ppx: dict[str, float] = {}
    for eval_lang in col_languages:
        ppx_before = _compute_perplexity(
            model=model,
            token_ids=eval_tokens[eval_lang],
            seq_len=seq_len,
            batch_size=batch_size,
            device=device,
        )
        baseline_ppx[eval_lang] = ppx_before
        print(f"  baseline {eval_lang}: ppx_before={ppx_before:.6f}")

    layers = _find_model_layers(model)
    matrix: dict[str, dict[str, float]] = {}

    for ablation_lang in row_languages:
        print(f"Evaluating ablation language: {ablation_lang}")
        component_masks = _build_component_masks(selected, ablation_lang, layers)
        handles = _register_ablation_hooks(layers, component_masks)

        row_result: dict[str, float] = {}
        for eval_lang in col_languages:
            ppx_after = _compute_perplexity(
                model=model,
                token_ids=eval_tokens[eval_lang],
                seq_len=seq_len,
                batch_size=batch_size,
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

        for h in handles:
            h.remove()

    save_dir = os.path.join(eval_cfg.save_dir, cfg.main.ex_id)
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, "log_ppx_ratio_matrix.csv")
    _save_matrix_csv(csv_path, row_languages, col_languages, matrix)
    png_path = os.path.join(save_dir, "log_ppx_ratio_matrix.png")
    _save_heatmap_from_csv(csv_path, png_path)

    result_payload = {
        "selected_neurons_path": selected_path,
        "model_path": cfg.main.model_path,
        "row_languages": row_languages,
        "col_languages": col_languages,
        "tokenized_dir": data_cfg.tokenized_dir,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "target_num_tokens": target_num_tokens,
        "metric": "log(ppx_after / ppx_before)",
        "baseline_ppx_before": baseline_ppx,
        "matrix": matrix,
    }

    pt_path = os.path.join(save_dir, "log_ppx_ratio_matrix.pt")
    torch.save(result_payload, pt_path)
    print(f"Saved matrix CSV to {csv_path}")
    print(f"Saved heatmap PNG to {png_path}")
    print(f"Saved matrix artifact to {pt_path}")


if __name__ == "__main__":
    main()
