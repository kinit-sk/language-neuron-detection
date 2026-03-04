import math
import os
from dataclasses import dataclass
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM

from eval_utils import (
    compute_diag_offdiag_metric,
    compute_perplexity,
    load_token_ids,
    save_heatmap_from_csv,
    save_matrix_csv,
)
from misc import get_device, get_pipeline_step, set_ex_id_from_config_name


MLP_KEYS = (
    "mlp_grad_average_activations",
    "mlp_over_threshold_rate",
)
ATTN_Q_KEYS = (
    "attn_q_proj_grad_average_activations",
    "attn_q_proj_over_threshold_rate",
)
ATTN_K_KEYS = (
    "attn_k_proj_grad_average_activations",
    "attn_k_proj_over_threshold_rate",
)
ATTN_V_KEYS = (
    "attn_v_proj_grad_average_activations",
    "attn_v_proj_over_threshold_rate",
)

COMPONENT_KEYS = {
    "mlp": MLP_KEYS,
    "attn_q": ATTN_Q_KEYS,
    "attn_k": ATTN_K_KEYS,
    "attn_v": ATTN_V_KEYS,
}


def _first_present_component(
    selected_results: dict[str, Any],
    candidate_keys: tuple[str, ...],
) -> tuple[str, dict[str, Any]] | None:
    for key in candidate_keys:
        value = selected_results.get(key)
        if isinstance(value, dict):
            return key, value
    return None


@dataclass
class EvalDataConfig:
    tokenized_dir: str


def _resolve_selected_neurons_path(cfg: DictConfig, ex_id: str) -> str:
    eval_cfg = get_pipeline_step(cfg, "step4_identified_neurons_eval")
    override_path = eval_cfg.selected_neurons_path
    if override_path:
        return str(override_path)
    identify_cfg = get_pipeline_step(cfg, "step3_identify_neurons")
    return os.path.join(
        identify_cfg.save_dir,
        ex_id,
        "lape_selected_neurons.pt",
    )


def _load_selected_neurons(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Selected neuron file not found: {path}")
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict) or "results" not in data:
        raise ValueError(f"Invalid selected neuron artifact: {path}")
    return data


def _build_eval_data_cfg(cfg: DictConfig, ex_id: str) -> EvalDataConfig:
    eval_cfg = get_pipeline_step(cfg, "step4_identified_neurons_eval")
    tokenize_cfg = get_pipeline_step(cfg, "step1_tokenize")
    record_cfg = get_pipeline_step(cfg, "step2_record_activations")
    tokenized_dir = eval_cfg.get("tokenized_dir")
    if tokenized_dir:
        resolved = str(tokenized_dir)
    elif record_cfg.extend_load_dir_with_ex_name:
        resolved = os.path.join(tokenize_cfg.save_dir, ex_id)
    else:
        resolved = tokenize_cfg.save_dir
    return EvalDataConfig(
        tokenized_dir=resolved,
    )


def _load_language_tokens(
    lang: str,
    data_cfg: EvalDataConfig,
    target_num_tokens: int,
) -> torch.Tensor:
    return load_token_ids(
        lang=lang,
        tokenized_dir=data_cfg.tokenized_dir,
        target_num_tokens=target_num_tokens,
        missing_hint="Run 1_tokenize.py first or set pipeline.step4_identified_neurons_eval.tokenized_dir.",
    )


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

    for component_name, candidate_keys in COMPONENT_KEYS.items():
        matched = _first_present_component(selected_results, candidate_keys)
        if matched is None:
            continue
        component_key, component = matched
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
            if component_name == "mlp":
                width = layers[layer_idx].mlp.gate_proj.out_features
            elif component_name == "attn_q":
                width = layers[layer_idx].self_attn.q_proj.out_features
            elif component_name == "attn_k":
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
        masks[component_name] = layer_masks

    return masks


def _mask_output(_module: torch.nn.Module, _inputs: tuple[torch.Tensor], output: torch.Tensor, mask: torch.Tensor):
    return output * mask.to(device=output.device, dtype=output.dtype)


def _register_ablation_hooks(
    layers: list[torch.nn.Module],
    component_masks: dict[str, list[torch.Tensor | None]],
) -> list[Any]:
    handles: list[Any] = []

    for layer_idx, layer in enumerate(layers):
        mlp_mask = component_masks.get("mlp", [None] * len(layers))[layer_idx]
        if mlp_mask is not None:
            handles.append(
                layer.mlp.gate_proj.register_forward_hook(
                    lambda m, i, o, local_mask=mlp_mask: _mask_output(m, i, o, local_mask)
                )
            )

        q_mask = component_masks.get("attn_q", [None] * len(layers))[layer_idx]
        if q_mask is not None:
            handles.append(
                layer.self_attn.q_proj.register_forward_hook(
                    lambda m, i, o, local_mask=q_mask: _mask_output(m, i, o, local_mask)
                )
            )

        k_mask = component_masks.get("attn_k", [None] * len(layers))[layer_idx]
        if k_mask is not None:
            handles.append(
                layer.self_attn.k_proj.register_forward_hook(
                    lambda m, i, o, local_mask=k_mask: _mask_output(m, i, o, local_mask)
                )
            )

        v_mask = component_masks.get("attn_v", [None] * len(layers))[layer_idx]
        if v_mask is not None:
            handles.append(
                layer.self_attn.v_proj.register_forward_hook(
                    lambda m, i, o, local_mask=v_mask: _mask_output(m, i, o, local_mask)
                )
            )

    return handles


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    ex_id = set_ex_id_from_config_name()
    eval_cfg = get_pipeline_step(cfg, "step4_identified_neurons_eval")
    selected_path = _resolve_selected_neurons_path(cfg, ex_id)
    selected = _load_selected_neurons(selected_path)

    row_languages = list(selected.get("languages", [])) or list(cfg.main.languages)
    col_languages = list(eval_cfg.eval_languages) if eval_cfg.eval_languages else list(cfg.main.languages)

    device = get_device()
    print(f"Using device: {device}")
    model_dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg.main.model_path, dtype=model_dtype).to(device, non_blocking=True)
    model.eval()

    data_cfg = _build_eval_data_cfg(cfg, ex_id)
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
        ppx_before = compute_perplexity(
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
        if not handles:
            print(
                f"  WARNING: no ablation hooks registered for {ablation_lang}. "
                "Check selected neuron artifact keys and language coverage."
            )

        row_result: dict[str, float] = {}
        for eval_lang in col_languages:
            ppx_after = compute_perplexity(
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

    save_dir = os.path.join(eval_cfg.save_dir, ex_id)
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, "log_ppx_ratio_matrix.csv")
    save_matrix_csv(csv_path, row_languages, col_languages, matrix)
    png_path = os.path.join(save_dir, "log_ppx_ratio_matrix.png")
    save_heatmap_from_csv(csv_path, png_path)
    diag_metric = compute_diag_offdiag_metric(row_languages, col_languages, matrix)

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
