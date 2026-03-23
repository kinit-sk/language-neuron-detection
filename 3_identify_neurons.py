import os
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, ListConfig

from misc import get_pipeline_step, set_ex_id_from_config_name


def _safe_abs(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return x.abs()


def _load_language_activation(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Activation file not found: {path}")
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"Invalid activation file format: {path}")
    return data


def _resolve_activation_path(load_dir: str, lang: str, recording_strategy: str) -> str:
    strategy_path = os.path.join(load_dir, f"{lang}_{recording_strategy}.pt")
    if os.path.exists(strategy_path):
        return strategy_path
    legacy_path = os.path.join(load_dir, f"{lang}.pt")
    if os.path.exists(legacy_path):
        return legacy_path
    return strategy_path


def _resolve_record_components(cfg: DictConfig) -> tuple[bool, bool]:
    record_cfg = get_pipeline_step(cfg, "step2_record_activations")
    components = record_cfg.get("components", ["mlp", "attn"])
    if isinstance(components, str):
        components = [components]
    elif isinstance(components, ListConfig):
        components = list(components)

    if not isinstance(components, (list, tuple)) or not components:
        raise ValueError("record_activations.components must be a non-empty list containing 'mlp' and/or 'attn'")
    normalized = {str(c).strip().lower() for c in components}
    invalid = normalized - {"mlp", "attn"}
    if invalid:
        raise ValueError(f"Unsupported record_activations.components values: {sorted(invalid)}")
    include_mlp = "mlp" in normalized
    include_attn = "attn" in normalized
    return include_mlp, include_attn


def _collect_activation_tensors(
    load_dir: str,
    languages: list[str],
    recording_strategy: str,
    lape_metric: str,
    include_mlp: bool,
    include_attn: bool,
) -> dict[str, torch.Tensor]:
    by_lang: dict[str, dict[str, torch.Tensor]] = {}
    common_keys: set[str] | None = None

    for lang in languages:
        data = _load_language_activation(_resolve_activation_path(load_dir, lang, recording_strategy))
        lang_tensors: dict[str, torch.Tensor] = {}

        if include_mlp:
            mlp_key = "mlp_grad_average_activations" if lape_metric == "grad_average_activations" else "mlp_over_threshold_rate"
            mlp = data.get(mlp_key)
            if torch.is_tensor(mlp):
                lang_tensors[f"mlp_{lape_metric}"] = _safe_abs(mlp)

        if include_attn:
            attn_key = "attn_grad_average_activations" if lape_metric == "grad_average_activations" else "attn_over_threshold_rate"
            attn = data.get(attn_key)
            if isinstance(attn, dict):
                for proj_name, proj_tensor in attn.items():
                    if torch.is_tensor(proj_tensor):
                        proj = "q_proj" if proj_name.startswith("q_proj") else ("k_proj" if proj_name.startswith("k_proj") else ("v_proj" if proj_name.startswith("v_proj") else proj_name))
                        lang_tensors[f"attn_{proj}_{lape_metric}"] = _safe_abs(proj_tensor)

        if not lang_tensors:
            raise ValueError(f"No valid activation tensors found for language: {lang}")

        by_lang[lang] = lang_tensors
        lang_keys = set(lang_tensors.keys())
        common_keys = lang_keys if common_keys is None else (common_keys & lang_keys)

    if not common_keys:
        raise ValueError("No common activation tensor keys across all languages")

    stacked: dict[str, torch.Tensor] = {}
    for key in sorted(common_keys):
        tensors = [by_lang[lang][key] for lang in languages]
        if any(t.shape != tensors[0].shape for t in tensors):
            raise ValueError(f"Shape mismatch for key '{key}' across languages")
        stacked[key] = torch.stack(tensors, dim=0).contiguous()

    return stacked


def _lape_select(
    stacked: torch.Tensor,
    top_rate: float,
    filter_rate: float,
    activation_bar_ratio: float,
) -> dict[str, torch.Tensor]:
    n_lang, n_layers, width = stacked.shape
    activation_probs = stacked.permute(1, 2, 0).contiguous()  # layer x hidden x lang
    largest = False

    normed_activation_probs = activation_probs / activation_probs.sum(dim=-1, keepdim=True)
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0
    log_probs = torch.where(
        normed_activation_probs > 0,
        normed_activation_probs.log(),
        torch.zeros_like(normed_activation_probs),
    )
    entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)

    if torch.isnan(entropy).sum():
        raise ValueError("NaN values found in entropy")

    flattened_probs = activation_probs.flatten()
    top_prob_k = round(len(flattened_probs) * filter_rate)
    top_prob_value = flattened_probs.kthvalue(top_prob_k).values.item()
    top_position = (activation_probs > top_prob_value).sum(dim=-1)
    entropy[top_position == 0] = -torch.inf if largest else torch.inf

    flattened_entropy = entropy.flatten()
    top_entropy_value = round(len(flattened_entropy) * top_rate)
    _, index = flattened_entropy.topk(top_entropy_value, largest=largest)
    row_index = index // entropy.size(1)
    col_index = index % entropy.size(1)
    selected_probs = activation_probs[row_index, col_index]  # n_selected x lang

    selected_probs_by_lang = selected_probs.transpose(0, 1)  # lang x n_selected
    activation_bar_k = round(len(flattened_probs) * activation_bar_ratio)
    activation_bar = flattened_probs.kthvalue(activation_bar_k).values.item()
    lang, indice = torch.where(selected_probs_by_lang > activation_bar)
    merged_index = torch.stack((row_index, col_index), dim=-1)

    selected_mask = torch.zeros((n_lang, n_layers, width), dtype=torch.bool)
    counts_by_lang = torch.bincount(lang, minlength=n_lang)
    for lang_idx, split_idx in enumerate(indice.split(counts_by_lang.tolist())):
        if split_idx.numel() == 0:
            continue
        lang_index = [tuple(row.tolist()) for row in merged_index[split_idx]]
        lang_index.sort()
        for layer_idx, hidden_idx in lang_index:
            selected_mask[lang_idx, layer_idx, hidden_idx] = True

    return {
        "selected_mask": selected_mask,
        "top_indices": merged_index,
        "top_scores": selected_probs,
        "top_valid": selected_probs_by_lang > activation_bar,
        "counts_per_lang_layer": selected_mask.sum(dim=2),
    }



def _random_select_from_most_active(
    stacked: torch.Tensor,
    top_rate: float,
    filter_rate: float,
    activation_bar_ratio: float,
) -> dict[str, torch.Tensor]:
    n_lang, n_layers, width = stacked.shape
    activation_probs = stacked.permute(1, 2, 0).contiguous()  # layer x hidden x lang

    flattened_probs = activation_probs.flatten()
    top_prob_k = round(len(flattened_probs) * filter_rate)
    top_prob_value = flattened_probs.kthvalue(top_prob_k).values.item()
    candidate_mask = (activation_probs > top_prob_value).any(dim=-1)
    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False)

    if candidate_indices.numel() == 0:
        empty_scores = activation_probs.new_zeros((0, n_lang))
        empty_valid = torch.zeros((n_lang, 0), dtype=torch.bool)
        empty_mask = torch.zeros((n_lang, n_layers, width), dtype=torch.bool)
        return {
            "selected_mask": empty_mask,
            "top_indices": candidate_indices,
            "top_scores": empty_scores,
            "top_valid": empty_valid,
            "counts_per_lang_layer": empty_mask.sum(dim=2),
        }

    n_selected = min(round((n_layers * width) * top_rate), candidate_indices.size(0))
    perm = torch.randperm(candidate_indices.size(0))[:n_selected]
    merged_index = candidate_indices[perm]
    row_index = merged_index[:, 0]
    col_index = merged_index[:, 1]
    selected_probs = activation_probs[row_index, col_index]  # n_selected x lang

    selected_probs_by_lang = selected_probs.transpose(0, 1)  # lang x n_selected
    activation_bar_k = round(len(flattened_probs) * activation_bar_ratio)
    activation_bar = flattened_probs.kthvalue(activation_bar_k).values.item()
    lang, indice = torch.where(selected_probs_by_lang > activation_bar)

    selected_mask = torch.zeros((n_lang, n_layers, width), dtype=torch.bool)
    counts_by_lang = torch.bincount(lang, minlength=n_lang)
    for lang_idx, split_idx in enumerate(indice.split(counts_by_lang.tolist())):
        if split_idx.numel() == 0:
            continue
        lang_index = merged_index[split_idx]
        selected_mask[lang_idx, lang_index[:, 0], lang_index[:, 1]] = True

    return {
        "selected_mask": selected_mask,
        "top_indices": merged_index,
        "top_scores": selected_probs,
        "top_valid": selected_probs_by_lang > activation_bar,
        "counts_per_lang_layer": selected_mask.sum(dim=2),
    }


def _build_serializable_index_lists(selected_mask: torch.Tensor, languages: list[str]) -> dict[str, list[list[int]]]:
    out: dict[str, list[list[int]]] = {}
    for lang_idx, lang in enumerate(languages):
        per_layer: list[list[int]] = []
        for layer_idx in range(selected_mask.size(1)):
            indices = torch.nonzero(selected_mask[lang_idx, layer_idx], as_tuple=False).squeeze(1).tolist()
            per_layer.append(indices)
        out[lang] = per_layer
    return out


def _log_selection_stats(key: str, selected_mask: torch.Tensor, languages: list[str]) -> tuple[torch.Tensor, int]:
    n_layers = selected_mask.size(1)
    width = selected_mask.size(2)
    total_neurons_per_lang = n_layers * width
    counts_per_lang = selected_mask.sum(dim=(1, 2))

    print(f"[{key}] language-specific neuron stats:")
    for lang_idx, lang in enumerate(languages):
        count = int(counts_per_lang[lang_idx].item())
        pct = (count / total_neurons_per_lang) * 100.0 if total_neurons_per_lang > 0 else 0.0
        print(f"  - {lang}: {count}/{total_neurons_per_lang} ({pct:.2f}%)")

    counts_per_layer = selected_mask.sum(dim=(0, 2))
    total_selected = int(counts_per_layer.sum().item())
    print(f"[{key}] selected neuron layer distribution (all languages):")
    if total_selected == 0:
        print("  - no neurons selected")
    else:
        for layer_idx, layer_count_tensor in enumerate(counts_per_layer):
            layer_count = int(layer_count_tensor.item())
            pct = (layer_count / total_selected) * 100.0
            print(f"  - layer {layer_idx}: {layer_count}/{total_selected} ({pct:.2f}%)")

    return counts_per_lang, total_neurons_per_lang


def _resolve_filter_rate(cfg: DictConfig, recording_strategy: str) -> float:
    identify_cfg = get_pipeline_step(cfg, "step3_identify_neurons")
    by_strategy = identify_cfg.get("filter_rate_by_strategy", None)
    if by_strategy is not None and recording_strategy in by_strategy:
        return float(by_strategy[recording_strategy])
    return float(identify_cfg.filter_rate)


def _resolve_lape_metric(cfg: DictConfig) -> str:
    identify_cfg = get_pipeline_step(cfg, "step3_identify_neurons")
    raw = str(identify_cfg.get("recorded_metric", "grad_average_activations")).strip().lower()
    valid = {"grad_average_activations", "over_threshold_rate"}
    if raw not in valid:
        raise ValueError(
            "pipeline.step3_identify_neurons.recorded_metric must be one of: "
            "'grad_average_activations', 'over_threshold_rate'"
        )
    return raw


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    ex_id = set_ex_id_from_config_name()
    record_cfg = get_pipeline_step(cfg, "step2_record_activations")
    identify_cfg = get_pipeline_step(cfg, "step3_identify_neurons")

    use_activations_from_step_2 = identify_cfg.get("use_activations_from_step_2", True)

    load_dir = os.path.join(record_cfg.save_dir, ex_id)
    if not os.path.isdir(load_dir):
        raise FileNotFoundError(f"Activation directory not found: {load_dir}")
    save_dir = os.path.join(identify_cfg.save_dir, ex_id)
    os.makedirs(save_dir, exist_ok=True)

    languages = list(cfg.main.languages)
    recording_strategy = record_cfg.get("recording_strategy", "grad_act")
    top_rate = float(identify_cfg.top_rate)
    filter_rate = _resolve_filter_rate(cfg, recording_strategy)
    activation_bar_ratio = float(identify_cfg.activation_bar_ratio)
    lape_metric = _resolve_lape_metric(cfg)
    include_mlp, include_attn = _resolve_record_components(cfg)

    stacked_by_key = _collect_activation_tensors(
        load_dir,
        languages,
        recording_strategy,
        lape_metric,
        include_mlp=include_mlp,
        include_attn=include_attn,
    )

    results: dict[str, dict[str, Any]] = {}
    aggregate_counts_per_lang = torch.zeros(len(languages), dtype=torch.long)
    aggregate_total_neurons_per_lang = 0
    for key, stacked in stacked_by_key.items():
        if use_activations_from_step_2:
            print("Using activations from step 2")
            lape = _lape_select(
                stacked=stacked,
                top_rate=top_rate,
                filter_rate=filter_rate,
                activation_bar_ratio=activation_bar_ratio,
            )
        else:
            print("Selecting neurons randomly")
            lape = _random_select_from_most_active(
                stacked=stacked,
                top_rate=top_rate,
                filter_rate=filter_rate,
                activation_bar_ratio=activation_bar_ratio,
            )
        results[key] = {
            "selected_mask": lape["selected_mask"],
            "counts_per_lang_layer": lape["counts_per_lang_layer"],
            "top_indices": lape["top_indices"],
            "top_scores": lape["top_scores"],
            "top_valid": lape["top_valid"],
            "selected_indices_by_language": _build_serializable_index_lists(lape["selected_mask"], languages),
        }
        key_counts, key_total = _log_selection_stats(key, lape["selected_mask"], languages)
        aggregate_counts_per_lang += key_counts.to(torch.long)
        aggregate_total_neurons_per_lang += key_total

    print("[overall] language-specific neuron stats across all activation groups:")
    for lang_idx, lang in enumerate(languages):
        count = int(aggregate_counts_per_lang[lang_idx].item())
        pct = (count / aggregate_total_neurons_per_lang) * 100.0 if aggregate_total_neurons_per_lang > 0 else 0.0
        print(f"  - {lang}: {count}/{aggregate_total_neurons_per_lang} ({pct:.2f}%)")

    out = {
        "method": "LAPE",
        "languages": languages,
        "params": {
            "recording_strategy": recording_strategy,
            "components": [c for c, enabled in (("mlp", include_mlp), ("attn", include_attn)) if enabled],
            "top_rate": top_rate,
            "filter_rate": filter_rate,
            "activation_bar_ratio": activation_bar_ratio,
            "recorded_metric": lape_metric,
        },
        "results": results,
    }

    save_path = os.path.join(save_dir, "lape_selected_neurons.pt")
    torch.save(out, save_path)
    print(f"Saved selected neurons to {save_path}")


if __name__ == "__main__":
    main()
