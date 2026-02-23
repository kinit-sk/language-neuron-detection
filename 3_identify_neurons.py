import os
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, ListConfig


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
    components = cfg.identify_neurons.record_activations.get("components", ["mlp", "attn"])
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
    top_k = max(1, int(width * top_rate))
    eps = 1e-12
    total_neurons = n_layers * width
    print(f"Total neurons before filtration: {total_neurons}")

    # Pre-filter globally low-activity neurons using a kthvalue threshold, then
    # keep a neuron slot (layer, hidden) if at least one language exceeds it.
    flat = stacked.flatten()
    k = int(round(flat.numel() * filter_rate))
    k = min(max(k, 1), flat.numel())
    prefilter_bar = flat.kthvalue(k).values
    prefilter_mask = (stacked > prefilter_bar).any(dim=0, keepdim=True)
    neurons_after_filter_rate = int(prefilter_mask.sum().item())
    import code; code.interact(local=dict(globals(), **locals()))
    print(f"Neurons after filter_rate + kthvalue filtration: {neurons_after_filter_rate}")

    lang_sum = stacked.sum(dim=0, keepdim=True) + eps
    lang_ratio = stacked / lang_sum

    bars = torch.quantile(stacked, q=activation_bar_ratio, dim=2, keepdim=True)
    strong_mask = stacked >= bars
    specific_mask = lang_ratio >= filter_rate
    candidate_mask = strong_mask & specific_mask & prefilter_mask

    score = stacked * lang_ratio
    masked_score = torch.where(candidate_mask, score, torch.full_like(score, -1.0))

    top_scores, top_indices = torch.topk(masked_score, k=top_k, dim=2, largest=True, sorted=True)
    top_valid = top_scores >= 0.0

    selected_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
    selected_mask.scatter_(2, top_indices, top_valid)
    neurons_after_top_rate = int(selected_mask.any(dim=0).sum().item())
    print(f"Neurons after top_rate filtration: {neurons_after_top_rate}")

    return {
        "selected_mask": selected_mask,
        "top_indices": top_indices,
        "top_scores": torch.where(top_valid, top_scores, torch.zeros_like(top_scores)),
        "top_valid": top_valid,
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

    return counts_per_lang, total_neurons_per_lang


def _resolve_filter_rate(cfg: DictConfig, recording_strategy: str) -> float:
    select_cfg = cfg.identify_neurons.select_neurons
    by_strategy = select_cfg.get("filter_rate_by_strategy", None)
    if by_strategy is not None and recording_strategy in by_strategy:
        return float(by_strategy[recording_strategy])
    return float(select_cfg.filter_rate)


def _resolve_lape_metric(cfg: DictConfig) -> str:
    raw = str(cfg.identify_neurons.select_neurons.get("recorded_metric", "grad_average_activations")).strip().lower()
    valid = {"grad_average_activations", "over_threshold_rate"}
    if raw not in valid:
        raise ValueError(
            "identify_neurons.select_neurons.recorded_metric must be one of: "
            "'grad_average_activations', 'over_threshold_rate'"
        )
    return raw


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    load_dir = os.path.join(cfg.identify_neurons.record_activations.save_dir, cfg.main.ex_id)
    if not os.path.isdir(load_dir):
        raise FileNotFoundError(f"Activation directory not found: {load_dir}")
    save_dir = os.path.join(cfg.identify_neurons.select_neurons.save_dir, cfg.main.ex_id)
    os.makedirs(save_dir, exist_ok=True)

    languages = list(cfg.main.languages)
    recording_strategy = cfg.identify_neurons.record_activations.get("recording_strategy", "grad_act")
    top_rate = float(cfg.identify_neurons.select_neurons.top_rate)
    filter_rate = _resolve_filter_rate(cfg, recording_strategy)
    activation_bar_ratio = float(cfg.identify_neurons.select_neurons.activation_bar_ratio)
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
        lape = _lape_select(
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
