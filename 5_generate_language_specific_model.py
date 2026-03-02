import os
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from misc import set_ex_id_from_config_name


MLP_KEYS = (
    # "mlp_grad_average_activations",
    "mlp_over_threshold_rate",
)
ATTN_Q_KEYS = (
    # "attn_q_proj_grad_average_activations",
    "attn_q_proj_over_threshold_rate",
)
ATTN_K_KEYS = (
    # "attn_k_proj_grad_average_activations",
    "attn_k_proj_over_threshold_rate",
)
ATTN_V_KEYS = (
    # "attn_v_proj_grad_average_activations",
    "attn_v_proj_over_threshold_rate",
)

COMPONENT_KEYS = {
    "mlp": MLP_KEYS,
    "attn_q": ATTN_Q_KEYS,
    "attn_k": ATTN_K_KEYS,
    "attn_v": ATTN_V_KEYS,
}


def _resolve_generate_cfg(cfg: DictConfig) -> DictConfig:
    generate_cfg = cfg.identify_neurons.get("generate_language_specific_model")
    if generate_cfg is None:
        return OmegaConf.create({})
    return generate_cfg


def _resolve_selected_neurons_path(cfg: DictConfig, ex_id: str) -> str:
    generate_cfg = _resolve_generate_cfg(cfg)
    override_path = generate_cfg.get("selected_neurons_path")
    if override_path:
        return str(override_path)
    return os.path.join(
        cfg.identify_neurons.select_neurons.save_dir,
        ex_id,
        "lape_selected_neurons.pt",
    )


def _resolve_output_dir(cfg: DictConfig, ex_id: str) -> str:
    generate_cfg = _resolve_generate_cfg(cfg)
    base_dir = str(generate_cfg.get("save_dir", "data/5_output"))
    if bool(generate_cfg.get("extend_save_dir_with_ex_name", True)):
        return os.path.join(base_dir, ex_id)
    return base_dir


def _load_selected_neurons(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Selected neuron file not found: {path}")
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict) or "results" not in data:
        raise ValueError(f"Invalid selected neuron artifact: {path}")
    return data


def _find_model_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError("Unsupported model architecture: expected model.model.layers")


def _first_present_component(
    selected_results: dict[str, Any],
    candidate_keys: tuple[str, ...],
) -> tuple[str, dict[str, Any]] | None:
    for key in candidate_keys:
        value = selected_results.get(key)
        if isinstance(value, dict):
            return key, value
    return None


def _zero_linear_output_neurons(linear: torch.nn.Linear, indices: list[int]) -> int:
    if not indices:
        return 0
    width = linear.weight.shape[0]
    valid_idx = sorted({i for i in indices if 0 <= i < width})
    if not valid_idx:
        return 0
    with torch.no_grad():
        index_tensor = torch.as_tensor(valid_idx, device=linear.weight.device)
        linear.weight.index_fill_(0, index_tensor, 0)
        if linear.bias is not None:
            linear.bias.index_fill_(0, index_tensor, 0)
    return len(valid_idx)


def _ablate_language_specific_neurons(
    model: torch.nn.Module,
    selected: dict[str, Any],
    language: str,
) -> dict[str, int]:
    layers = _find_model_layers(model)
    selected_results = selected["results"]
    counts = {key: 0 for key in COMPONENT_KEYS}

    for component_name, candidate_keys in COMPONENT_KEYS.items():
        matched = _first_present_component(selected_results, candidate_keys)
        if matched is None:
            continue
        component_key, component = matched
        by_lang = component.get("selected_indices_by_language", {})
        per_layer_indices = by_lang.get(language)
        if per_layer_indices is None:
            raise ValueError(f"Language '{language}' not found in selected neuron artifact for {component_key}")
        if len(per_layer_indices) != len(layers):
            raise ValueError(
                f"Layer mismatch for {component_key}/{language}: "
                f"{len(per_layer_indices)} indices vs {len(layers)} model layers"
            )

        for layer_idx, idx_list in enumerate(per_layer_indices):
            if component_name == "mlp":
                counts[component_name] += _zero_linear_output_neurons(layers[layer_idx].mlp.gate_proj, idx_list)
            elif component_name == "attn_q":
                counts[component_name] += _zero_linear_output_neurons(layers[layer_idx].self_attn.q_proj, idx_list)
            elif component_name == "attn_k":
                counts[component_name] += _zero_linear_output_neurons(layers[layer_idx].self_attn.k_proj, idx_list)
            elif component_name == "attn_v":
                counts[component_name] += _zero_linear_output_neurons(layers[layer_idx].self_attn.v_proj, idx_list)

    return counts


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    ex_id = set_ex_id_from_config_name()
    generate_cfg = _resolve_generate_cfg(cfg)
    selected_path = _resolve_selected_neurons_path(cfg, ex_id)
    selected = _load_selected_neurons(selected_path)

    output_dir = _resolve_output_dir(cfg, ex_id)
    os.makedirs(output_dir, exist_ok=True)

    artifact_languages = list(selected.get("languages", []))
    config_languages = list(cfg.main.languages)
    languages = artifact_languages or config_languages
    if not languages:
        raise ValueError("No languages found in selected neuron artifact or cfg.main.languages")

    model_dtype = generate_cfg.get("torch_dtype")
    torch_dtype = None
    if model_dtype:
        dtype_name = str(model_dtype)
        torch_dtype = getattr(torch, dtype_name, None)
        if torch_dtype is None:
            raise ValueError(f"Unsupported generate_language_specific_model.torch_dtype: {dtype_name}")

    save_tokenizer = bool(generate_cfg.get("save_tokenizer", True))
    safe_serialization = bool(generate_cfg.get("safe_serialization", True))

    tokenizer = AutoTokenizer.from_pretrained(cfg.main.model_path) if save_tokenizer else None

    print(f"Loading selected neurons from: {selected_path}")
    print(f"Saving language-specific models under: {output_dir}")

    for language in languages:
        print(f"Generating ablated model for language: {language}")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.main.model_path,
            torch_dtype=torch_dtype,
        )
        counts = _ablate_language_specific_neurons(model, selected, language)

        language_dir = os.path.join(output_dir, language)
        os.makedirs(language_dir, exist_ok=True)
        model.save_pretrained(language_dir, safe_serialization=safe_serialization)
        if tokenizer is not None:
            tokenizer.save_pretrained(language_dir)

        total = sum(counts.values())
        print(
            f"  Saved to {language_dir} "
            f"(total ablated={total}, mlp={counts['mlp']}, q={counts['attn_q']}, "
            f"k={counts['attn_k']}, v={counts['attn_v']})"
        )


if __name__ == "__main__":
    main()
