import os
from collections import defaultdict
from typing import Any

import torch
from runexp import TrackedExperiment
from transformers import AutoModelForCausalLM, AutoTokenizer


COMPONENT_KEY_ALIASES = {
    "mlp": (
        "mlp_grad_average_activations",
        "mlp_over_threshold_rate",
    ),
    "attn_q": (
        "attn_q_proj_grad_average_activations",
        "attn_q_proj_over_threshold_rate",
    ),
    "attn_k": (
        "attn_k_proj_grad_average_activations",
        "attn_k_proj_over_threshold_rate",
    ),
    "attn_v": (
        "attn_v_proj_grad_average_activations",
        "attn_v_proj_over_threshold_rate",
    ),
}


class Experiment(TrackedExperiment):
    def __call__(self):
        trainer = self.config_built.trainer
        trainer.train()


def take(dataset, num_take):
    return dataset.take(num_take)


def shard_take(dataset, num_shards, index, num_take):
    return dataset.shard(num_shards=num_shards, index=index).take(num_take)


def load_tokenizer(model):
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def _load_selected_neurons(path: str) -> dict[str, Any]:
    resolved_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Selected neuron artifact not found: {resolved_path}")

    data = torch.load(resolved_path, map_location="cpu")
    if not isinstance(data, dict) or "results" not in data:
        raise ValueError(f"Invalid selected neuron artifact: {resolved_path}")
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


def _normalize_languages(raw_languages, artifact_languages: list[str], excluded_languages: set[str]) -> list[str]:
    if raw_languages is None:
        return [lang for lang in artifact_languages if lang not in excluded_languages]
    return [str(lang) for lang in raw_languages if str(lang) not in excluded_languages]


def _build_union_indices_by_component(
    selected: dict[str, Any],
    train_languages,
    exclude_languages,
) -> dict[str, list[list[int]]]:
    artifact_languages = [str(lang) for lang in selected.get("languages", [])]
    excluded_languages = {str(lang) for lang in (exclude_languages or [])}
    included_languages = _normalize_languages(train_languages, artifact_languages, excluded_languages)
    if not included_languages:
        raise ValueError("No train languages left after applying exclude_languages")

    selected_results = selected["results"]
    union_by_component: dict[str, list[list[int]]] = {}

    print(f"Train-language union will use: {included_languages}")
    if excluded_languages:
        print(f"Explicitly excluding languages from training mask: {sorted(excluded_languages)}")

    for component_name, candidate_keys in COMPONENT_KEY_ALIASES.items():
        matched = _first_present_component(selected_results, candidate_keys)
        if matched is None:
            continue

        component_key, component = matched
        by_lang = component.get("selected_indices_by_language", {})
        sample_lang = next((lang for lang in included_languages if lang in by_lang), None)
        if sample_lang is None:
            print(f"Skipping {component_key}: none of {included_languages} are present in artifact")
            continue

        num_layers = len(by_lang[sample_lang])
        layer_sets = [set() for _ in range(num_layers)]

        for lang in included_languages:
            per_layer_indices = by_lang.get(lang)
            if per_layer_indices is None:
                raise ValueError(f"Language '{lang}' not found in selected neuron artifact for {component_key}")
            if len(per_layer_indices) != num_layers:
                raise ValueError(
                    f"Layer mismatch for {component_key}/{lang}: {len(per_layer_indices)} indices vs {num_layers}"
                )
            for layer_idx, indices in enumerate(per_layer_indices):
                layer_sets[layer_idx].update(int(idx) for idx in indices)

        for lang in excluded_languages:
            per_layer_indices = by_lang.get(lang)
            if per_layer_indices is None:
                continue
            if len(per_layer_indices) != num_layers:
                raise ValueError(
                    f"Layer mismatch for excluded language {component_key}/{lang}: "
                    f"{len(per_layer_indices)} indices vs {num_layers}"
                )
            for layer_idx, indices in enumerate(per_layer_indices):
                layer_sets[layer_idx].difference_update(int(idx) for idx in indices)

        union_by_component[component_name] = [sorted(layer_indices) for layer_indices in layer_sets]
        total = sum(len(layer_indices) for layer_indices in union_by_component[component_name])
        print(f"Component {component_name} -> {total} trainable neuron indices across {num_layers} layers")

    if not union_by_component:
        raise ValueError("No component masks were built from the selected neuron artifact")

    return union_by_component


def _register_mask_hook(param: torch.nn.Parameter, allowed_indices: list[int], dim: int) -> int:
    if param is None or not allowed_indices:
        return 0

    axis_size = param.shape[dim]
    valid_indices = sorted({idx for idx in allowed_indices if 0 <= idx < axis_size})
    if not valid_indices:
        return 0

    index_tensor = torch.tensor(valid_indices, dtype=torch.long)

    def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
        if grad is None:
            return grad

        local_index = index_tensor.to(device=grad.device)
        masked = torch.zeros_like(grad)
        if dim == 0:
            masked[local_index] = grad[local_index]
        elif dim == 1:
            masked[:, local_index] = grad[:, local_index]
        else:
            raise ValueError(f"Unsupported gradient mask dimension: {dim}")
        return masked

    param.requires_grad_(True)
    param.register_hook(_mask_grad)
    return len(valid_indices)


def _apply_trainable_masks(
    model: torch.nn.Module,
    union_by_component: dict[str, list[list[int]]],
) -> dict[str, int]:
    layers = _find_model_layers(model)
    counts = defaultdict(int)

    for param in model.parameters():
        param.requires_grad_(False)

    for layer_idx, layer in enumerate(layers):
        mlp_indices = union_by_component.get("mlp", [])
        if layer_idx < len(mlp_indices) and mlp_indices[layer_idx]:
            indices = mlp_indices[layer_idx]
            counts["mlp_gate_rows"] += _register_mask_hook(layer.mlp.gate_proj.weight, indices, dim=0)
            if getattr(layer.mlp.gate_proj, "bias", None) is not None:
                counts["mlp_gate_bias"] += _register_mask_hook(layer.mlp.gate_proj.bias, indices, dim=0)
            counts["mlp_up_rows"] += _register_mask_hook(layer.mlp.up_proj.weight, indices, dim=0)
            if getattr(layer.mlp.up_proj, "bias", None) is not None:
                counts["mlp_up_bias"] += _register_mask_hook(layer.mlp.up_proj.bias, indices, dim=0)
            counts["mlp_down_cols"] += _register_mask_hook(layer.mlp.down_proj.weight, indices, dim=1)

        for component_name, proj_name in (("attn_q", "q_proj"), ("attn_k", "k_proj"), ("attn_v", "v_proj")):
            component_indices = union_by_component.get(component_name, [])
            if layer_idx >= len(component_indices) or not component_indices[layer_idx]:
                continue
            indices = component_indices[layer_idx]
            proj = getattr(layer.self_attn, proj_name)
            counts[f"{component_name}_rows"] += _register_mask_hook(proj.weight, indices, dim=0)
            if getattr(proj, "bias", None) is not None:
                counts[f"{component_name}_bias"] += _register_mask_hook(proj.bias, indices, dim=0)

    total_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Trainable parameter tensors enabled: {sum(param.requires_grad for param in model.parameters())}")
    print(f"Trainable parameters: {total_trainable}/{total_params} ({100 * total_trainable / total_params:.6f}%)")
    for name in sorted(counts):
        print(f"  - {name}: {counts[name]} selected indices")

    return dict(counts)


def load_masked_model(
    model_name: str,
    selected_neurons_path: str,
    train_languages=None,
    exclude_languages=None,
    dtype=None,
    torch_dtype=None,
):
    resolved_dtype = torch_dtype if torch_dtype is not None else dtype
    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=resolved_dtype,
    )
    model.config.use_cache = False

    if selected_neurons_path == "":
        print("No selected neuron artifact provided, finetuning whole model.")
        for param in model.parameters():
            param.requires_grad_(True)
        return model

    print(f"Loading selected neuron artifact: {selected_neurons_path}")
    selected = _load_selected_neurons(selected_neurons_path)
    union_by_component = _build_union_indices_by_component(
        selected=selected,
        train_languages=train_languages,
        exclude_languages=exclude_languages,
    )
    _apply_trainable_masks(model, union_by_component)
    return model
