import os
import types
from functools import partial
from typing import Optional

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from misc import get_device, get_pipeline_step, set_ex_id_from_config_name


def _safe_float_tensor(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if not torch.isfinite(x).all():
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def _init_stats(num_layers: int, width: int) -> dict[str, torch.Tensor]:
    return {
        "sums": torch.zeros(num_layers, width, dtype=torch.float32),
        "over_threshold": torch.zeros(num_layers, width, dtype=torch.float32),
        "counts": torch.zeros(num_layers, dtype=torch.int64),
    }


def _resolve_record_components(cfg: DictConfig) -> tuple[bool, bool]:
    record_cfg = get_pipeline_step(cfg, "step2_record_activations")
    components = record_cfg.get("components", ["mlp", "attn"])
    if isinstance(components, str):
        components = [components]
    if isinstance(components, ListConfig):
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


def _resolve_recording_strategy(cfg: DictConfig) -> str:
    record_cfg = get_pipeline_step(cfg, "step2_record_activations")
    recording_strategy = str(record_cfg.get("recording_strategy", "grad_act")).strip().lower()
    if recording_strategy not in {"grad_act", "act"}:
        raise ValueError(
            f"Unsupported recording_strategy='{recording_strategy}'. "
            "Supported strategies are: 'grad_act', 'act'."
        )
    return recording_strategy


def _resolve_mlp_variant(cfg: DictConfig) -> str:
    record_cfg = get_pipeline_step(cfg, "step2_record_activations")
    variant_raw = str(record_cfg.variant).strip().lower()
    if variant_raw not in {"gate", "gate_up"}:
        raise ValueError(
            "record_activations.variant must be one of: 'gate', 'gate_up'"
        )
    return variant_raw


def _resolve_activation_thresholds(cfg: DictConfig) -> tuple[float, float]:
    record_cfg = get_pipeline_step(cfg, "step2_record_activations")
    mlp_threshold = float(record_cfg.get("mlp_activation_threshold", 0.0))
    attn_threshold = float(record_cfg.get("attn_activation_threshold", 0.0))
    return mlp_threshold, attn_threshold


def _accumulate_act_grad(
    stats: dict[str, torch.Tensor], layer_idx: int, act: torch.Tensor, grad: torch.Tensor, threshold: float
):
    act_clean = _safe_float_tensor(act.detach())
    metric = (act_clean * _safe_float_tensor(grad)).abs()
    with torch.no_grad():
        stats["sums"][layer_idx] += metric.sum(dim=(0, 1)).cpu()
        stats["over_threshold"][layer_idx] += (metric > threshold).sum(dim=(0, 1)).cpu()
        stats["counts"][layer_idx] += metric.size(0) * metric.size(1)


def _accumulate_act(stats: dict[str, torch.Tensor], layer_idx: int, act: torch.Tensor, threshold: float):
    metric = _safe_float_tensor(act.detach()).abs()
    with torch.no_grad():
        stats["sums"][layer_idx] += metric.sum(dim=(0, 1)).cpu()
        stats["over_threshold"][layer_idx] += (metric > threshold).sum(dim=(0, 1)).cpu()
        stats["counts"][layer_idx] += metric.size(0) * metric.size(1)


def _record_mlp(
    stats: dict[str, torch.Tensor],
    layer_idx: int,
    tracked: torch.Tensor,
    recording_strategy: str,
    threshold: float,
    mlp_variant: str,
):
    if recording_strategy == "grad_act":
        def grad_hook_fn(grad):
            _accumulate_act_grad(stats, layer_idx, tracked, grad, threshold)

        tracked.register_hook(grad_hook_fn)
        return

    if recording_strategy == "act":
        with torch.no_grad():
            if mlp_variant == "gate":
                tracked_clean = _safe_float_tensor(tracked.detach())
                stats["sums"][layer_idx] += tracked_clean.sum(dim=(0, 1)).cpu()
                stats["over_threshold"][layer_idx] += (tracked > 0).sum(dim=(0, 1)).cpu()
                stats["counts"][layer_idx] += tracked.size(0) * tracked.size(1)
            else:
                # The same as the first variant only using abs()
                # because tracked is not after Silu here.
                _accumulate_act(stats, layer_idx, tracked, threshold)
        return

    raise ValueError(
        f"Unsupported recording_strategy='{recording_strategy}'. "
        "Supported strategies are: 'grad_act', 'act'."
    )


def _record_attn(
    stats: dict[str, torch.Tensor],
    layer_idx: int,
    output: torch.Tensor,
    recording_strategy: str,
    threshold: float,
):
    if recording_strategy == "grad_act":
        def grad_hook_fn(grad):
            _accumulate_act_grad(stats, layer_idx, output, grad, threshold)

        output.register_hook(grad_hook_fn)
        return

    if recording_strategy == "act":
        _accumulate_act(stats, layer_idx, output, threshold)
        return

    raise ValueError(
        f"Unsupported recording_strategy='{recording_strategy}'. "
        "Supported strategies are: 'grad_act', 'act'."
    )


def register_mlp_patched_forward(
    mlp: torch.nn.Module,
    layer_idx: int,
    mlp_grad_stats: dict[str, torch.Tensor],
    recording_strategy: str,
    mlp_variant: str,
    mlp_activation_threshold: float,
):
    original_forward = mlp.forward

    def patched_forward(self, x):
        gate = self.gate_proj(x)
        gate_act = F.silu(gate)
        up = self.up_proj(x)
        gated = gate_act * up
        tracked = gate_act if mlp_variant == "gate" else gated

        _record_mlp(
            mlp_grad_stats,
            layer_idx,
            tracked,
            recording_strategy,
            mlp_activation_threshold,
            mlp_variant,
        )
        return self.down_proj(gated)

    mlp.forward = types.MethodType(patched_forward, mlp)
    return original_forward


def register_attn_hooks(
    layer: torch.nn.Module,
    layer_idx: int,
    attn_grad_stats: dict[str, dict[str, torch.Tensor]],
    recording_strategy: str,
    attn_activation_threshold: float,
):
    def attn_hook_fn(module_name: str, _layer_idx: int, _module: torch.nn.Module, _inputs, output: torch.Tensor):
        _record_attn(attn_grad_stats[module_name], _layer_idx, output, recording_strategy, attn_activation_threshold)

    return [
        layer.self_attn.q_proj.register_forward_hook(partial(attn_hook_fn, "q_proj", layer_idx)),
        layer.self_attn.k_proj.register_forward_hook(partial(attn_hook_fn, "k_proj", layer_idx)),
        layer.self_attn.v_proj.register_forward_hook(partial(attn_hook_fn, "v_proj", layer_idx)),
    ]


def attach_hooks(
    model: torch.nn.Module,
    mlp_grad_stats: dict[str, torch.Tensor] | None,
    attn_grad_stats: dict[str, dict[str, torch.Tensor]] | None,
    include_mlp: bool,
    include_attn: bool,
    recording_strategy: str,
    mlp_variant: Optional[str],
    mlp_activation_threshold: float,
    attn_activation_threshold: float,
):
    handles = []
    original_mlp_forwards = []
    for i, layer in enumerate(model.model.layers):
        if include_mlp:
            if mlp_grad_stats is None:
                raise ValueError("MLP stats are required when include_mlp=True")
            if mlp_variant is None:
                raise ValueError("MLP variant must be provided when include_mlp=True")
            original_mlp_forward = register_mlp_patched_forward(
                layer.mlp,
                i,
                mlp_grad_stats,
                recording_strategy,
                mlp_variant,
                mlp_activation_threshold,
            )
            original_mlp_forwards.append((layer.mlp, original_mlp_forward))

        if include_attn:
            if attn_grad_stats is None:
                raise ValueError("Attention stats are required when include_attn=True")
            if not hasattr(layer, "self_attn") or not hasattr(layer.self_attn, "q_proj") or not hasattr(layer.self_attn, "k_proj") or not hasattr(layer.self_attn, "v_proj"):
                print(f"Skipping layer {i} because it does not have q, k, v projections")
                continue
            handles.extend(register_attn_hooks(layer, i, attn_grad_stats, recording_strategy, attn_activation_threshold))

    return handles, original_mlp_forwards


def _compute_average(stats: dict[str, torch.Tensor]) -> torch.Tensor:
    counts = stats["counts"].unsqueeze(1).float()
    return torch.where(
        counts > 0,
        # stats["sums"], # / counts # we dont divide by counts because grads are tiny
        stats["sums"] / counts, # we dont divide by counts because grads are tiny
        torch.zeros_like(stats["sums"]),
    )


def _compute_over_threshold_rate(stats: dict[str, torch.Tensor]) -> torch.Tensor:
    counts = stats["counts"].unsqueeze(1).float()
    return torch.where(
        counts > 0,
        stats["over_threshold"] / counts,
        torch.zeros_like(stats["over_threshold"]),
    )


def save_activations(
    mlp_grad_stats: dict[str, torch.Tensor] | None,
    attn_grad_stats: dict[str, dict[str, torch.Tensor]] | None,
    lang: str,
    save_dir: str,
    size: int,
    recording_strategy: str,
    mlp_activation_threshold: float,
    attn_activation_threshold: float,
    include_mlp: bool,
    include_attn: bool,
):
    output = {
        "n": size,
        "recording_strategy": recording_strategy,
        "components": [c for c, enabled in (("mlp", include_mlp), ("attn", include_attn)) if enabled],
        "mlp_activation_threshold": mlp_activation_threshold,
        "attn_activation_threshold": attn_activation_threshold,
    }
    if include_mlp:
        if mlp_grad_stats is None:
            raise ValueError("MLP stats are required when include_mlp=True")
        output["mlp_grad_average_activations"] = _compute_average(mlp_grad_stats).to("cpu")
        output["mlp_over_threshold_rate"] = _compute_over_threshold_rate(mlp_grad_stats).to("cpu")
    if include_attn:
        if attn_grad_stats is None:
            raise ValueError("Attention stats are required when include_attn=True")
        attn_grad_average = {
            "q_proj_average": _compute_average(attn_grad_stats["q_proj"]),
            "k_proj_average": _compute_average(attn_grad_stats["k_proj"]),
            "v_proj_average": _compute_average(attn_grad_stats["v_proj"]),
        }
        attn_over_threshold_rate = {
            "q_proj_rate": _compute_over_threshold_rate(attn_grad_stats["q_proj"]),
            "k_proj_rate": _compute_over_threshold_rate(attn_grad_stats["k_proj"]),
            "v_proj_rate": _compute_over_threshold_rate(attn_grad_stats["v_proj"]),
        }
        output["attn_grad_average_activations"] = {k: v.to("cpu") for k, v in attn_grad_average.items()}
        output["attn_over_threshold_rate"] = {k: v.to("cpu") for k, v in attn_over_threshold_rate.items()}

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{lang}_{recording_strategy}.pt")
    torch.save(output, save_path)
    print(f"Saved activations to {save_path}/\n")


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    ex_id = set_ex_id_from_config_name()
    tokenize_cfg = get_pipeline_step(cfg, "step1_tokenize")
    record_cfg = get_pipeline_step(cfg, "step2_record_activations")
    if record_cfg.extend_load_dir_with_ex_name:
        tokenize_path = os.path.join(tokenize_cfg.save_dir, ex_id)
    else:
        tokenize_path = tokenize_cfg.save_dir
    if not os.path.exists(tokenize_path):
        raise FileNotFoundError("Tokenized data directory does not exist: ensure tokenization step is completed first")

    device = get_device()
    print(f"Using device: {device}")
    model = AutoModelForCausalLM.from_pretrained(cfg.main.model_path, dtype=torch.float16).to(device, non_blocking=True)
    model.eval()

    chunk_size = record_cfg.chunk_size
    batch_size = record_cfg.batch_size
    recording_strategy = _resolve_recording_strategy(cfg)
    mlp_activation_threshold, attn_activation_threshold = _resolve_activation_thresholds(cfg)
    include_mlp, include_attn = _resolve_record_components(cfg)
    if not include_mlp and not include_attn:
        raise ValueError("record_activations.components must include at least one of: 'mlp', 'attn'")
    mlp_variant = _resolve_mlp_variant(cfg) if include_mlp else None
    num_layers = len(model.model.layers)
    intermediate_size = model.model.layers[0].mlp.gate_proj.out_features if include_mlp else None
    q_size = model.model.layers[0].self_attn.q_proj.out_features if include_attn else None
    k_size = model.model.layers[0].self_attn.k_proj.out_features if include_attn else None
    v_size = model.model.layers[0].self_attn.v_proj.out_features if include_attn else None

    for lang in cfg.main.languages:
        mlp_grad_stats = _init_stats(num_layers, intermediate_size) if include_mlp and intermediate_size is not None else None
        attn_grad_stats = (
            {
                "q_proj": _init_stats(num_layers, q_size),
                "k_proj": _init_stats(num_layers, k_size),
                "v_proj": _init_stats(num_layers, v_size),
            }
            if include_attn and q_size is not None and k_size is not None and v_size is not None
            else None
        )
        handles, original_mlp_forwards = attach_hooks(
            model,
            mlp_grad_stats,
            attn_grad_stats,
            include_mlp,
            include_attn,
            recording_strategy,
            mlp_variant,
            mlp_activation_threshold,
            attn_activation_threshold,
        )

        ids = torch.load(os.path.join(tokenize_path, f"train_{lang}.pt"))
        print(f"{lang} tokens loaded succesfully")
        if record_cfg.max_tokens > 0:
            size_to_use = min(len(ids), record_cfg.max_tokens)
        else:
            size_to_use = len(ids)
        num_tokens = (size_to_use // chunk_size) * chunk_size
        ids = ids[:num_tokens]
        ids = ids.view(-1, chunk_size)

        for b in tqdm(range(0, ids.size(0), batch_size)):
            batch = ids[b:b + batch_size].to(device, non_blocking=True)
            if batch.size(1) < 2:
                continue

            if recording_strategy == "grad_act":
                model.zero_grad(set_to_none=True)
                outputs = model(batch, use_cache=False)
                logits = outputs.logits.float()
                target = batch[:, 1:]
                pred = logits[:, :-1, :]
                token_logprobs = F.log_softmax(pred, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
                scalar_objective = token_logprobs.mean()
                scalar_objective.backward()
            else:
                with torch.no_grad():
                    model(batch, use_cache=False)

        save_path = os.path.join(record_cfg.save_dir, ex_id)
        save_activations(
            mlp_grad_stats,
            attn_grad_stats,
            lang,
            save_path,
            ids.size(0),
            recording_strategy,
            mlp_activation_threshold,
            attn_activation_threshold,
            include_mlp,
            include_attn,
        )

        for h in handles:
            h.remove()
        for mlp, original_forward in original_mlp_forwards:
            mlp.forward = original_forward


if __name__ == "__main__":
    main()
