import os
import types
from functools import partial

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from misc import get_device


def _safe_float_tensor(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if not torch.isfinite(x).all():
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def _init_stats(num_layers: int, width: int) -> dict[str, torch.Tensor]:
    return {
        "sums": torch.zeros(num_layers, width, dtype=torch.float32),
        "counts": torch.zeros(num_layers, dtype=torch.int64),
    }


def _accumulate_act_grad(stats: dict[str, torch.Tensor], layer_idx: int, act: torch.Tensor, grad: torch.Tensor):
    act_grad = _safe_float_tensor(act.detach()) * _safe_float_tensor(grad)
    with torch.no_grad():
        stats["sums"][layer_idx] += act_grad.abs().sum(dim=(0, 1)).cpu()
        stats["counts"][layer_idx] += act_grad.size(0) * act_grad.size(1)


def _accumulate_act(stats: dict[str, torch.Tensor], layer_idx: int, act: torch.Tensor):
    act = _safe_float_tensor(act.detach())
    with torch.no_grad():
        stats["sums"][layer_idx] += act.abs().sum(dim=(0, 1)).cpu()
        stats["counts"][layer_idx] += act.size(0) * act.size(1)


def register_mlp_patched_forward(
    cfg: DictConfig,
    mlp: torch.nn.Module,
    layer_idx: int,
    mlp_grad_stats: dict[str, torch.Tensor],
):
    variant = cfg.identify_neurons.record_activations.variant
    recording_strategy = cfg.identify_neurons.record_activations.get("recording_strategy", "grad_act")
    original_forward = mlp.forward

    def patched_forward(self, x):
        gate = self.gate_proj(x)
        gate_act = F.silu(gate)
        up = self.up_proj(x)
        gated = gate_act * up
        tracked = gate_act if variant == "gate" else gated

        if recording_strategy == "grad_act":
            def grad_hook_fn(grad):
                _accumulate_act_grad(mlp_grad_stats, layer_idx, tracked, grad)

            tracked.register_hook(grad_hook_fn)
        elif recording_strategy == "act":
            _accumulate_act(mlp_grad_stats, layer_idx, tracked)
        else:
            raise ValueError(
                f"Unsupported recording_strategy='{recording_strategy}'. "
                "Supported strategies are: 'grad_act', 'act'."
            )
        return self.down_proj(gated)

    mlp.forward = types.MethodType(patched_forward, mlp)
    return original_forward


def register_attn_hooks(
    cfg: DictConfig,
    layer: torch.nn.Module,
    layer_idx: int,
    attn_grad_stats: dict[str, dict[str, torch.Tensor]],
):
    recording_strategy = cfg.identify_neurons.record_activations.get("recording_strategy", "grad_act")

    def attn_hook_fn(module_name: str, _layer_idx: int, _module: torch.nn.Module, _inputs, output: torch.Tensor):
        if recording_strategy == "grad_act":
            def grad_hook_fn(grad):
                _accumulate_act_grad(attn_grad_stats[module_name], _layer_idx, output, grad)

            output.register_hook(grad_hook_fn)
        elif recording_strategy == "act":
            _accumulate_act(attn_grad_stats[module_name], _layer_idx, output)
        else:
            raise ValueError(
                f"Unsupported recording_strategy='{recording_strategy}'. "
                "Supported strategies are: 'grad_act', 'act'."
            )

    return [
        layer.self_attn.q_proj.register_forward_hook(partial(attn_hook_fn, "q_proj", layer_idx)),
        layer.self_attn.k_proj.register_forward_hook(partial(attn_hook_fn, "k_proj", layer_idx)),
        layer.self_attn.v_proj.register_forward_hook(partial(attn_hook_fn, "v_proj", layer_idx)),
    ]


def init_stats(intermediate_size: int, num_layers: int, q_size: int, k_size: int, v_size: int):
    mlp_grad_stats = _init_stats(num_layers, intermediate_size)
    attn_grad_stats = {
        "q_proj": _init_stats(num_layers, q_size),
        "k_proj": _init_stats(num_layers, k_size),
        "v_proj": _init_stats(num_layers, v_size),
    }
    return mlp_grad_stats, attn_grad_stats


def attach_hooks(
    cfg: DictConfig,
    model: torch.nn.Module,
    mlp_grad_stats: dict[str, torch.Tensor],
    attn_grad_stats: dict[str, dict[str, torch.Tensor]],
):
    handles = []
    original_mlp_forwards = []
    for i, layer in enumerate(model.model.layers):
        original_mlp_forward = register_mlp_patched_forward(cfg, layer.mlp, i, mlp_grad_stats)
        original_mlp_forwards.append((layer.mlp, original_mlp_forward))

        if not hasattr(layer, "self_attn") or not hasattr(layer.self_attn, "q_proj") or not hasattr(layer.self_attn, "k_proj") or not hasattr(layer.self_attn, "v_proj"):
            print(f"Skipping layer {i} because it does not have q, k, v projections")
            continue
        handles.extend(register_attn_hooks(cfg, layer, i, attn_grad_stats))

    return handles, original_mlp_forwards


def _compute_average(stats: dict[str, torch.Tensor]) -> torch.Tensor:
    counts = stats["counts"].unsqueeze(1).float()
    return torch.where(
        counts > 0,
        stats["sums"], # / counts # we dont divide by counts because grads are tiny
        torch.zeros_like(stats["sums"]),
    )


def save_activations(
    mlp_grad_stats: dict[str, torch.Tensor],
    attn_grad_stats: dict[str, dict[str, torch.Tensor]],
    lang: str,
    save_dir: str,
    size: int,
    recording_strategy: str,
):
    mlp_grad_average = _compute_average(mlp_grad_stats)
    attn_grad_average = {
        "q_proj_average": _compute_average(attn_grad_stats["q_proj"]),
        "k_proj_average": _compute_average(attn_grad_stats["k_proj"]),
        "v_proj_average": _compute_average(attn_grad_stats["v_proj"]),
    }

    output = dict(
        n=size,
        recording_strategy=recording_strategy,
        mlp_grad_average_activations=mlp_grad_average.to("cpu"),
        attn_grad_average_activations={k: v.to("cpu") for k, v in attn_grad_average.items()},
    )

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{lang}_{recording_strategy}.pt")
    torch.save(output, save_path)
    print(f"Saved activations to {save_path}/\n")


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    tokenize_path = os.path.join(cfg.identify_neurons.tokenize.save_dir, cfg.main.ex_id)
    if not os.path.exists(tokenize_path):
        raise FileNotFoundError("Tokenized data directory does not exist: ensure tokenization step is completed first")

    device = get_device()
    print(f"Using device: {device}")
    model = AutoModelForCausalLM.from_pretrained(cfg.main.model_path, dtype=torch.float16).to(device, non_blocking=True)
    model.eval()

    chunk_size = cfg.identify_neurons.record_activations.chunk_size
    batch_size = cfg.identify_neurons.record_activations.batch_size
    recording_strategy = cfg.identify_neurons.record_activations.get("recording_strategy", "grad_act")
    if recording_strategy not in {"grad_act", "act"}:
        raise ValueError(
            f"Unsupported recording_strategy='{recording_strategy}'. "
            "Supported strategies are: 'grad_act', 'act'."
        )
    num_layers = len(model.model.layers)
    intermediate_size = model.model.layers[0].mlp.gate_proj.out_features
    q_size = model.model.layers[0].self_attn.q_proj.out_features
    k_size = model.model.layers[0].self_attn.k_proj.out_features
    v_size = model.model.layers[0].self_attn.v_proj.out_features

    for lang in cfg.main.languages:
        mlp_grad_stats, attn_grad_stats = init_stats(intermediate_size, num_layers, q_size, k_size, v_size)
        handles, original_mlp_forwards = attach_hooks(cfg, model, mlp_grad_stats, attn_grad_stats)

        ids = torch.load(os.path.join(cfg.identify_neurons.tokenize.save_dir, cfg.main.ex_id, f"{lang}.pt"))
        print(f"{lang} tokens loaded succesfully")
        if cfg.identify_neurons.record_activations.max_tokens > 0:
            size_to_use = min(len(ids), cfg.identify_neurons.record_activations.max_tokens)
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

        save_path = os.path.join(cfg.identify_neurons.record_activations.save_dir, cfg.main.ex_id)
        save_activations(mlp_grad_stats, attn_grad_stats, lang, save_path, ids.size(0), recording_strategy)

        for h in handles:
            h.remove()
        for mlp, original_forward in original_mlp_forwards:
            mlp.forward = original_forward


if __name__ == "__main__":
    main()
