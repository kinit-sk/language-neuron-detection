import os
import hydra
import torch
import types
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from functools import partial
from omegaconf import DictConfig
from tqdm import tqdm


from misc import get_device


def _safe_float_tensor(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if not torch.isfinite(x).all():
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def _update_mlp_activation_stats(
    activations: dict[str, torch.Tensor],
    layer_idx: int,
    act: torch.Tensor,
    threshold: float,
    use_abs_sum: bool,
):
    with torch.no_grad():
        if threshold > 0:
            over_zero = (act.abs() > threshold)
        else:
            over_zero = (act > 0)
        activations["over_zero"][layer_idx, :] += over_zero.sum(dim=(0, 1)).cpu()
        act_to_sum = act.abs() if use_abs_sum else act
        activations["activation_sums"][layer_idx, :] += act_to_sum.sum(dim=(0, 1)).cpu()
        activations["activation_counts"][layer_idx, :] += act.size(0) * act.size(1)


def _update_mlp_grad_stats(grad_stats: dict[str, torch.Tensor], layer_idx: int, grad_act: torch.Tensor, threshold: float):
    with torch.no_grad():
        grad_stats["over_zero"][layer_idx, :] += (grad_act.abs() > threshold).sum(dim=(0, 1)).cpu()
        grad_stats["activation_sums"][layer_idx, :] += grad_act.sum(dim=(0, 1)).cpu()
        grad_stats["activation_counts"][layer_idx, :] += grad_act.size(0) * grad_act.size(1)


def register_mlp_patched_forward(
    cfg,
    mlp: torch.nn.Module,
    layer_idx: int,
    mlp_activations: dict[str, torch.Tensor],
    mlp_gradients: dict[str, torch.Tensor],
):
    record_cfg = cfg.identify_neurons.record_activations
    variant = record_cfg.variant
    gated_up_act_threshold = float(record_cfg.gated_up_act_treshold)
    grad_act_threshold = float(record_cfg.get("grad_act_threshold", 0.0))
    original_forward = mlp.forward

    # must include self in the patched function
    def patched_forward(self, x):
        gate = self.gate_proj(x)
        gate_act = F.silu(gate)
        up = self.up_proj(x)
        gated = gate_act * up

        tracked = gate_act if variant == "gate" else gated
        tracked_for_stats = _safe_float_tensor(tracked.detach())

        act_threshold = gated_up_act_threshold if variant == "gated_up" else 0.0
        _update_mlp_activation_stats(
            mlp_activations,
            layer_idx,
            tracked_for_stats,
            act_threshold,
            use_abs_sum=(variant == "gate"),
        )

        def grad_hook_fn(grad):
            grad = _safe_float_tensor(grad)
            grad_act = _safe_float_tensor(tracked_for_stats * grad)
            _update_mlp_grad_stats(mlp_gradients, layer_idx, grad_act, grad_act_threshold)

        tracked.register_hook(grad_hook_fn)

        down = self.down_proj(gated)
        return down

    mlp.forward = types.MethodType(patched_forward, mlp)
    return original_forward


def register_attn_hooks(
    layer: torch.nn.Module,
    layer_idx: int,
    attn_activations: dict[str, torch.Tensor],
    attn_gradients: dict[str, torch.Tensor],
    att_activation_threshold: float,
    att_grad_act_threshold: float,
):
    def attn_hook_fn(module_name, layer_idx, module, inputs, output):
        safe_out = _safe_float_tensor(output.detach())
        with torch.no_grad():
            attn_activations[f"{module_name}_activated"][layer_idx, :] += (safe_out.abs() > att_activation_threshold).sum(dim=(0, 1)).cpu()
            attn_activations[f"{module_name}_sums"][layer_idx, :] += safe_out.abs().sum(dim=(0, 1)).cpu()
            attn_activations[f"{module_name}_counts"][layer_idx] += output.size(0) * output.size(1)

        def attn_grad_hook(grad):
            safe_grad = _safe_float_tensor(grad)
            grad_act = _safe_float_tensor(safe_out * safe_grad)
            with torch.no_grad():
                attn_gradients[f"{module_name}_activated"][layer_idx, :] += (grad_act.abs() > att_grad_act_threshold).sum(dim=(0, 1)).cpu()
                attn_gradients[f"{module_name}_sums"][layer_idx, :] += grad_act.sum(dim=(0, 1)).cpu()
                attn_gradients[f"{module_name}_counts"][layer_idx] += grad_act.size(0) * grad_act.size(1)

        output.register_hook(attn_grad_hook)
    return [
        layer.self_attn.q_proj.register_forward_hook(partial(attn_hook_fn, "q_proj", layer_idx)),
        layer.self_attn.k_proj.register_forward_hook(partial(attn_hook_fn, "k_proj", layer_idx)),
        layer.self_attn.v_proj.register_forward_hook(partial(attn_hook_fn, "v_proj", layer_idx)),
    ]



def init_activations(intermediate_size: int, num_layers: int, q_size: int, k_size: int, v_size: int):
    mlp_activations = {
        "over_zero": torch.zeros(num_layers, intermediate_size, dtype=torch.int32),
        "activation_sums": torch.zeros(num_layers, intermediate_size, dtype=torch.float32),
        "activation_counts": torch.zeros(num_layers, intermediate_size, dtype=torch.int32)
    }
    mlp_gradients = {
        "over_zero": torch.zeros(num_layers, intermediate_size, dtype=torch.int32),
        "activation_sums": torch.zeros(num_layers, intermediate_size, dtype=torch.float32),
        "activation_counts": torch.zeros(num_layers, intermediate_size, dtype=torch.int32)
    }
    attn_activations = {
        "q_proj_activated": torch.zeros(num_layers, q_size, dtype=torch.int32),
        "q_proj_sums": torch.zeros(num_layers, q_size, dtype=torch.float32),
        "q_proj_counts": torch.zeros(num_layers, dtype=torch.int32),
        "k_proj_activated": torch.zeros(num_layers, k_size, dtype=torch.int32),
        "k_proj_sums": torch.zeros(num_layers, k_size, dtype=torch.float32),
        "k_proj_counts": torch.zeros(num_layers, dtype=torch.int32),
        "v_proj_activated": torch.zeros(num_layers, v_size, dtype=torch.int32),
        "v_proj_sums": torch.zeros(num_layers, v_size, dtype=torch.float32),
        "v_proj_counts": torch.zeros(num_layers, dtype=torch.int32),
    }
    attn_gradients = {
        "q_proj_activated": torch.zeros(num_layers, q_size, dtype=torch.int32),
        "q_proj_sums": torch.zeros(num_layers, q_size, dtype=torch.float32),
        "q_proj_counts": torch.zeros(num_layers, dtype=torch.int32),
        "k_proj_activated": torch.zeros(num_layers, k_size, dtype=torch.int32),
        "k_proj_sums": torch.zeros(num_layers, k_size, dtype=torch.float32),
        "k_proj_counts": torch.zeros(num_layers, dtype=torch.int32),
        "v_proj_activated": torch.zeros(num_layers, v_size, dtype=torch.int32),
        "v_proj_sums": torch.zeros(num_layers, v_size, dtype=torch.float32),
        "v_proj_counts": torch.zeros(num_layers, dtype=torch.int32),
    }
    return mlp_activations, mlp_gradients, attn_activations, attn_gradients


def attach_hooks(
    cfg,
    model: torch.nn.Module,
    mlp_activations: dict[str, torch.Tensor],
    mlp_gradients: dict[str, torch.Tensor],
    attn_activations: dict[str, torch.Tensor],
    attn_gradients: dict[str, torch.Tensor],
):
    handles = []
    original_mlp_forwards = []
    att_activation_threshold = float(cfg.identify_neurons.record_activations.get("att_activation_threshold", 0.1))
    att_grad_act_threshold = float(cfg.identify_neurons.record_activations.get("attn_grad_act_threshold", 0.0))
    for i, layer in enumerate(model.model.layers):
        # MLP layer: patch forward so we can capture both activations and activation*gradient.
        original_mlp_forward = register_mlp_patched_forward(cfg, layer.mlp, i, mlp_activations, mlp_gradients)
        original_mlp_forwards.append((layer.mlp, original_mlp_forward))

        # Attention layer
        if not hasattr(layer, "self_attn") or not hasattr(layer.self_attn, "q_proj") or not hasattr(layer.self_attn, "k_proj") or not hasattr(layer.self_attn, "v_proj"):
            print(f"Skipping layer {i} because it does not have q, k, v projections")
            continue

        handles.extend(
            register_attn_hooks(
                layer,
                i,
                attn_activations,
                attn_gradients,
                att_activation_threshold,
                att_grad_act_threshold,
            )
        )
    return handles, original_mlp_forwards


def _compute_average_activations(stats: dict[str, torch.Tensor]) -> torch.Tensor:
    counts = stats["activation_counts"].float()
    return torch.where(
        counts > 0,
        stats["activation_sums"] / counts,
        torch.zeros_like(stats["activation_sums"]),
    )


def _compute_attn_average_activations(attn_stats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for t in ["q", "k", "v"]:
        counts = attn_stats[f"{t}_proj_counts"].unsqueeze(1).float()
        out[f"{t}_proj_average"] = torch.where(
            counts > 0,
            attn_stats[f"{t}_proj_sums"] / counts,
            torch.zeros_like(attn_stats[f"{t}_proj_sums"]),
        )
        out[f"{t}_proj_activation_rate"] = torch.where(
            counts > 0,
            attn_stats[f"{t}_proj_activated"] / counts,
            torch.zeros_like(attn_stats[f"{t}_proj_sums"]),
        )
    return out


def save_activations(
    mlp_activations,
    mlp_gradients,
    attn_activations,
    attn_gradients,
    lang: str,
    save_dir: str,
    size: int,
):

    ## Calculate average activations
    mlp_activations["average_activations"] = _compute_average_activations(mlp_activations)
    mlp_gradients["average_activations"] = _compute_average_activations(mlp_gradients)
    attn_average_activations = _compute_attn_average_activations(attn_activations)
    attn_gradient_average_activations = _compute_attn_average_activations(attn_gradients)

    output = dict(
        n=size,
        mlp_over_zero=mlp_activations["over_zero"].to('cpu'),
        mlp_average_activations=mlp_activations["average_activations"].to('cpu'),
        mlp_activation_counts=mlp_activations["activation_counts"].to('cpu'),
        mlp_grad_over_zero=mlp_gradients["over_zero"].to('cpu'),
        mlp_grad_average_activations=mlp_gradients["average_activations"].to('cpu'),
        mlp_grad_activation_counts=mlp_gradients["activation_counts"].to('cpu'),
        attn_activations={k: v.to('cpu') for k, v in attn_activations.items()},
        attn_average_activations={k: v.to('cpu') for k, v in attn_average_activations.items()},
        attn_grad_activations={k: v.to('cpu') for k, v in attn_gradients.items()},
        attn_grad_average_activations={k: v.to('cpu') for k, v in attn_gradient_average_activations.items()},
    )

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{lang}.pt')
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
    num_layers = len(model.model.layers)  # LLama like models have model.model.layers
    intermediate_size = model.model.layers[0].mlp.gate_proj.out_features
    q_size = model.model.layers[0].self_attn.q_proj.out_features
    k_size = model.model.layers[0].self_attn.k_proj.out_features
    v_size = model.model.layers[0].self_attn.v_proj.out_features

    for lang in cfg.main.languages:

        mlp_activations, mlp_gradients, attn_activations, attn_gradients = init_activations(
            intermediate_size,
            num_layers,
            q_size,
            k_size,
            v_size,
        )
        handles, original_mlp_forwards = attach_hooks(cfg, model, mlp_activations, mlp_gradients, attn_activations, attn_gradients)

        ids = torch.load(os.path.join(cfg.identify_neurons.tokenize.save_dir, cfg.main.ex_id, f"{lang}.pt"))
        print(f"{lang} tokens loaded succesfully")
        num_tokens = (len(ids) // chunk_size) * chunk_size
        ids = ids[:min(num_tokens, cfg.identify_neurons.record_activations.max_tokens)]
        ids = ids.view(-1, chunk_size)

        # 2) Run forward+backward using teacher-forced mean gold-token logprob objective.
        for b in tqdm(range(0, ids.size(0), batch_size)):
            batch = ids[b:b+batch_size].to(device, non_blocking=True)
            if batch.size(1) < 2:
                continue

            model.zero_grad(set_to_none=True)
            outputs = model(batch, use_cache=False)
            logits = outputs.logits.float()
            target = batch[:, 1:]
            pred = logits[:, :-1, :]
            token_logprobs = F.log_softmax(pred, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
            scalar_objective = token_logprobs.mean()
            scalar_objective.backward()

        # 3) Save activations and remove hooks
        save_path = os.path.join(cfg.identify_neurons.record_activations.save_dir, cfg.main.ex_id)
        save_activations(mlp_activations, mlp_gradients, attn_activations, attn_gradients, lang, save_path, ids.size(0))
        for h in handles: 
            if h is not None:
                h.remove()
        for mlp, original_forward in original_mlp_forwards:
            mlp.forward = original_forward




if __name__ == "__main__":
    main()
