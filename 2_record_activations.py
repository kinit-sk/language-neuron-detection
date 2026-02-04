import os
import hydra
import torch
import types
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from functools import partial
from omegaconf import DictConfig
from tqdm import tqdm


from misc import get_device, ActivationAnalyzer


def _safe_float_tensor(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if not torch.isfinite(x).all():
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def register_gate_activation_hooks_on_mlp(mlp: torch.nn.Module, layer_idx: int, activations: dict[str, torch.Tensor]):
    def hook_fn(module, inputs, output):
        act = _safe_float_tensor(F.silu(output))
        with torch.no_grad():
            activations["over_zero"][layer_idx, :] += (act > 0).sum(dim=(0, 1)).cpu() # the gate-fn >0 means (by concept) "active neuron" not value
            activations["activation_sums"][layer_idx, :] += act.sum(dim=(0, 1)).cpu()
            activations["activation_counts"][layer_idx, :] += act.size(0) * act.size(1)
    return mlp.gate_proj.register_forward_hook(hook_fn)



def register_gateup_activation_hooks_on_mlp(mlp: torch.nn.Module, layer_idx: int, activations: dict[str, torch.Tensor], gated_up_act_treshold: float):
    # must include self in the patched function
    def patched_forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gated = _safe_float_tensor(F.silu(gate) * up)

        with torch.no_grad():
            ## count only neurons that do not have activation close to 0 
            activations["over_zero"][layer_idx, :] += (gated.abs() > gated_up_act_treshold).sum(dim=(0, 1)).cpu() # here it is not "active neuron" but the value is importatat (so do not cut negatives)
            activations["activation_sums"][layer_idx, :] += gated.sum(dim=(0, 1)).cpu()
            activations["activation_counts"][layer_idx, :] += gated.size(0) * gated.size(1)
        down = self.down_proj(gated)
        return down

    mlp.forward = types.MethodType(patched_forward, mlp)
    return None  # no handle, because we patched forward



def init_activations(intermediate_size: int, num_layers: int):
    mlp_activations = {
        "over_zero": torch.zeros(num_layers, intermediate_size, dtype=torch.int32),
        "activation_sums": torch.zeros(num_layers, intermediate_size, dtype=torch.float32),
        "activation_counts": torch.zeros(num_layers, intermediate_size, dtype=torch.int32)
    }
    # TODO: Remove hardcoded values 2048 and 512 for attn_activations
    attn_activations = {
        "q_proj_sums": torch.zeros(num_layers, 2048, dtype=torch.float32),
        "q_proj_counts": torch.zeros(num_layers, dtype=torch.int32),
        "k_proj_sums": torch.zeros(num_layers, 512, dtype=torch.float32),
        "k_proj_counts": torch.zeros(num_layers, dtype=torch.int32),
        "v_proj_sums": torch.zeros(num_layers, 512, dtype=torch.float32),
        "v_proj_counts": torch.zeros(num_layers, dtype=torch.int32),
    }
    return mlp_activations, attn_activations


def attach_hooks(cfg, model: torch.nn.Module, mlp_activations: dict[str, torch.Tensor], attn_activations: dict[str, torch.Tensor]):
    handles = []
    for i, layer in enumerate(model.model.layers):
            
        # MLP layer
        if cfg.identify_neurons.record_activations.variant == 'gate':
            handles.append(register_gate_activation_hooks_on_mlp(layer.mlp, i, mlp_activations))
        elif cfg.identify_neurons.record_activations.variant == 'gated_up':
            register_gateup_activation_hooks_on_mlp(layer.mlp, i, mlp_activations, cfg.identify_neurons.record_activations.gated_up_act_treshold)

        # Attention layer
        if not hasattr(layer, "self_attn") or not hasattr(layer.self_attn, "q_proj") or not hasattr(layer.self_attn, "k_proj") or not hasattr(layer.self_attn, "v_proj"):
            print(f"Skipping layer {i} because it does not have q, k, v projections")
            continue

        def attn_hook_fn(module_name, layer_idx, module, inputs, output):
            with torch.no_grad():
                safe_out = _safe_float_tensor(output)
                attn_activations[f"{module_name}_sums"][layer_idx, :] += safe_out.abs().sum(dim=(0, 1)).cpu()
                attn_activations[f"{module_name}_counts"][layer_idx] += output.size(0) * output.size(1)
                
        handles.append(layer.self_attn.q_proj.register_forward_hook(partial(attn_hook_fn, 'q_proj', i)))
        handles.append(layer.self_attn.k_proj.register_forward_hook(partial(attn_hook_fn, 'k_proj', i)))
        handles.append(layer.self_attn.v_proj.register_forward_hook(partial(attn_hook_fn, 'v_proj', i)))
    return handles


def save_activations(mlp_activations, attn_activations, lang: str, save_dir: str, size: int):

    ## Calculate average activations
    mlp_counts = mlp_activations["activation_counts"].float()
    mlp_activations["average_activations"] = torch.where(
        mlp_counts > 0,
        mlp_activations["activation_sums"] / mlp_counts,
        torch.zeros_like(mlp_activations["activation_sums"]),
    )
    attn_average_activations = {
        "q_proj_average": torch.where(
            attn_activations["q_proj_counts"].unsqueeze(1) > 0,
            attn_activations["q_proj_sums"] / attn_activations["q_proj_counts"].unsqueeze(1).float(),
            torch.zeros_like(attn_activations["q_proj_sums"]),
        ),
        "k_proj_average": torch.where(
            attn_activations["k_proj_counts"].unsqueeze(1) > 0,
            attn_activations["k_proj_sums"] / attn_activations["k_proj_counts"].unsqueeze(1).float(),
            torch.zeros_like(attn_activations["k_proj_sums"]),
        ),
        "v_proj_average": torch.where(
            attn_activations["v_proj_counts"].unsqueeze(1) > 0,
            attn_activations["v_proj_sums"] / attn_activations["v_proj_counts"].unsqueeze(1).float(),
            torch.zeros_like(attn_activations["v_proj_sums"]),
        ),
    }

    output = dict(
        n=size,
        mlp_over_zero=mlp_activations["over_zero"].to('cpu'),
        mlp_average_activations=mlp_activations["average_activations"].to('cpu'),
        mlp_activation_counts=mlp_activations["activation_counts"].to('cpu'),
        attn_activations={k: v.to('cpu') for k, v in attn_activations.items()},
        attn_average_activations={k: v.to('cpu') for k, v in attn_average_activations.items()},
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

    for lang in cfg.main.languages:


        mlp_activations, attn_activations = init_activations(intermediate_size, num_layers)
        handles = attach_hooks(cfg, model, mlp_activations, attn_activations)

        ids = torch.load(os.path.join(cfg.identify_neurons.tokenize.save_dir, cfg.main.ex_id, f"{lang}.pt"))
        print(f"{lang} tokens loaded succesfully")
        num_tokens = (len(ids) // chunk_size) * chunk_size
        ids = ids[:min(num_tokens, cfg.identify_neurons.record_activations.max_tokens)]
        ids = ids.view(-1, chunk_size)

        # 2) Run inference
        with torch.inference_mode(): # should be even more efficient than .no_grad()
            for b in tqdm(range(0, ids.size(0), batch_size)):
                batch = ids[b:b+batch_size].to(device, non_blocking=True)
                _ = model(batch)  # forward, patch runs automatically

        # 3) Save activations and remove hooks
        save_path = os.path.join(cfg.identify_neurons.record_activations.save_dir, cfg.main.ex_id)
        save_activations(mlp_activations, attn_activations, lang, save_path, ids.size(0))
        for h in handles: 
            if h is not None:
                h.remove()




if __name__ == "__main__":
    main()