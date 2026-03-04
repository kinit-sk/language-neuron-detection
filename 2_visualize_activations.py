import os
from datetime import datetime

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from omegaconf import DictConfig

from misc import get_pipeline_step, set_ex_id_from_config_name


def load_activation_data(load_dir: str, lang: str, recording_strategy: str):
    strategy_path = os.path.join(load_dir, f"{lang}_{recording_strategy}.pt")
    if os.path.exists(strategy_path):
        return torch.load(strategy_path, map_location="cpu")

    legacy_path = os.path.join(load_dir, f"{lang}.pt")
    if os.path.exists(legacy_path):
        return torch.load(legacy_path, map_location="cpu")

    print(f"File not found for {lang}: {strategy_path}")
    return None


def compute_layer_stats(avg_acts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if avg_acts.dim() == 1:
        return avg_acts, avg_acts.mean()
    return avg_acts.mean(dim=1), avg_acts.mean()


def collect_activations(
    load_dir: str,
    langs: list[str],
    recording_strategy: str,
) -> dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    activations_by_type: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}

    for lang in langs:
        data = load_activation_data(load_dir, lang, recording_strategy)
        if data is None:
            continue

        mlp_acts = data.get("mlp_average_activations")
        if torch.is_tensor(mlp_acts):
            activations_by_type.setdefault("mlp_average_activations", {})[lang] = compute_layer_stats(mlp_acts)

        attn_acts = data.get("attn_grad_average_activations")
        if isinstance(attn_acts, dict):
            for key, proj_acts in attn_acts.items():
                if torch.is_tensor(proj_acts):
                    activations_by_type.setdefault(key, {})[lang] = compute_layer_stats(proj_acts)

    return activations_by_type


def build_heatmap_data(per_lang: dict[str, tuple[torch.Tensor, torch.Tensor]]):
    langs = list(per_lang.keys())
    max_layers = max(len(values[0]) for values in per_lang.values())
    cols = [f"Layer_{index:02d}" for index in range(max_layers)] + ["Global_mean"]
    data = np.full((len(langs), len(cols)), np.nan, dtype=float)

    for row_idx, lang in enumerate(langs):
        layer_means, global_mean = per_lang[lang]
        layer_vals = layer_means.detach().cpu().numpy().astype(float)
        data[row_idx, : len(layer_vals)] = layer_vals
        data[row_idx, -1] = float(global_mean.item())

    return data, langs, cols


def save_heatmap(
    values: np.ndarray,
    y_labels: list[str],
    x_labels: list[str],
    key: str,
    out_dir: str,
    recording_strategy: str,
    timestamp: str,
):
    vmin = float(np.nanpercentile(values, 5))
    vmax = float(np.nanpercentile(values, 95))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))

    plt.figure(figsize=(14, 6))
    sns.heatmap(
        values,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        linewidths=0.3,
        cbar_kws={"label": "Avg activation"},
        annot=True,
        fmt=".3f",
        annot_kws={"size": 7},
        xticklabels=x_labels,
        yticklabels=y_labels,
    )
    plt.title(f"Average neuron activations ({key.replace('_', ' ')}) per layer and language", fontsize=14)
    plt.xlabel("Layer")
    plt.ylabel("Language")
    plt.tight_layout()

    filename = os.path.join(out_dir, f"{timestamp}-{key}-{recording_strategy}.png")
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved heatmap to {filename}")


def visualize_activations(load_dir: str, out_dir: str, langs: list[str], recording_strategy: str):
    activations_by_type = collect_activations(load_dir, langs, recording_strategy)
    if not activations_by_type:
        print("No activation data found.")
        return

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for key, per_lang in activations_by_type.items():
        values, y_labels, x_labels = build_heatmap_data(per_lang)
        save_heatmap(values, y_labels, x_labels, key, out_dir, recording_strategy, timestamp)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    ex_id = set_ex_id_from_config_name()
    record_cfg = get_pipeline_step(cfg, "step2_record_activations")

    load_dir = os.path.join(record_cfg.save_dir, ex_id)
    default_strategy = record_cfg.get("recording_strategy", "grad_act")
    visualize_cfg = record_cfg.get("visualize")
    recording_strategy = (
        visualize_cfg.get("recording_strategy", default_strategy) if visualize_cfg is not None else default_strategy
    )
    out_base_dir = visualize_cfg.get("save_dir", record_cfg.save_dir) if visualize_cfg is not None else record_cfg.save_dir
    out_dir = os.path.join(out_base_dir, ex_id)

    visualize_activations(load_dir, out_dir, list(cfg.main.languages), recording_strategy)


if __name__ == "__main__":
    main()
