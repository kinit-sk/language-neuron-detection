import hydra
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from omegaconf import DictConfig


class ActivationVisualizer:
    def __init__(self, load_dir: str, langs: list[str], recording_strategy: str):
        self.langs = langs
        self.load_dir = load_dir
        self.recording_strategy = recording_strategy

    def load_activation_data(self, lang):
        file_path = os.path.join(self.load_dir, f"{lang}_{self.recording_strategy}.pt")
        if not os.path.exists(file_path):
            legacy_path = os.path.join(self.load_dir, f"{lang}.pt")
            if os.path.exists(legacy_path):
                file_path = legacy_path
            else:
                print(f"File not found for {lang}: {file_path}")
                return None
        if not os.path.exists(file_path):
            return None
        return torch.load(file_path, map_location="cpu")

    def _compute_layer_stats(self, avg_acts: torch.Tensor):
        # avg_acts = avg_acts.clone()
        # avg_acts[torch.isnan(avg_acts)] = 0
        # avg_acts[torch.isinf(avg_acts)] = 0
        if avg_acts.dim() == 1:
            layer_means = avg_acts
        else:
            layer_means = avg_acts.mean(dim=1)
        global_mean = avg_acts.mean()
        return layer_means, global_mean

    def _collect_activations(self):
        activations_by_type = {}
        for lang in self.langs:
            data = self.load_activation_data(lang)
            if data is None:
                continue

            # MLP activations
            if "mlp_average_activations" in data:
                avg_acts = data["mlp_average_activations"]
                layer_means, global_mean = self._compute_layer_stats(avg_acts)
                key = "mlp_average_activations"
                if key not in activations_by_type:
                    activations_by_type[key] = {}
                activations_by_type[key][lang] = (layer_means, global_mean)

            # Attention activations
            if "attn_grad_average_activations" in data and isinstance(data["attn_grad_average_activations"], dict):
                for key, proj_acts in data["attn_grad_average_activations"].items():
                    if not torch.is_tensor(proj_acts):
                        continue
                    layer_means, global_mean = self._compute_layer_stats(proj_acts)
                    if key not in activations_by_type:
                        activations_by_type[key] = {}
                    activations_by_type[key][lang] = (layer_means, global_mean)

        if not activations_by_type:
            print("No activation data found.")
            return None
        return activations_by_type

    def _build_heatmap_data(self, per_lang: dict[str, tuple[torch.Tensor, torch.Tensor]]):
        langs = list(per_lang.keys())
        max_layers = max(len(v[0]) for v in per_lang.values())
        cols = [f"Layer_{i:02d}" for i in range(max_layers)] + ["Global_mean"]
        data = np.full((len(langs), len(cols)), np.nan, dtype=float)
        for row, lang in enumerate(langs):
            layer_means, global_mean = per_lang[lang]
            layer_vals = layer_means.detach().cpu().numpy().astype(float)
            data[row, : len(layer_vals)] = layer_vals
            data[row, -1] = float(global_mean.item())
        return data, langs, cols

    def visualize(self):
        activations_by_type = self._collect_activations()
        if not activations_by_type:
            print("No data to visualize.")
            return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for key, per_lang in activations_by_type.items():
            values, y_labels, x_labels = self._build_heatmap_data(per_lang)
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

            title_key = key.replace("_", " ")
            plt.title(
                f"Average neuron activations ({title_key}) per layer and language",
                fontsize=14,
            )
            plt.xlabel("Layer")
            plt.ylabel("Language")
            plt.tight_layout()

            filename = os.path.join(self.load_dir, f"{timestamp}-{key}-activations.png")
            plt.savefig(filename, dpi=200)
            print(f"Saved heatmap to {filename}")

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):

    save_dir = os.path.join(cfg.identify_neurons.record_activations.save_dir, cfg.main.ex_id)
    default_strategy = cfg.identify_neurons.record_activations.get("recording_strategy", "grad_act")
    visualize_cfg = cfg.identify_neurons.record_activations.get("visualize", None)
    recording_strategy = (
        visualize_cfg.get("recording_strategy", default_strategy)
        if visualize_cfg is not None
        else default_strategy
    )
    visualizer = ActivationVisualizer(
        save_dir,
        cfg.main.languages,
        recording_strategy,
    )
    visualizer.visualize()



if __name__ == "__main__":
    main()
