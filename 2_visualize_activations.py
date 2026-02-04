import hydra
import os
import torch
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import numpy as np


class ActivationAnalyzer:
    def __init__(self, load_dir, save_dir: str, langs: list[str]):
        self.langs = langs
        self.load_dir = load_dir
        self.save_dir = save_dir

        self.avg_activations_per_lang = {}
        self.avg_df = None
        self.avg_dfs = {}
        os.makedirs(self.save_dir, exist_ok=True)

    def load_activation_data(self, lang):
        file_path = f"{self.load_dir}/{lang}.pt"
        if not os.path.exists(file_path):
            print(f"⚠️ File not found for {lang}: {file_path}")
            return None
        return torch.load(file_path, map_location="cpu")

    def _compute_layer_stats(self, avg_acts):
        avg_acts = avg_acts.clone()
        avg_acts[torch.isnan(avg_acts)] = 0
        avg_acts[torch.isinf(avg_acts)] = 0
        if avg_acts.dim() == 1:
            layer_means = avg_acts
        else:
            layer_means = avg_acts.mean(dim=1)
        global_mean = avg_acts.mean()
        return layer_means, global_mean

    def _activation_key(self, name):
        return str(name).replace("/", "_").replace(" ", "_")

    def compute_average_activations(self):
        self.avg_dfs = {}
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
                activations_by_type[key][lang] = {
                    f"Layer_{i:02d}": layer_means[i].item() for i in range(len(layer_means))
                }
                activations_by_type[key][lang]["Global_mean"] = global_mean.item()

            # Attention activations
            if "attn_average_activations" in data and isinstance(data["attn_average_activations"], dict):
                for proj_name, proj_acts in data["attn_average_activations"].items():
                    if not torch.is_tensor(proj_acts):
                        continue
                    layer_means, global_mean = self._compute_layer_stats(proj_acts)
                    key = self._activation_key(proj_name)
                    if key not in activations_by_type:
                        activations_by_type[key] = {}
                    activations_by_type[key][lang] = {
                        f"Layer_{i:02d}": layer_means[i].item() for i in range(len(layer_means))
                    }
                    activations_by_type[key][lang]["Global_mean"] = global_mean.item()

        if not activations_by_type:
            print("❌ No activation data found.")
            return None

        for key, per_lang in activations_by_type.items():
            df = pd.DataFrame.from_dict(per_lang, orient="index")
            cols = sorted([c for c in df.columns if c.startswith("Layer_")])
            if "Global_mean" in df.columns:
                df = df[cols + ["Global_mean"]]
            else:
                df = df[cols]
            self.avg_dfs[key] = df

        self.avg_df = self.avg_dfs.get("mlp_average_activations")
        return self.avg_df


    def visualize(self):
        if not self.avg_dfs:
            print("⚠️ No data to visualize.")
            return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for key, df in self.avg_dfs.items():
            values = df.to_numpy(dtype=float)
            vmin = float(np.nanpercentile(values, 5))
            vmax = float(np.nanpercentile(values, 95))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin = float(np.nanmin(values))
                vmax = float(np.nanmax(values))
            plt.figure(figsize=(14, 6))
            sns.heatmap(
                df,
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
                linewidths=0.3,
                cbar_kws={'label': 'Avg activation'},
                annot=True,
                fmt=".3f",
                annot_kws={"size": 7},
            )

            title_key = key.replace("_", " ")
            plt.title(
                f"Average neuron activations ({title_key}) per layer and language for ...",
                fontsize=14,
            )
            plt.xlabel("Layer")
            plt.ylabel("Language")
            plt.tight_layout()

            if key == "mlp_average_activations":
                filename = f"{self.save_dir}/{timestamp}-average-activations.png"
            else:
                filename = f"{self.save_dir}/{timestamp}-{key}-average-activations.png"
            plt.savefig(filename, dpi=200)
            print(f"Saved heatmap to {filename}")

    def run(self):
        self.compute_average_activations()
        if self.avg_df is not None:
            pd.set_option("display.float_format", "{:,.6f}".format)
            self.visualize()


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):

    if cfg.identify_neurons.record_activations.visualize:

        save_dir = os.path.join(cfg.identify_neurons.record_activations.save_dir, cfg.main.ex_id)
        analyzer = ActivationAnalyzer(
            save_dir,
            save_dir,
            cfg.main.languages
        )
        analyzer.run()



if __name__ == "__main__":
    main()