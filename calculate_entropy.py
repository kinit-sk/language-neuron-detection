import os

import torch


def load_activation_counts(model_path: str, langs: list[str], target_num_tokens: int) -> torch.Tensor:
    model_name = model_path.split("/")[1:][0]
    load_dir = os.path.join("activations", model_name, f"{target_num_tokens}T")
    over_zero = []

    for lang in langs:
        file_path = os.path.join(load_dir, f"{lang}-activations.pt")
        data = torch.load(file_path, map_location="cpu")
        over_zero.append(data["over_zero"])

    return torch.stack(over_zero, dim=0)


def select_language_neurons(
    stacked: torch.Tensor,
    top_rate: float,
    filter_rate: float,
    activation_bar_ratio: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    n_lang, n_layers, width = stacked.shape
    activation_probs = stacked.permute(1, 2, 0).contiguous()
    normed_activation_probs = activation_probs / activation_probs.sum(dim=-1, keepdim=True)
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0

    log_probs = torch.where(
        normed_activation_probs > 0,
        normed_activation_probs.log(),
        torch.zeros_like(normed_activation_probs),
    )
    entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)
    if torch.isnan(entropy).any():
        raise ValueError("NaN values found in entropy")

    flattened_probs = activation_probs.flatten()
    top_prob_k = round(len(flattened_probs) * filter_rate)
    top_prob_value = flattened_probs.kthvalue(top_prob_k).values.item()
    top_position = (activation_probs > top_prob_value).sum(dim=-1)
    entropy[top_position == 0] = torch.inf

    flattened_entropy = entropy.flatten()
    top_entropy_k = round(len(flattened_entropy) * top_rate)
    _, index = flattened_entropy.topk(top_entropy_k, largest=False)
    row_index = index // entropy.size(1)
    col_index = index % entropy.size(1)
    selected_probs = activation_probs[row_index, col_index]

    selected_probs_by_lang = selected_probs.transpose(0, 1)
    activation_bar_k = round(len(flattened_probs) * activation_bar_ratio)
    activation_bar = flattened_probs.kthvalue(activation_bar_k).values.item()
    lang_index, selected_index = torch.where(selected_probs_by_lang > activation_bar)

    selected_mask = torch.zeros((n_lang, n_layers, width), dtype=torch.bool)
    counts_by_lang = torch.bincount(lang_index, minlength=n_lang)
    merged_index = torch.stack((row_index, col_index), dim=-1)

    for current_lang_idx, split_idx in enumerate(selected_index.split(counts_by_lang.tolist())):
        if split_idx.numel() == 0:
            continue
        for layer_idx, hidden_idx in merged_index[split_idx].tolist():
            selected_mask[current_lang_idx, layer_idx, hidden_idx] = True

    stats = {
        "total_neurons": torch.tensor(n_layers * width),
        "neurons_after_filter_rate": (top_position != 0).sum(),
        "neurons_after_activation_bar": (selected_probs_by_lang > activation_bar).sum(),
    }
    return {"selected_mask": selected_mask, "top_valid": selected_probs_by_lang > activation_bar}, stats


def build_final_indices(selected_mask: torch.Tensor, langs: list[str]) -> dict[str, list[torch.Tensor]]:
    final_indices_per_lang: dict[str, list[torch.Tensor]] = {}

    for lang_idx, lang in enumerate(langs):
        per_layer = []
        for layer_idx in range(selected_mask.size(1)):
            indices = torch.nonzero(selected_mask[lang_idx, layer_idx], as_tuple=False).squeeze(1).to(torch.long)
            per_layer.append(indices)
        final_indices_per_lang[lang] = per_layer

    return final_indices_per_lang


def calculate_entropy(
    model_path: str,
    langs: list[str],
    target_num_tokens: int,
    top_rate: float,
    filter_rate: float,
    activation_bar_ratio: float,
    visualize: bool = False,
    debug: bool = False,
):
    stacked = load_activation_counts(model_path, langs, target_num_tokens)
    print(f"Loaded activation counts for {len(langs)} languages")
    if debug:
        print(f"Stacked tensor shape: {tuple(stacked.shape)}")

    selection, stats = select_language_neurons(stacked, top_rate, filter_rate, activation_bar_ratio)
    selected_mask = selection["selected_mask"]
    final_indices_per_lang = build_final_indices(selected_mask, langs)

    print(f"Total original neurons: {int(stats['total_neurons'].item())}")
    print(f"Neurons after filter-rate threshold: {int(stats['neurons_after_filter_rate'].item())}")
    print(f"Neurons after activation-bar threshold: {int(stats['neurons_after_activation_bar'].item())}")

    total_selected = sum(indices.numel() for per_lang in final_indices_per_lang.values() for indices in per_lang)
    print(f"Approximate selected neurons per language: {total_selected / len(langs):.2f}")

    model_name = model_path.split("/")[1:][0]
    save_dir = os.path.join("entropies", model_name)
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"entropy-{target_num_tokens}T.pt")
    torch.save(final_indices_per_lang, output_path)
    print(f"Saved entropies to {output_path}")

    if visualize:
        print("Visualization is not implemented in this legacy utility.")


if __name__ == "__main__":
    calculate_entropy(
        model_path="meta-llama/Meta-Llama-3-8B",
        langs=["en", "de", "fr", "es", "sk", "pl", "ja", "zh"],
        target_num_tokens=200_000,
        top_rate=0.04,
        filter_rate=0.95,
        activation_bar_ratio=0.85,
        visualize=False,
        debug=False,
    )
