"""
this script calculates the entropy of neuron activations 
calculated in record_activations.py and saves the results to a .pt file
"""

import os
import torch

def calculate_entropy(model_path, langs, target_num_tokens,
                      top_rate, filter_rate, activation_bar_ratio, visualize=True, debug=False, ):
    n, over_zero = [], []
    for lang in langs:
        load_path = f"activations/{model_path.split('/')[1:][0]}/{target_num_tokens}T"
        data = torch.load(f'{load_path}/{lang}-activations.pt')
        n.append(data['n'])
        over_zero.append(data['over_zero'])

    n = torch.tensor(n) # number of tokens per language
    over_zero = torch.stack(over_zero, dim=-1) # number of times neuron >0 per language
    
    num_layers, intermediate_size, lang_num = over_zero.size()
    print(f"📂 {len(langs)} languages loaded succesfully")
    if debug: print(f"layers: {num_layers}, intermediate size: {intermediate_size}, number of languages: {lang_num}")


    # top_rate = 0.2
    # filter_rate = 0.95
    # activation_bar_ratio = 0.95
    activation_probs = over_zero / n  # [layer, inter, lang]
    normed_activation_probs = activation_probs / activation_probs.sum(dim=-1, keepdim=True)
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0

    log_probs = torch.where(normed_activation_probs > 0, normed_activation_probs.log(), 0)
    entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)
    largest = False

    if torch.isnan(entropy).sum():
        print(torch.isnan(entropy).sum())
        raise ValueError
    
    # --- Print original number of neurons ---
    total_neurons = activation_probs.size(0) * activation_probs.size(1)
    print("\n0️⃣  Total original neurons:", total_neurons)

    # --- First threshold: filter neurons that are ever above the filter_rate ---
    flattened_probs = activation_probs.flatten()
    top_prob_value = flattened_probs.kthvalue(round(len(flattened_probs) * filter_rate)).values.item()
    if debug:  print("Filter rate threshold value =", top_prob_value)

    top_position = (activation_probs > top_prob_value).sum(dim=-1)
    entropy[top_position == 0] = -torch.inf if largest else torch.inf

    neurons_after_first_threshold = (top_position != 0).sum().item()
    print("1️⃣  Neurons after first threshold (filter rate):", neurons_after_first_threshold)

    # --- Second threshold: topk by entropy ---
    flattened_entropy = entropy.flatten()
    top_entropy_value = round(len(flattened_entropy) * top_rate)
    _, index = flattened_entropy.topk(top_entropy_value, largest=largest)
    row_index = index // entropy.size(1)
    col_index = index % entropy.size(1)
    selected_probs = activation_probs[row_index, col_index]  # [n, lang]

    selected_probs = selected_probs.transpose(0, 1)
    activation_bar = flattened_probs.kthvalue(round(len(flattened_probs) * activation_bar_ratio)).values.item()
    if debug: print("Second activation bar threshold value =", activation_bar)

    above_second_threshold = (selected_probs > activation_bar).sum().item()
    print("2️⃣  Neurons after second threshold (activation bar):", above_second_threshold)

    lang, indice = torch.where(selected_probs > activation_bar)
    num_after_activation_bar = len(indice)

    if debug: print((selected_probs > activation_bar).sum(dim=1).tolist())

    merged_index = torch.stack((row_index, col_index), dim=-1)
    final_indices_per_lang = {}

    for lang_idx, index in enumerate(indice.split(torch.bincount(lang).tolist())):
        lang_index = [tuple(row.tolist()) for row in merged_index[index]]
        lang_index.sort()
        layer_index = [[] for _ in range(num_layers)]
        for l, h in lang_index:
            layer_index[l].append(h)
        for l, h in enumerate(layer_index):
            layer_index[l] = torch.tensor(h).long()
        final_indices_per_lang[langs[lang_idx]] = layer_index

    total_selected = sum(x.numel() for lang in final_indices_per_lang.values() for x in lang)
    print(f"\n✅ Approximate selected neurons for one language: {total_selected/len(langs)} 🧠")

    save_path = f"entropies/{model_path.split('/')[1:][0]}"
    os.makedirs(save_path, exist_ok=True)
    torch.save(final_indices_per_lang, f'{save_path}/entropy-{target_num_tokens}T.pt')

    print(f"📁 Saved entropies to {save_path}/ )\n")

    if visualize:
        from visual import visualize_entropies
        visualize_entropies(final_indices_per_lang, model_path, target_num_tokens)
        

if __name__ == "__main__":
    calculate_entropy(
        # model_path = "Qwen/Qwen2.5-0.5B-Instruct",
        # model_path = "nvidia/Mistral-NeMo-Minitron-8B-Base",
        model_path = "meta-llama/Meta-Llama-3-8B",
        langs = ["en", "de", "fr", "es", "sk", "pl", "ja", "zh"],
        target_num_tokens = 200_000,
        top_rate=0.04,
        filter_rate=0.95,
        activation_bar_ratio=0.85,
        visualize=True,
        debug=False,
    )


