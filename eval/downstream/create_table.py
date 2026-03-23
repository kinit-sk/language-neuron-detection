import json
import matplotlib.pyplot as plt

origin_path = "/home/kopal/language-neuron-detection/results/results/meta-llama/Llama-3.2-3B-instruct/results_2026-03-18T15-21-06.118091.json"
latn_path = "/home/kopal/language-neuron-detection/results/results/home/kopal/language-neuron-detection/data/7_finetuning/latn/checkpoint-25000/results_2026-03-19T10-32-38.903308.json"
non_latn_path = "/home/kopal/language-neuron-detection/results/results/home/kopal/language-neuron-detection/data/7_finetuning/non-latn/checkpoint-21000/results_2026-03-19T11-45-22.317912.json"

col_labels = ["Llama-3.2-3B-instruct", "LATN", "NON-LATN"]

def process(path: str):
    with open(path, "r") as f:
        results = json.loads(f.read())["results"]
    keys = list(sorted((results.keys())))
    values, values_std, key_strings = [], [], []
    for key in keys:
        if 'acc' in results[key]:
            values.append(results[key]['acc'])
            values_std.append(results[key]['acc_stderr'])
            key_strings.append(f"(acc) {key}")
        elif 'f1' in results[key]:
            values.append(results[key]['f1'])
            values_std.append(results[key]['f1_stderr'])
            key_strings.append(f"(f1) {key}")
    return key_strings, values, values_std


data, data_std = [], []

row_keys, values, values_std = process(origin_path)
data.append(values)
data_std.append(values_std)

row_keys, values, values_std = process(latn_path)
data.append(values)
data_std.append(values_std)

row_keys, values, values_std = process(non_latn_path)
data.append(values)
data_std.append(values_std)


cell_text = [
    [f"{val:.4f} ± {std:.2f}" for val, std in zip(row, row_std)]
    for row, row_std in zip(data, data_std)
]

# Transpose data
data = list(zip(*cell_text))
# data_std = list(zip(*values_std))


# Create figure
fig, ax = plt.subplots()
ax.axis('off')  # hide axes
# Create table
table = ax.table(cellText=data, rowLabels=row_keys, colLabels=col_labels, loc='left')

# Optional styling
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.4)


# Make values bold
for i, (row, row_name) in enumerate(zip(data, row_keys)):
    if 'acc' in row_name:
        key_val = max(row)
    elif 'f1' in row_name:
        key_val = min(row)
    for j, val in enumerate(row):
        if val == key_val:
            # +1 because row 0 is header in matplotlib table
            table[(i + 1, j)].set_text_props(weight='bold', fontsize=11)

plt.savefig("results.png", bbox_inches='tight')














