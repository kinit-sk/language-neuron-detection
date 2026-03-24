import json
import os
import matplotlib.pyplot as plt




names = []
results_root = "/home/kopal/language-neuron-detection/eval/downstream/results"
for root, dirs, files in os.walk(results_root):
    for file in files:
        full_path = os.path.join(root, file)
        split = full_path.split("/")
        print(full_path)
        name = split[[i for i, s in enumerate(split) if s in ('7_finetuning', 'meta-llama')][0] + 1]
        names.append((name, full_path))




# origin_path = "/home/kopal/language-neuron-detection/results/results/meta-llama/Llama-3.2-3B-instruct/results_2026-03-18T15-21-06.118091.json"
# latn_path = "/home/kopal/language-neuron-detection/results/results/home/kopal/language-neuron-detection/data/7_finetuning/latn/checkpoint-25000/results_2026-03-19T10-32-38.903308.json"
# non_latn_path = "/home/kopal/language-neuron-detection/results/results/home/kopal/language-neuron-detection/data/7_finetuning/non-latn/checkpoint-21000/results_2026-03-19T11-45-22.317912.json"

# col_labels = ["Llama-3.2-3B-instruct", "LATN", "NON-LATN"]
col_labels = [n for n, _ in names]

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


for name, path in names:
    row_keys, values, values_std = process(path)
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
fig_width = max(12, len(col_labels) * 3)
fig_height = max(6, len(data) * 0.45)
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.axis('off')  # hide axes
# Create table
table = ax.table(cellText=data, rowLabels=row_keys, colLabels=col_labels, loc='left')
table.auto_set_column_width(col=list(range(len(col_labels))))

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














