import json
import os
import matplotlib.pyplot as plt



def task_filter_predicate(s: str):
    if 'all' in s:
        return False
    if s.startswith("(en) (em) mmlu:") and not s.startswith("(en) (em) mmlu:_average"):
        return False
    return True

def process_file(path: str):
    with open(path, "r") as f:
        results = json.loads(f.read())["results"]

    lang_code = 'cz' if 'cz' in path else 'en'

    file_results = {}
    for key, value in results.items():
        if 'acc' in results[key]:
            file_results[f"({lang_code}) (acc) {key}"] = (value['acc'], value['acc_stderr'])
        elif 'f1' in results[key]:
            file_results[f"({lang_code}) (f1) {key}"] = (value['f1'], value['f1_stderr'])
        elif 'em' in results[key]:
            file_results[f"({lang_code}) (em) {key}"] = (value['em'], value['em_stderr'])
        else:
            raise Exception(f"Unexpected metric: {value}")
    return file_results




if __name__ == "__main__":
    # 1) For each model get list of json files with results
    roots = [
        "/home/kopal/language-neuron-detection/eval/downstream/results_en_tasks",
        "/home/kopal/language-neuron-detection/eval/downstream/results_czech_tasks",
    ]
    names = {}
    for root in roots:
        for sub_root, dirs, files in os.walk(root):
            for file in files:
                full_path = os.path.join(sub_root, file)
                split = full_path.split("/")
                name = split[[i for i, s in enumerate(split) if s in ('7_finetuning', 'meta-llama')][0] + 1]
                if name in names:
                    names[name].append(full_path)
                else:
                    names[name] = [full_path]

    # 2) Put data into 2d format
    col_labels = list(names.keys())
    row_labels = None # Will be set later
    data, data_std = [], []
    for paths in names.values():

        model_results = {}
        for path in paths:
            model_results.update(process_file(path))

        # Set row_labels from first model dict results
        if row_labels is None:
            row_labels = sorted(model_results)
            row_labels = [r for r in row_labels if task_filter_predicate(r)]

        row_data = [f"{model_results[k][0]:.4f} ± {model_results[k][1]:.2f}" for k in row_labels]
        data.append(row_data)

    data = list(zip(*data)) # Transpose data

    # 3) Create figure
    fig_width = max(12, len(col_labels) * 3)
    fig_height = max(6, len(data) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')  # hide axes
    # Create table
    table = ax.table(cellText=data, rowLabels=row_labels, colLabels=col_labels, loc='left')
    table.auto_set_column_width(col=list(range(len(col_labels))))
    # Optional styling
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.4)

    # 4) Make values bold
    for i, (row, row_name) in enumerate(zip(data, row_labels)):
        key_val = max(row)
        for j, val in enumerate(row):
            if val == key_val:
                # +1 because row 0 is header in matplotlib table
                table[(i + 1, j)].set_text_props(weight='bold', fontsize=11)

    plt.savefig("results.png", bbox_inches='tight')






