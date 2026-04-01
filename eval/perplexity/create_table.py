import json
from pathlib import Path

import matplotlib.pyplot as plt

# METRICS = ("loss", "perplexity")
METRICS = ("loss", "byte_perplexity")

def format_row(values, widths):
    return "| " + " | ".join(value.ljust(width) for value, width in zip(values, widths)) + " |"


def main():
    results_dir = Path(__file__).resolve().parent / "results-wiki-20k"
    rows = []
    languages = set()

    for json_path in sorted(results_dir.glob("*.json")):
        payload = json.loads(json_path.read_text())
        metrics_by_lang = payload.get("metrics") or {}
        if not metrics_by_lang:
            continue

        row = {"model": payload.get("model_name", json_path.stem).split('/')[-2]}
        for lang, metrics in metrics_by_lang.items():
            languages.add(lang)
            for metric_name in METRICS:
                row[f"{lang}_{metric_name}"] = metrics[metric_name]
        rows.append(row)

    if not rows:
        print("No result JSON files found.")
        return

    headers = ["metric", *[row["model"] for row in rows]]
    metric_keys = [f"{lang}_{metric_name}" for lang in sorted(languages) for metric_name in METRICS]
    raw_values = [[row.get(metric_key, float("nan")) for row in rows] for metric_key in metric_keys]
    metric_keys = [m.replace("_Latn", "") for m in metric_keys]
    table = []
    for metric_key, values in zip(metric_keys, raw_values):
        lowest = min(values)
        table.append(
            [metric_key, *(f"**{value:.4f}**" if value == lowest else f"{value:.4f}" for value in values)]
        )

    widths = [max(len(headers[i]), *(len(line[i]) for line in table)) for i in range(len(headers))]
    output_path = Path(__file__).resolve().parent / "perplexity_table_20k.png"

    fig, ax = plt.subplots(figsize=(16, max(2, 0.45 * (len(table) + 1))))
    ax.axis("off")
    rendered_table = ax.table(
        cellText=[[metric_key, *(f"{value:.4f}" for value in values)] for metric_key, values in zip(metric_keys, raw_values)],
        colLabels=headers,
        cellLoc="left",
        loc="center",
    )
    rendered_table.auto_set_font_size(False)
    rendered_table.set_fontsize(13)
    rendered_table.scale(1, 1.25)
    for row_index, values in enumerate(raw_values, start=1):
        lowest = min(values)
        for col_index, value in enumerate(values, start=1):
            if value == lowest:
                rendered_table[row_index, col_index].set_text_props(weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(format_row(headers, widths))
    print("| " + " | ".join("-" * width for width in widths) + " |")
    for line in table:
        print(format_row(line, widths))
    print(f"Saved table image to {output_path}")


if __name__ == "__main__":
    main()
