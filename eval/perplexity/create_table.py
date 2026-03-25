import json
from pathlib import Path

import matplotlib.pyplot as plt


def iter_rows(results_dir: Path):
    for json_path in sorted(results_dir.glob("*.json")):
        payload = json.loads(json_path.read_text())
        metrics = payload.get("metrics", {}).get("cze_Latn")
        if not metrics:
            continue

        yield {
            "model": payload.get("model_name", json_path.stem),
            "loss": metrics["loss"],
            "perplexity": metrics["perplexity"],
        }


def main():
    results_dir = Path(__file__).resolve().parent / "results"
    rows = list(iter_rows(results_dir))

    if not rows:
        print("No result JSON files found.")
        return

    headers = ("model", "loss", "perplexity")
    formatted_rows = [
        (row["model"], f"{row['loss']:.6f}", f"{row['perplexity']:.6f}")
        for row in rows
    ]
    widths = [
        max(len(header), *(len(row[idx]) for row in formatted_rows))
        for idx, header in enumerate(headers)
    ]

    def format_row(values):
        return "| " + " | ".join(value.ljust(width) for value, width in zip(values, widths)) + " |"

    output_path = Path(__file__).resolve().parent / "perplexity_table.png"
    fig_height = max(1.5, 0.45 * (len(formatted_rows) + 1))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=formatted_rows,
        colLabels=headers,
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.3)

    for (row_idx, _col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(format_row(headers))
    print("| " + " | ".join("-" * width for width in widths) + " |")
    for row in formatted_rows:
        print(format_row(row))
    print(f"Saved table image to {output_path}")


if __name__ == "__main__":
    main()
