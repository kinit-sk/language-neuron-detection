import json
from pathlib import Path

METRICS = ("loss", "perplexity")


def format_row(values, widths):
    return "| " + " | ".join(value.ljust(width) for value, width in zip(values, widths)) + " |"


def main():
    results_dir = Path(__file__).resolve().parent / "results-wiki-5k"
    rows = []
    languages = set()

    for json_path in sorted(results_dir.glob("*.json")):
        payload = json.loads(json_path.read_text())
        metrics_by_lang = payload.get("metrics") or {}
        if not metrics_by_lang:
            continue

        row = {"model": payload.get("model_name", json_path.stem)}
        for lang, metrics in metrics_by_lang.items():
            languages.add(lang)
            for metric_name in METRICS:
                row[f"{lang}_{metric_name}"] = metrics[metric_name]
        rows.append(row)

    if not rows:
        print("No result JSON files found.")
        return

    headers = ["metric", *[row["model"] for row in rows]]
    import code; code.interact(local=dict(globals(), **locals()))
    table = [
        [f"{lang}_{metric_name}", *(f"{row.get(f'{lang}_{metric_name}', float('nan')):.6f}" for row in rows)]
        for lang in sorted(languages)
        for metric_name in METRICS
    ]
    widths = [max(len(headers[i]), *(len(line[i]) for line in table)) for i in range(len(headers))]

    print(format_row(headers, widths))
    print("| " + " | ".join("-" * width for width in widths) + " |")
    for line in table:
        print(format_row(line, widths))


if __name__ == "__main__":
    main()
