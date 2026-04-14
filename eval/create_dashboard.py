import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

PERPLEXITY_METRICS = ("loss", "byte_perplexity")
MODEL_ORDER = [
    "Llama-3.2-3B-instruct",
    "latn",
    "non-latn",
    "whole_model",
    "whole_model_small_lr",
    "random",
]
DISPLAY_NAMES = {
    "Llama-3.2-3B-instruct": "Original",
    "latn": "LATN",
    "non-latn": "Non-LATN",
    "whole_model": "Whole model",
    "whole_model_small_lr": "Whole model (small lr)",
    "random": "Random",
}
SUMMARY_TARGET_MODELS = ("latn", "non-latn")
SUMMARY_REFERENCE_MODELS = ("Llama-3.2-3B-instruct", "random", "whole_model")


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Create a single dashboard report for all benchmarks.")
    parser.add_argument(
        "--perplexity-dir",
        type=Path,
        default=script_dir / "perplexity" / "results-wiki-20k",
        help="Directory with perplexity JSON results.",
    )
    parser.add_argument(
        "--downstream-en-dir",
        type=Path,
        default=script_dir / "downstream" / "results_en_8k" / "results",
        help="Directory with English downstream JSON results.",
    )
    parser.add_argument(
        "--downstream-cs-dir",
        type=Path,
        default=script_dir / "downstream" / "results_czech_tasks" / "results",
        help="Directory with Czech downstream JSON results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "dashboard_report.png",
        help="Output path for the dashboard image.",
    )
    return parser.parse_args()


def normalize_model_name(model_name: str) -> str:
    parts = Path(model_name).parts
    if "meta-llama" in parts:
        return parts[-1]
    if "7_finetuning" in parts:
        idx = parts.index("7_finetuning")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if parts and parts[-1].startswith("checkpoint-") and len(parts) >= 2:
        return parts[-2]
    return parts[-1] if parts else model_name


def model_sort_key(model_name: str):
    if model_name in MODEL_ORDER:
        return (0, MODEL_ORDER.index(model_name))
    return (1, model_name)


def display_model_name(model_name: str) -> str:
    return DISPLAY_NAMES.get(model_name, model_name)


def keep_downstream_task(task_name: str) -> bool:
    if "all" in task_name:
        return False
    if task_name.startswith("mmlu:") and not task_name.startswith("mmlu:_average"):
        return False
    return True


def latest_json_files(root: Path):
    latest = {}
    for json_path in sorted(root.glob("**/*.json")):
        model_key = normalize_model_name(str(json_path.parent))
        previous = latest.get(model_key)
        if previous is None or json_path.name > previous.name:
            latest[model_key] = json_path
    return latest


def load_downstream_results(root: Path):
    print(f"Loading downstream results from {root}")
    model_to_path = latest_json_files(root)
    results_by_model = {}
    for model_name, json_path in sorted(model_to_path.items(), key=lambda item: model_sort_key(item[0])):
        with json_path.open() as f:
            payload = json.load(f)

        metrics = {}
        for task_name, values in payload.get("results", {}).items():
            if not keep_downstream_task(task_name):
                continue

            metric_name = next((name for name in ("acc", "f1", "em") if name in values), None)
            if metric_name is None:
                continue

            stderr_key = f"{metric_name}_stderr"
            metrics[task_name] = {
                "metric_name": metric_name,
                "value": values[metric_name],
                "stderr": values.get(stderr_key, float("nan")),
            }

        results_by_model[model_name] = metrics
        print(f"  {model_name}: {json_path.name} ({len(metrics)} tasks)")

    return results_by_model


def latest_perplexity_files(root: Path):
    latest = {}
    for json_path in sorted(root.glob("*.json")):
        with json_path.open() as f:
            payload = json.load(f)
        model_name = normalize_model_name(payload.get("model_name", json_path.stem))
        previous = latest.get(model_name)
        if previous is None or json_path.name > previous.name:
            latest[model_name] = {"path": json_path, "payload": payload}
    return latest


def load_perplexity_results(root: Path):
    print(f"Loading perplexity results from {root}")
    latest = latest_perplexity_files(root)
    results_by_model = {}
    for model_name, item in sorted(latest.items(), key=lambda item: model_sort_key(item[0])):
        payload = item["payload"]
        metrics_by_lang = {}
        for language, values in payload.get("metrics", {}).items():
            for metric_name in PERPLEXITY_METRICS:
                metrics_by_lang[f"{language}_{metric_name}"] = values[metric_name]
        results_by_model[model_name] = metrics_by_lang
        print(f"  {model_name}: {item['path'].name} ({len(metrics_by_lang)} metrics)")

    return results_by_model


def format_task_label(task_name: str, metric_name: str) -> str:
    clean_task = task_name.split("|", 1)[0]
    return f"{clean_task} ({metric_name})"


def format_perplexity_label(metric_key: str) -> str:
    language, metric_name = metric_key.rsplit("_", 1)
    return f"{language.replace('_Latn', '')} {metric_name}"


def build_downstream_table(results_by_model):
    model_names = list(results_by_model)
    task_keys = sorted({task_key for metrics in results_by_model.values() for task_key in metrics})
    headers = ["task", *[display_model_name(model_name) for model_name in model_names]]
    rows = []
    numeric_rows = []

    for task_key in task_keys:
        metric_name = next(
            results_by_model[model_name][task_key]["metric_name"]
            for model_name in model_names
            if task_key in results_by_model[model_name]
        )
        row = [format_task_label(task_key, metric_name)]
        numeric_values = []
        for model_name in model_names:
            values = results_by_model[model_name].get(task_key)
            if values is None:
                row.append("n/a")
                numeric_values.append(None)
                continue

            row.append(f"{values['value']:.4f} +/- {values['stderr']:.2f}")
            numeric_values.append(values["value"])

        rows.append(row)
        numeric_rows.append(numeric_values)

    return headers, rows, numeric_rows


def build_perplexity_table(results_by_model):
    model_names = list(results_by_model)
    metric_keys = sorted({metric_key for metrics in results_by_model.values() for metric_key in metrics})
    headers = ["metric", *[display_model_name(model_name) for model_name in model_names]]
    rows = []
    numeric_rows = []

    for metric_key in metric_keys:
        row = [format_perplexity_label(metric_key)]
        numeric_values = []
        for model_name in model_names:
            value = results_by_model[model_name].get(metric_key)
            if value is None:
                row.append("n/a")
                numeric_values.append(None)
                continue

            row.append(f"{value:.4f}")
            numeric_values.append(value)

        rows.append(row)
        numeric_rows.append(numeric_values)

    return headers, rows, numeric_rows


def comparison_counts(results_by_model, target_model: str, reference_model: str):
    target_metrics = results_by_model.get(target_model, {})
    reference_metrics = results_by_model.get(reference_model, {})
    shared_tasks = sorted(set(target_metrics) & set(reference_metrics))

    better = worse = same = 0
    for task_name in shared_tasks:
        target_value = target_metrics[task_name]["value"]
        reference_value = reference_metrics[task_name]["value"]
        if target_value > reference_value:
            better += 1
        elif target_value < reference_value:
            worse += 1
        else:
            same += 1

    return {
        "better": better,
        "worse": worse,
        "same": same,
        "total": len(shared_tasks),
    }


def build_downstream_summary(dataset_label: str, results_by_model):
    lines = []
    for target_model in SUMMARY_TARGET_MODELS:
        if target_model not in results_by_model:
            continue

        for reference_model in SUMMARY_REFERENCE_MODELS:
            if reference_model not in results_by_model:
                continue

            counts = comparison_counts(results_by_model, target_model, reference_model)
            if counts["total"] == 0:
                continue

            prefix = f"{dataset_label} {display_model_name(target_model)} vs {display_model_name(reference_model)}"
            lines.append(
                f"{prefix}: better {counts['better']} / {counts['total']}, "
                f"worse {counts['worse']} / {counts['total']}, "
                f"same {counts['same']} / {counts['total']}"
            )

    return lines


def render_table(ax, title: str, headers, rows, numeric_rows, higher_is_better: bool):
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=10)

    table = ax.table(cellText=rows, colLabels=headers, cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(headers))))
    table.scale(1, 1.18)

    for col_index in range(len(headers)):
        table[(0, col_index)].set_text_props(weight="bold")

    for row_index, values in enumerate(numeric_rows, start=1):
        present_values = [value for value in values if value is not None]
        if not present_values:
            continue

        best_value = max(present_values) if higher_is_better else min(present_values)
        for col_index, value in enumerate(values, start=1):
            if value is None:
                table[(row_index, col_index)].set_facecolor("#f2f2f2")
                continue
            if value == best_value:
                table[(row_index, col_index)].set_text_props(weight="bold")


def render_summary(ax, title: str, lines):
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=10)
    summary_text = "\n".join(lines) if lines else "No comparable models found."
    ax.text(
        0.0,
        1.0,
        summary_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )


def main():
    args = parse_args()

    perplexity_results = load_perplexity_results(args.perplexity_dir)
    downstream_en_results = load_downstream_results(args.downstream_en_dir)
    downstream_cs_results = load_downstream_results(args.downstream_cs_dir)

    if not perplexity_results:
        raise ValueError(f"No perplexity results found in {args.perplexity_dir}")
    if not downstream_en_results:
        raise ValueError(f"No English downstream results found in {args.downstream_en_dir}")
    if not downstream_cs_results:
        raise ValueError(f"No Czech downstream results found in {args.downstream_cs_dir}")

    perplexity_table = build_perplexity_table(perplexity_results)
    downstream_en_table = build_downstream_table(downstream_en_results)
    downstream_cs_table = build_downstream_table(downstream_cs_results)
    downstream_en_summary = build_downstream_summary("EN", downstream_en_results)
    downstream_cs_summary = build_downstream_summary("CZ", downstream_cs_results)

    sections = [
        {
            "kind": "table",
            "title": "Perplexity Benchmark",
            "payload": (*perplexity_table, False),
            "height": max(2.5, len(perplexity_table[1]) * 0.42 + 1.8),
        },
        {
            "kind": "summary",
            "title": "Downstream Summary (EN)",
            "payload": downstream_en_summary,
            "height": max(1.8, len(downstream_en_summary) * 0.35 + 0.9),
        },
        {
            "kind": "table",
            "title": "Downstream Benchmark (EN)",
            "payload": (*downstream_en_table, True),
            "height": max(2.5, len(downstream_en_table[1]) * 0.42 + 1.8),
        },
        {
            "kind": "summary",
            "title": "Downstream Summary (CZ)",
            "payload": downstream_cs_summary,
            "height": max(1.8, len(downstream_cs_summary) * 0.35 + 0.9),
        },
        {
            "kind": "table",
            "title": "Downstream Benchmark (CZ)",
            "payload": (*downstream_cs_table, True),
            "height": max(2.5, len(downstream_cs_table[1]) * 0.42 + 1.8),
        },
    ]

    row_counts = [section["height"] for section in sections]
    fig, axes = plt.subplots(
        nrows=len(sections),
        figsize=(18, sum(row_counts)),
        gridspec_kw={"height_ratios": row_counts},
    )
    fig.suptitle("Benchmark Dashboard", fontsize=18, fontweight="bold", y=0.995)

    for ax, section in zip(axes, sections):
        if section["kind"] == "summary":
            render_summary(ax, section["title"], section["payload"])
            continue

        headers, rows, numeric_rows, higher_is_better = section["payload"]
        render_table(ax, section["title"], headers, rows, numeric_rows, higher_is_better)

    fig.tight_layout(rect=(0, 0, 1, 0.985))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved dashboard report to {args.output}")


if __name__ == "__main__":
    main()
