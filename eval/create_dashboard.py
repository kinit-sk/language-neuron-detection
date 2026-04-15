import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

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
SUMMARY_REFERENCE_ORDER = ("Llama-3.2-3B-instruct", "random", "whole_model")
SUMMARY_REFERENCE_LABELS = {
    "Llama-3.2-3B-instruct": "original",
    "random": "random",
    "whole_model": "whole model",
}
COLORS = {
    "page": "#efe7db",
    "panel": "#fffaf2",
    "panel_alt": "#f7efe1",
    "ink": "#1f2933",
    "header": "#17324d",
    "accent": "#bd6b2c",
    "accent_soft": "#f3d9b9",
    "accent_strong": "#e8b36c",
    "muted": "#6a7683",
    "row_even": "#fffaf2",
    "row_odd": "#f8efdf",
    "highlight": "#ffe3a3",
    "na": "#e7ded0",
    "grid": "#d8c8af",
    "pill_dark": "#243b53",
    "pill_light": "#f7d8b6",
    "summary_bad": "#d97841",
    "summary_good": "#2f7d59",
}


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
    headers = ["Task", *[display_model_name(model_name) for model_name in model_names]]
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

            row.append(f"{values['value']:.4f} ± {values['stderr']:.2f}")
            numeric_values.append(values["value"])

        rows.append(row)
        numeric_rows.append(numeric_values)

    return headers, rows, numeric_rows


def build_perplexity_table(results_by_model):
    model_names = list(results_by_model)
    metric_keys = sorted({metric_key for metrics in results_by_model.values() for metric_key in metrics})
    headers = ["Metric", *[display_model_name(model_name) for model_name in model_names]]
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

    better = worse = 0
    for task_name in shared_tasks:
        target_value = target_metrics[task_name]["value"]
        reference_value = reference_metrics[task_name]["value"]
        if target_value > reference_value:
            better += 1
        elif target_value < reference_value:
            worse += 1

    return {"better": better, "worse": worse, "total": len(shared_tasks)}


def build_downstream_summary(dataset_label: str, results_by_model):
    lines = []
    for reference_model in SUMMARY_REFERENCE_ORDER:
        if reference_model not in results_by_model:
            continue

        reference_label = SUMMARY_REFERENCE_LABELS[reference_model]
        for direction in ("worse", "better"):
            if direction == "better" and reference_model != "Llama-3.2-3B-instruct":
                continue

            block_lines = []
            for target_model in SUMMARY_TARGET_MODELS:
                if target_model not in results_by_model:
                    continue

                counts = comparison_counts(results_by_model, target_model, reference_model)
                if counts["total"] == 0:
                    continue

                value = counts[direction]
                target_label = target_model.lower()
                block_lines.append(
                    f"{dataset_label} {direction} than {reference_label} {target_label}: {value} / {counts['total']}"
                )
            if direction == "better":
                numeric_values = []
                for block_line in block_lines:
                    count_text = block_line.rsplit(": ", 1)[-1].split(" / ", 1)[0]
                    numeric_values.append(int(count_text))
                if not any(numeric_values):
                    continue

            if block_lines:
                lines.extend(block_lines)
                lines.append("")

    while lines and not lines[-1]:
        lines.pop()
    return lines


def add_panel_background(ax, facecolor=None):
    ax.set_facecolor(facecolor or COLORS["panel"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_alpha(0)
    card = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.015,rounding_size=0.04",
        transform=ax.transAxes,
        linewidth=1.2,
        edgecolor=COLORS["grid"],
        facecolor=facecolor or COLORS["panel"],
        zorder=-10,
    )
    ax.add_patch(card)


def draw_header_banner(ax, title: str, subtitle: str):
    ax.axis("off")
    gradient_panel = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        transform=ax.transAxes,
        linewidth=0,
        facecolor=COLORS["header"],
        zorder=-10,
    )
    ax.add_patch(gradient_panel)
    ax.add_patch(
        FancyBboxPatch(
            (0.64, 0.12),
            0.28,
            0.76,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            transform=ax.transAxes,
            linewidth=0,
            facecolor="#274d73",
            alpha=0.92,
            zorder=-9,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.71, 0.2),
            0.19,
            0.58,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            transform=ax.transAxes,
            linewidth=0,
            facecolor=COLORS["accent"],
            alpha=0.88,
            zorder=-8,
        )
    )
    ax.text(
        0.05,
        0.68,
        title,
        transform=ax.transAxes,
        fontsize=26,
        fontweight="bold",
        color="white",
        ha="left",
        va="center",
    )
    ax.text(
        0.05,
        0.36,
        subtitle,
        transform=ax.transAxes,
        fontsize=11,
        color="#dce7f2",
        ha="left",
        va="center",
    )
    ax.text(
        0.84,
        0.5,
        "LM\nBenchmarks",
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
    )


def split_summary_blocks(lines):
    blocks = []
    block = []
    for line in lines:
        if line == "":
            if block:
                blocks.append(block)
                block = []
            continue
        block.append(line)
    if block:
        blocks.append(block)
    return blocks


def draw_section_title(ax, title: str, subtitle: str | None = None):
    ax.axis("off")
    add_panel_background(ax)
    ax.text(
        0.03,
        0.72,
        title,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        color=COLORS["header"],
        ha="left",
        va="center",
    )
    if subtitle:
        ax.text(
            0.03,
            0.28,
            subtitle,
            transform=ax.transAxes,
            fontsize=10,
            color=COLORS["muted"],
            ha="left",
            va="center",
        )


def render_table(ax, title: str, headers, rows, numeric_rows, higher_is_better: bool, subtitle: str):
    add_panel_background(ax)
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        title,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color=COLORS["header"],
        ha="left",
        va="top",
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.89),
            0.16,
            0.042,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            transform=ax.transAxes,
            linewidth=0,
            facecolor=COLORS["pill_light"],
        )
    )
    ax.text(
        0.03,
        0.911,
        "Best-in-row highlighted",
        transform=ax.transAxes,
        fontsize=8.5,
        color=COLORS["header"],
        ha="left",
        va="center",
        fontweight="bold",
    )

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="left",
        colLoc="left",
        bbox=[0.02, 0.03, 0.96, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(headers))))
    table.scale(1, 1.14)

    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["grid"])
        cell.set_linewidth(0.6)
        if row_index == 0:
            cell.set_facecolor(COLORS["pill_dark"])
            cell.set_text_props(color="white", weight="bold")
            continue

        cell.set_facecolor(COLORS["row_even"] if row_index % 2 == 0 else COLORS["row_odd"])
        if col_index == 0:
            cell.set_text_props(color=COLORS["ink"], weight="bold")
        else:
            cell.set_text_props(color=COLORS["ink"])

    for row_index, values in enumerate(numeric_rows, start=1):
        present_values = [value for value in values if value is not None]
        if not present_values:
            continue

        best_value = max(present_values) if higher_is_better else min(present_values)
        for col_index, value in enumerate(values, start=1):
            if value is None:
                table[(row_index, col_index)].set_facecolor(COLORS["na"])
                table[(row_index, col_index)].set_text_props(color=COLORS["muted"])
                continue
            if value == best_value:
                table[(row_index, col_index)].set_facecolor(COLORS["highlight"])
                table[(row_index, col_index)].set_text_props(weight="bold", color=COLORS["ink"])


def render_summary_cards(ax, title: str, cz_lines, en_lines, subtitle: str):
    add_panel_background(ax, COLORS["panel_alt"])
    ax.axis("off")
    ax.text(
        0.02,
        0.95,
        title,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        color=COLORS["header"],
        ha="left",
        va="top",
    )
    summary_groups = [("CZ", split_summary_blocks(cz_lines), 0.03), ("EN", split_summary_blocks(en_lines), 0.515)]
    for dataset_label, blocks, x0 in summary_groups:
        card = FancyBboxPatch(
            (x0, 0.08),
            0.45,
            0.74,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            transform=ax.transAxes,
            linewidth=1.0,
            edgecolor=COLORS["grid"],
            facecolor=COLORS["panel"],
        )
        ax.add_patch(card)
        ax.add_patch(
            FancyBboxPatch(
                (x0 + 0.02, 0.72),
                0.08,
                0.08,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                transform=ax.transAxes,
                linewidth=0,
                facecolor=COLORS["pill_dark"] if dataset_label == "CZ" else COLORS["accent"],
            )
        )
        ax.text(
            x0 + 0.06,
            0.76,
            dataset_label,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
        )
        total_lines = sum(len(block) for block in blocks)
        gap_count = max(0, len(blocks) - 1)
        units = total_lines + 0.55 * gap_count
        top_y = 0.66
        bottom_y = 0.14
        available_height = max(0.2, top_y - bottom_y)
        line_step = available_height / max(1, units)
        gap_step = line_step * 0.55
        font_size = min(10.2, max(8.4, 7.6 + line_step * 28))

        y = top_y
        for block in blocks:
            for line in block:
                tone = COLORS["summary_good"] if "better than original" in line else COLORS["ink"]
                if "worse than" in line:
                    tone = COLORS["summary_bad"]
                ax.text(
                    x0 + 0.03,
                    y,
                    line,
                    transform=ax.transAxes,
                    fontsize=font_size,
                    color=tone,
                    ha="left",
                    va="top",
                )
                y -= line_step
            y -= gap_step


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
            "kind": "hero",
            "title": "Benchmark Dashboard",
            "subtitle": "Combined view of perplexity and downstream benchmarks with latest available result files.",
            "height": 1.55,
        },
        {
            "kind": "summary",
            "title": "Downstream Summary",
            "subtitle": "Pairwise counts for LATN and Non-LATN against the original, random, and whole-model baselines.",
            "payload": {"cz": downstream_cs_summary, "en": downstream_en_summary},
            "height": 3.9,
        },
        {
            "kind": "table",
            "title": "Perplexity Benchmark",
            "subtitle": "Lower is better. Best value in each row is highlighted.",
            "payload": (*perplexity_table, False),
            "height": max(3.4, len(perplexity_table[1]) * 0.44 + 2.1),
        },
        {
            "kind": "table",
            "title": "Downstream Benchmark (CZ)",
            "subtitle": "Higher is better. Best value in each row is highlighted.",
            "payload": (*downstream_cs_table, True),
            "height": max(4.1, len(downstream_cs_table[1]) * 0.45 + 2.1),
        },
        {
            "kind": "table",
            "title": "Downstream Benchmark (EN)",
            "subtitle": "Higher is better. Best value in each row is highlighted.",
            "payload": (*downstream_en_table, True),
            "height": max(2.8, len(downstream_en_table[1]) * 0.52 + 1.9),
        },
    ]

    row_counts = [section["height"] for section in sections]
    fig, axes = plt.subplots(
        nrows=len(sections),
        figsize=(18, sum(row_counts)),
        gridspec_kw={"height_ratios": row_counts},
    )
    fig.patch.set_facecolor(COLORS["page"])

    for ax, section in zip(axes, sections):
        if section["kind"] == "hero":
            draw_header_banner(ax, section["title"], section["subtitle"])
            continue

        if section["kind"] == "summary":
            render_summary_cards(
                ax,
                section["title"],
                section["payload"]["cz"],
                section["payload"]["en"],
                section["subtitle"],
            )
            continue

        headers, rows, numeric_rows, higher_is_better = section["payload"]
        render_table(ax, section["title"], headers, rows, numeric_rows, higher_is_better, section["subtitle"])

    fig.tight_layout(pad=1.4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved dashboard report to {args.output}")


if __name__ == "__main__":
    main()
