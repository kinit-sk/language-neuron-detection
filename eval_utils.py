import csv
import math
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F


def load_token_ids(
    lang: str,
    tokenized_dir: str,
    target_num_tokens: int,
    missing_hint: str,
    split_prefix: str | None = None,
) -> torch.Tensor:
    if split_prefix:
        token_file = f"{split_prefix}_{lang}.pt"
    else:
        token_file = f"{lang}.pt"
    token_path = os.path.join(tokenized_dir, token_file)
    if not os.path.exists(token_path):
        raise FileNotFoundError(f"Tokenized file not found for {lang}: {token_path}. {missing_hint}")

    token_ids = torch.load(token_path, map_location="cpu")
    if isinstance(token_ids, list):
        token_ids = torch.tensor(token_ids, dtype=torch.long)
    elif isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.to(dtype=torch.long, device="cpu")
    else:
        raise ValueError(f"Unsupported token file format at {token_path}: {type(token_ids)!r}")

    if token_ids.ndim != 1:
        token_ids = token_ids.reshape(-1)

    if target_num_tokens > 0:
        token_ids = token_ids[:target_num_tokens]

    if token_ids.numel() < 2:
        raise ValueError(f"Not enough tokens for language {lang} in {token_path}")

    return token_ids


def compute_perplexity(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> float:
    usable = (token_ids.numel() // seq_len) * seq_len
    if usable < seq_len:
        raise ValueError("Not enough tokens to form one evaluation chunk")
    ids = token_ids[:usable].view(-1, seq_len)

    total_nll = 0.0
    total_tokens = 0

    with torch.inference_mode():
        for start in range(0, ids.size(0), batch_size):
            batch = ids[start : start + batch_size].to(device, non_blocking=True)
            outputs = model(batch, use_cache=False)
            logits = outputs.logits[:, :-1, :].float()
            labels = batch[:, 1:]

            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="sum",
            )
            total_nll += float(nll.item())
            total_tokens += int(labels.numel())

    avg_nll = total_nll / max(total_tokens, 1)
    return float(math.exp(avg_nll))


def save_matrix_csv(path: str, row_langs: list[str], col_langs: list[str], matrix: dict[str, dict[str, float]]):
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["ablated_language", *col_langs])
        for row_lang in row_langs:
            row = [row_lang] + [f"{matrix[row_lang][col_lang]:.8f}" for col_lang in col_langs]
            writer.writerow(row)


def save_heatmap_from_csv(csv_path: str, png_path: str):
    with open(csv_path, "r", encoding="utf-8", newline="") as file_obj:
        reader = csv.reader(file_obj)
        rows = list(reader)

    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(f"CSV does not contain a valid matrix: {csv_path}")

    col_langs = rows[0][1:]
    row_langs: list[str] = []
    values: list[list[float]] = []

    for row in rows[1:]:
        if len(row) != len(col_langs) + 1:
            raise ValueError(f"Malformed CSV row in {csv_path}: {row}")
        row_langs.append(row[0])
        values.append([float(value) for value in row[1:]])

    fig_w = max(8.0, 1.0 + 0.9 * len(col_langs))
    fig_h = max(6.0, 1.0 + 0.7 * len(row_langs))
    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        values,
        xticklabels=col_langs,
        yticklabels=row_langs,
        cmap="coolwarm",
        annot=True,
        fmt=".4f",
        center=0.0,
        cbar_kws={"label": "log(ppx_after / ppx_before)"},
    )
    ax.set_xlabel("Evaluation language")
    ax.set_ylabel("Ablated language")
    ax.set_title("Log-Perplexity-Ratio Matrix After Language-Specific Neuron Ablation")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()


def compute_diag_offdiag_metric(
    row_langs: list[str],
    col_langs: list[str],
    matrix: dict[str, dict[str, float]],
) -> dict[str, float]:
    diagonal_sum = 0.0
    off_diagonal_sum = 0.0
    total_sum = 0.0

    for row_lang in row_langs:
        row = matrix[row_lang]
        for col_lang in col_langs:
            value = float(row[col_lang])
            total_sum += value
            if row_lang == col_lang:
                diagonal_sum += value
            else:
                off_diagonal_sum += value

    return {
        "diagonal_sum": diagonal_sum,
        "off_diagonal_sum": off_diagonal_sum,
        "diag_minus_offdiag": diagonal_sum - off_diagonal_sum,
        "diag_fraction_of_total": (diagonal_sum / total_sum) if total_sum != 0.0 else 0.0,
    }
