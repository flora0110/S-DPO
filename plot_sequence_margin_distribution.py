import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_sequence_logprob_margins(jsonl_path):
    margins = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"[Warning] Skip invalid JSON at line {line_id}")
                continue

            value = row.get("sequence_logprob_margin", None)

            if value is None:
                print(f"[Warning] Missing sequence_logprob_margin at line {line_id}")
                continue

            try:
                value = float(value)
            except ValueError:
                print(f"[Warning] Invalid margin value at line {line_id}: {value}")
                continue

            if np.isnan(value) or np.isinf(value):
                print(f"[Warning] Skip NaN/Inf at line {line_id}: {value}")
                continue

            margins.append(value)

    return np.array(margins)


def plot_distribution_and_boxplot(
    margins,
    output_path="sequence_logprob_margin_distribution.png",
    bins=50,
):
    q25 = np.percentile(margins, 25)
    q50 = np.percentile(margins, 50)
    q75 = np.percentile(margins, 75)

    mean = np.mean(margins)
    std = np.std(margins)

    print("=" * 80)
    print("sequence_logprob_margin statistics")
    print("=" * 80)
    print(f"count: {len(margins)}")
    print(f"mean:  {mean:.6f}")
    print(f"std:   {std:.6f}")
    print(f"min:   {np.min(margins):.6f}")
    print(f"q25:   {q25:.6f}")
    print(f"q50:   {q50:.6f}")
    print(f"q75:   {q75:.6f}")
    print(f"max:   {np.max(margins):.6f}")
    print("=" * 80)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 8),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # =========================
    # 1. Histogram
    # =========================
    axes[0].hist(
        margins,
        bins=bins,
        edgecolor="black",
        alpha=0.75,
    )

    axes[0].axvline(q25, linestyle="--", linewidth=2, label=f"q25 = {q25:.3f}")
    axes[0].axvline(q50, linestyle="--", linewidth=2, label=f"q50 = {q50:.3f}")
    axes[0].axvline(q75, linestyle="--", linewidth=2, label=f"q75 = {q75:.3f}")

    axes[0].set_title("Distribution of sequence_logprob_margin")
    axes[0].set_xlabel("sequence_logprob_margin")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # =========================
    # 2. Boxplot
    # =========================
    axes[1].boxplot(
        margins,
        vert=False,
        showmeans=True,
    )

    axes[1].axvline(q25, linestyle="--", linewidth=2)
    axes[1].axvline(q50, linestyle="--", linewidth=2)
    axes[1].axvline(q75, linestyle="--", linewidth=2)

    axes[1].set_title("Boxplot of sequence_logprob_margin")
    axes[1].set_xlabel("sequence_logprob_margin")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--jsonl_path",
        type=str,
        default="candidate_scores.jsonl",
        help="Path to candidate_scores.jsonl",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="sequence_logprob_margin_distribution.png",
        help="Output figure path",
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins",
    )

    args = parser.parse_args()

    margins = load_sequence_logprob_margins(args.jsonl_path)

    if len(margins) == 0:
        raise ValueError("No valid sequence_logprob_margin values found.")

    plot_distribution_and_boxplot(
        margins=margins,
        output_path=args.output_path,
        bins=args.bins,
    )


if __name__ == "__main__":
    main()