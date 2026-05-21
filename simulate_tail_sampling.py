import json
import random
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_candidate_scores(jsonl_path):
    score_by_example = defaultdict(list)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)

            if "example_id" not in row:
                print(f"[Warning] Missing example_id at line {line_id}")
                continue

            score_by_example[int(row["example_id"])].append(row)

    return dict(score_by_example)


def select_tail_random(
    sorted_rows,
    neg_num,
    tail_direction="low",
    tail_temperature=0.25,
):
    """
    Long-tail sampling over sorted candidate rows.

    Args:
        sorted_rows:
            Candidate rows sorted by sort_metric in ascending order.

        neg_num:
            Number of negatives to sample.

        tail_direction:
            "low":
                highest probability near lowest metric side.
                Probability decays toward higher metric values.

            "high":
                highest probability near highest metric side.
                Probability decays toward lower metric values.

        tail_temperature:
            Controls how flat or sharp the long-tail distribution is.

            smaller value:
                more concentrated near the head side.

            larger value:
                flatter distribution, closer to random.

            Example:
                0.10 -> very sharp
                0.25 -> moderate
                0.50 -> flatter
                1.00 -> close to random
    """
    n = len(sorted_rows)

    if n < neg_num:
        raise ValueError(
            f"Not enough rows for tail_random selection. "
            f"num_rows={n}, neg_num={neg_num}"
        )

    if tail_direction not in ["low", "high"]:
        raise ValueError(
            f"tail_direction must be 'low' or 'high', got {tail_direction}"
        )

    if tail_temperature <= 0:
        raise ValueError(
            f"tail_temperature must be positive, got {tail_temperature}"
        )

    weights = []

    for i in range(n):
        # relative sorted position in [0, 1]
        q = i / max(1, n - 1)

        if tail_direction == "low":
            # high probability at q=0, decays toward q=1
            weight = np.exp(-q / tail_temperature)

        elif tail_direction == "high":
            # high probability at q=1, decays toward q=0
            weight = np.exp(-(1.0 - q) / tail_temperature)

        weights.append(weight)

    selected_rows = []
    available_rows = list(sorted_rows)
    available_weights = list(weights)

    # sample without replacement
    for _ in range(neg_num):
        total_weight = sum(available_weights)

        if total_weight <= 0:
            chosen_idx = random.randrange(len(available_rows))
        else:
            r = random.random() * total_weight
            cumulative = 0.0
            chosen_idx = 0

            for idx, w in enumerate(available_weights):
                cumulative += w
                if cumulative >= r:
                    chosen_idx = idx
                    break

        selected_rows.append(available_rows.pop(chosen_idx))
        available_weights.pop(chosen_idx)

    return selected_rows


def simulate_tail_sampling(
    score_by_example,
    sort_metric="sequence_logprob_margin",
    tail_direction="low",
    tail_temperature=0.25,
    neg_num=3,
    num_simulations=1,
    seed=1958,
):
    random.seed(seed)
    np.random.seed(seed)

    all_metric_values = []
    sampled_metric_values = []
    sampled_relative_positions = []

    skipped_examples = 0

    for sim_id in range(num_simulations):
        for example_id, rows in score_by_example.items():

            valid_rows = []

            for row in rows:
                if sort_metric not in row:
                    raise ValueError(
                        f"`{sort_metric}` not found in row. "
                        f"Available keys: {list(row.keys())}"
                    )

                value = row[sort_metric]

                if value is None:
                    continue

                value = float(value)

                if np.isnan(value) or np.isinf(value):
                    continue

                candidate = row.get("candidate_item", row.get("rejected"))
                true_selection = row.get("trueSelection", row.get("chosen"))

                if candidate == true_selection:
                    continue

                valid_rows.append(row)

            if len(valid_rows) < neg_num:
                skipped_examples += 1
                continue

            sorted_rows = sorted(
                valid_rows,
                key=lambda x: float(x[sort_metric]),
                reverse=False,
            )

            if sim_id == 0:
                all_metric_values.extend(
                    [float(row[sort_metric]) for row in sorted_rows]
                )

            selected_rows = select_tail_random(
                sorted_rows=sorted_rows,
                neg_num=neg_num,
                tail_direction=tail_direction,
                tail_temperature=tail_temperature,
            )

            sampled_metric_values.extend(
                [float(row[sort_metric]) for row in selected_rows]
            )

            n = len(sorted_rows)

            row_to_rank = {
                id(row): rank
                for rank, row in enumerate(sorted_rows)
            }

            for row in selected_rows:
                rank = row_to_rank[id(row)]
                relative_pos = rank / (n - 1) if n > 1 else 0.0
                sampled_relative_positions.append(relative_pos)

    return (
        np.array(all_metric_values),
        np.array(sampled_metric_values),
        np.array(sampled_relative_positions),
        skipped_examples,
    )


def print_stats(name, values):
    print("=" * 80)
    print(name)
    print("=" * 80)
    print(f"count: {len(values)}")
    print(f"mean:  {np.mean(values):.6f}")
    print(f"std:   {np.std(values):.6f}")
    print(f"min:   {np.min(values):.6f}")
    print(f"q25:   {np.percentile(values, 25):.6f}")
    print(f"q50:   {np.percentile(values, 50):.6f}")
    print(f"q75:   {np.percentile(values, 75):.6f}")
    print(f"max:   {np.max(values):.6f}")


def plot_distribution(
    all_values,
    sampled_values,
    sampled_relative_positions,
    output_path,
    sort_metric,
    tail_direction,
    tail_temperature,
    neg_num,
    bins=50,
):
    all_q25 = np.percentile(all_values, 25)
    all_q50 = np.percentile(all_values, 50)
    all_q75 = np.percentile(all_values, 75)

    sampled_q25 = np.percentile(sampled_values, 25)
    sampled_q50 = np.percentile(sampled_values, 50)
    sampled_q75 = np.percentile(sampled_values, 75)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(11, 11),
        gridspec_kw={"height_ratios": [3, 3, 2]},
    )

    # =========================
    # 1. Original distribution
    # =========================
    axes[0].hist(
        all_values,
        bins=bins,
        alpha=0.65,
        edgecolor="black",
        label=f"All candidates: {sort_metric}",
    )

    axes[0].axvline(
        all_q25,
        linestyle="--",
        linewidth=2,
        label=f"all q25 = {all_q25:.3f}",
    )
    axes[0].axvline(
        all_q50,
        linestyle="--",
        linewidth=2,
        label=f"all q50 = {all_q50:.3f}",
    )
    axes[0].axvline(
        all_q75,
        linestyle="--",
        linewidth=2,
        label=f"all q75 = {all_q75:.3f}",
    )

    axes[0].set_title(f"Original distribution of {sort_metric}")
    axes[0].set_xlabel(sort_metric)
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # =========================
    # 2. Sampled distribution
    # =========================
    axes[1].hist(
        sampled_values,
        bins=bins,
        alpha=0.65,
        edgecolor="black",
        label="Sampled negatives",
    )

    axes[1].axvline(
        sampled_q25,
        linestyle="--",
        linewidth=2,
        label=f"sampled q25 = {sampled_q25:.3f}",
    )
    axes[1].axvline(
        sampled_q50,
        linestyle="--",
        linewidth=2,
        label=f"sampled q50 = {sampled_q50:.3f}",
    )
    axes[1].axvline(
        sampled_q75,
        linestyle="--",
        linewidth=2,
        label=f"sampled q75 = {sampled_q75:.3f}",
    )

    axes[1].set_title(
        f"Sampled distribution: tail_direction={tail_direction}, "
        f"tail_temperature={tail_temperature}, neg_num={neg_num}"
    )
    axes[1].set_xlabel(f"sampled {sort_metric}")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # =========================
    # 3. Relative position distribution
    # =========================
    axes[2].hist(
        sampled_relative_positions,
        bins=30,
        alpha=0.75,
        edgecolor="black",
    )

    if tail_direction == "low":
        target_pos = 0.0
    else:
        target_pos = 1.0

    axes[2].axvline(
        target_pos,
        linestyle="--",
        linewidth=2,
        label=f"tail head = {target_pos}",
    )

    axes[2].set_title("Sampled relative rank positions within each example")
    axes[2].set_xlabel(f"relative sorted position by {sort_metric}")
    axes[2].set_ylabel("Frequency")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

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
        default="tail_sample_distribution.png",
        help="Output figure path",
    )

    parser.add_argument(
        "--sort_metric",
        type=str,
        default="sequence_logprob_margin",
        help="Metric used for sorting and plotting",
    )

    parser.add_argument(
        "--tail_direction",
        type=str,
        default="low",
        choices=["low", "high"],
        help="Tail direction: low means bias toward lowest metric; high means bias toward highest metric.",
    )

    parser.add_argument(
        "--tail_temperature",
        type=float,
        default=0.25,
        help="Tail temperature. Smaller = sharper; larger = flatter.",
    )

    parser.add_argument(
        "--neg_num",
        type=int,
        default=3,
        help="Number of negatives sampled per example",
    )

    parser.add_argument(
        "--num_simulations",
        type=int,
        default=1,
        help="1 mimics one actual dataset construction; larger values estimate expectation",
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1958,
    )

    args = parser.parse_args()

    score_by_example = load_candidate_scores(args.jsonl_path)

    (
        all_values,
        sampled_values,
        sampled_relative_positions,
        skipped_examples,
    ) = simulate_tail_sampling(
        score_by_example=score_by_example,
        sort_metric=args.sort_metric,
        tail_direction=args.tail_direction,
        tail_temperature=args.tail_temperature,
        neg_num=args.neg_num,
        num_simulations=args.num_simulations,
        seed=args.seed,
    )

    if len(all_values) == 0:
        raise ValueError("No valid original metric values found.")

    if len(sampled_values) == 0:
        raise ValueError("No sampled metric values found.")

    print_stats(f"All candidate {args.sort_metric}", all_values)
    print_stats(f"Sampled {args.sort_metric}", sampled_values)
    print_stats("Sampled relative rank position", sampled_relative_positions)

    print("=" * 80)
    print("Sampling config")
    print("=" * 80)
    print(f"sort_metric:        {args.sort_metric}")
    print(f"tail_direction:     {args.tail_direction}")
    print(f"tail_temperature:   {args.tail_temperature}")
    print(f"neg_num:            {args.neg_num}")
    print(f"num_simulations:    {args.num_simulations}")
    print(f"skipped_examples:   {skipped_examples}")
    print("=" * 80)

    plot_distribution(
        all_values=all_values,
        sampled_values=sampled_values,
        sampled_relative_positions=sampled_relative_positions,
        output_path=args.output_path,
        sort_metric=args.sort_metric,
        tail_direction=args.tail_direction,
        tail_temperature=args.tail_temperature,
        neg_num=args.neg_num,
        bins=args.bins,
    )


if __name__ == "__main__":
    main()