import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def plot_real_vs_synthetic_tsne(
    real_scaled,
    synthetic_scaled,
    snapshot_path,
    step,
    synthetic_age,
    seed,
    real_label,
    title_kind,
    output_suffix,
):
    """
    Joint t-SNE of one real set vs synthetic, then save a scatter plot.
    """

    joint = np.concatenate(
        [
            real_scaled,
            synthetic_scaled,
        ],
        axis=0
    )

    n = len(joint)
    perplexity = min(
        30,
        max(2, n - 1)
    )

    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(
        joint
    )

    n_real = len(real_scaled)
    real_2d = embedding[:n_real]
    syn_2d = embedding[n_real:]

    plt.figure(
        figsize=(7, 6)
    )

    plt.scatter(
        real_2d[:, 0],
        real_2d[:, 1],
        s=5,
        alpha=0.4,
        label=real_label
    )

    plt.scatter(
        syn_2d[:, 0],
        syn_2d[:, 1],
        s=5,
        alpha=0.4,
        label="Synthetic"
    )

    plt.title(
        f"Step {step}: "
        f"{title_kind}\n"
        f"Synthetic age = {synthetic_age}"
    )

    plt.xlabel(
        "t-SNE dim 1"
    )
    plt.ylabel(
        "t-SNE dim 2"
    )

    plt.legend()
    plt.tight_layout()

    output_path = snapshot_path.replace(
        ".npz",
        output_suffix
    )

    plt.savefig(
        output_path,
        dpi=200
    )
    plt.close()

    print(
        f"Saved: {output_path}"
    )

    return output_path


def tsne_plot(
    real_scaled,
    recent_scaled,
    synthetic_scaled,
    snapshot_path,
    step,
    synthetic_age,
    seed,
):
    print()

    plot_real_vs_synthetic_tsne(
        real_scaled,
        synthetic_scaled,
        snapshot_path=snapshot_path,
        step=step,
        synthetic_age=synthetic_age,
        seed=seed,
        real_label="Real replay",
        title_kind="Full real vs synthetic",
        output_suffix="_full_real_tsne.png",
    )

    if len(recent_scaled) > 0:
        plot_real_vs_synthetic_tsne(
            recent_scaled,
            synthetic_scaled,
            snapshot_path=snapshot_path,
            step=step,
            synthetic_age=synthetic_age,
            seed=seed,
            real_label="Recent real",
            title_kind="Recent real vs synthetic",
            output_suffix="_recent_real_tsne.png",
        )


def build_transition_features(
    obs,
    actions,
    next_obs
):
    """
    Main analysis representation:

        x = [s, a, s']

    Reward is intentionally excluded for now.
    """

    return np.concatenate(
        [
            obs,
            actions,
            next_obs,
        ],
        axis=-1
    ).astype(np.float32)


def subsample(x, max_samples, rng):
    if len(x) == 0:
        return x

    n = min(
        max_samples,
        len(x)
    )

    indices = rng.choice(
        len(x),
        size=n,
        replace=False
    )

    return x[indices]


def knn_distance(reference, query):
    knn = NearestNeighbors(
        n_neighbors=1
    )

    knn.fit(
        reference
    )

    return (
        knn
        .kneighbors(
            query,
            return_distance=True
        )[0]
        .reshape(-1)
    )


def collect_snapshot_paths(snapshots, snapshot_dir):
    """
    Resolve --snapshot / --snapshot_dir into a unique list of .npz files.

    --snapshot may be a file, a directory, or a glob.
    --snapshot_dir collects snapshot_step_*.npz, falling back to *.npz.
    """

    paths = []

    for item in snapshots or []:
        if os.path.isdir(item):
            found = sorted(
                glob.glob(
                    os.path.join(
                        item,
                        "snapshot_step_*.npz"
                    )
                )
            )
            if not found:
                found = sorted(
                    glob.glob(
                        os.path.join(
                            item,
                            "*.npz"
                        )
                    )
                )
            paths.extend(found)
        elif any(ch in item for ch in "*?["):
            paths.extend(
                sorted(
                    glob.glob(item)
                )
            )
        else:
            paths.append(item)

    if snapshot_dir is not None:
        found = sorted(
            glob.glob(
                os.path.join(
                    snapshot_dir,
                    "snapshot_step_*.npz"
                )
            )
        )
        if not found:
            found = sorted(
                glob.glob(
                    os.path.join(
                        snapshot_dir,
                        "*.npz"
                    )
                )
            )
        paths.extend(found)

    unique = []
    seen = set()

    for path in paths:
        abs_path = os.path.abspath(path)
        if not abs_path.endswith(".npz"):
            continue
        if abs_path in seen:
            continue
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(
                f"Snapshot not found: {path}"
            )
        seen.add(abs_path)
        unique.append(abs_path)

    return unique


def analyze_snapshot(
    snapshot_path,
    max_samples,
    seed,
    do_tsne=False,
):
    data = np.load(
        snapshot_path
    )

    step = int(data["step"][0])
    synthetic_age = int(
        data["synthetic_age"][0]
    )

    print(
        f"step={step}, "
        f"synthetic_age={synthetic_age}"
    )

    real = build_transition_features(
        data["real_obs"],
        data["real_actions"],
        data["real_next_obs"]
    )

    recent = build_transition_features(
        data["recent_real_obs"],
        data["recent_real_actions"],
        data["recent_real_next_obs"]
    )

    synthetic = build_transition_features(
        data["syn_obs"],
        data["syn_actions"],
        data["syn_next_obs"]
    )

    rng = np.random.default_rng(
        seed
    )

    real = subsample(real, max_samples, rng)
    recent = subsample(recent, max_samples, rng)
    synthetic = subsample(synthetic, max_samples, rng)

    print(
        "Shapes:",
        "real=", real.shape,
        "recent=", recent.shape,
        "synthetic=", synthetic.shape,
    )

    scaler = StandardScaler()
    scaler.fit(real)

    real_scaled = scaler.transform(
        real
    )
    recent_scaled = scaler.transform(
        recent
    ) if len(recent) > 0 else recent
    synthetic_scaled = scaler.transform(
        synthetic
    )

    if do_tsne:
        tsne_plot(
            real_scaled,
            recent_scaled,
            synthetic_scaled,
            snapshot_path=snapshot_path,
            step=step,
            synthetic_age=synthetic_age,
            seed=seed,
        )

    syn_to_real_dist = knn_distance(
        real_scaled,
        synthetic_scaled
    )

    if len(recent_scaled) > 0:
        syn_to_recent_dist = knn_distance(
            recent_scaled,
            synthetic_scaled
        )
        syn_to_recent_mean = float(
            syn_to_recent_dist.mean()
        )
        syn_to_recent_std = float(
            syn_to_recent_dist.std()
        )
    else:
        syn_to_recent_mean = float("nan")
        syn_to_recent_std = float("nan")

    print()
    print("===== kNN distance =====")

    print(
        "synthetic -> full real:",
        f"{syn_to_real_dist.mean():.4f}",
        "+/-",
        f"{syn_to_real_dist.std():.4f}"
    )

    print(
        "synthetic -> recent real:",
        f"{syn_to_recent_mean:.4f}",
        "+/-",
        f"{syn_to_recent_std:.4f}"
    )

    return {
        "snapshot": snapshot_path,
        "step": step,
        "synthetic_age": synthetic_age,
        "syn_to_real_mean": float(syn_to_real_dist.mean()),
        "syn_to_real_std": float(syn_to_real_dist.std()),
        "syn_to_recent_mean": syn_to_recent_mean,
        "syn_to_recent_std": syn_to_recent_std,
    }


def save_knn_summary(rows, summary_dir):
    os.makedirs(summary_dir, exist_ok=True)

    csv_path = os.path.join(
        summary_dir,
        "knn_distance_summary.csv"
    )

    fieldnames = [
        "snapshot",
        "step",
        "synthetic_age",
        "syn_to_real_mean",
        "syn_to_real_std",
        "syn_to_recent_mean",
        "syn_to_recent_std",
    ]

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        f"Saved: {csv_path}"
    )

    if len(rows) < 2:
        return csv_path

    rows_sorted = sorted(
        rows,
        key=lambda row: row["step"]
    )

    steps = [
        row["step"] for row in rows_sorted
    ]
    real_mean = [
        row["syn_to_real_mean"] for row in rows_sorted
    ]
    real_std = [
        row["syn_to_real_std"] for row in rows_sorted
    ]
    recent_mean = [
        row["syn_to_recent_mean"] for row in rows_sorted
    ]
    recent_std = [
        row["syn_to_recent_std"] for row in rows_sorted
    ]

    plot_path = os.path.join(
        summary_dir,
        "knn_distance_vs_step.png"
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.errorbar(
        steps,
        real_mean,
        yerr=real_std,
        marker="o",
        capsize=3,
        label="synthetic -> full real"
    )

    plt.errorbar(
        steps,
        recent_mean,
        yerr=recent_std,
        marker="s",
        capsize=3,
        label="synthetic -> recent real"
    )

    plt.xlabel(
        "Step"
    )
    plt.ylabel(
        "kNN distance"
    )
    plt.title(
        "Synthetic distance to real manifold"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        plot_path,
        dpi=200
    )
    plt.close()

    print(
        f"Saved: {plot_path}"
    )

    return csv_path


def process_multifiles(
    snapshot_paths,
    max_samples,
    seed,
    do_tsne=False,
    summary_dir=None,
):
    """
    Run analyze_snapshot on every file, then write a kNN summary.
    """

    if not snapshot_paths:
        raise ValueError(
            "No snapshot .npz files to process."
        )

    if summary_dir is None:
        summary_dir = os.path.dirname(
            snapshot_paths[0]
        )

    rows = []
    n_files = len(snapshot_paths)

    for i, snapshot_path in enumerate(snapshot_paths, start=1):
        print()
        print("=" * 60)
        print(
            f"[{i}/{n_files}] {snapshot_path}"
        )
        print("=" * 60)

        row = analyze_snapshot(
            snapshot_path,
            max_samples=max_samples,
            seed=seed,
            do_tsne=do_tsne,
        )
        rows.append(row)

    save_knn_summary(
        rows,
        summary_dir
    )

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--snapshot",
        nargs="+",
        default=None,
        help="One or more snapshot .npz files, directories, or globs."
    )

    parser.add_argument(
        "--snapshot_dir",
        type=str,
        default=None,
        help="Directory of snapshot_step_*.npz files."
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )

    parser.add_argument(
        "--tsne",
        action="store_true",
        help="Also save t-SNE plots for each snapshot."
    )

    parser.add_argument(
        "--summary_dir",
        type=str,
        default=None,
        help="Where to write knn_distance_summary.csv / knn_distance_vs_step.png."
    )

    args = parser.parse_args()

    snapshot_paths = collect_snapshot_paths(
        args.snapshot,
        args.snapshot_dir
    )

    if not snapshot_paths:
        parser.error(
            "Provide --snapshot and/or --snapshot_dir "
            "pointing to snapshot .npz files."
        )

    summary_dir = args.summary_dir
    if summary_dir is None and args.snapshot_dir is not None:
        summary_dir = args.snapshot_dir

    process_multifiles(
        snapshot_paths,
        max_samples=args.max_samples,
        seed=args.seed,
        do_tsne=args.tsne,
        summary_dir=summary_dir,
    )


if __name__ == "__main__":
    main()
