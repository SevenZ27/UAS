"""Truthful static figures from aggregate evaluation files, with source CSV/JSON."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/evaluation/stage-two"
OUT = ROOT / "docs/figures/stage-two"
LABELS = {
    "prior_baseline": "Prior baseline",
    "logistic_regression": "Logistic regression",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "random_forest_regularized": "Regularized forest",
    "hist_gradient_boosting": "Histogram boosting",
    "sensitivity_no_clipping": "RF: no clipping (sensitivity)",
    "sensitivity_missing_indicators": "RF: missing indicators (sensitivity)",
}


def export(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=180)
    fig.savefig(OUT / (name + ".svg"))
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cv = json.loads((DATA / "cv-results.json").read_text())
    report = json.loads((DATA / "holdout-results.json").read_text())
    summaries = sorted(cv["candidates"], key=lambda x: x["mean_macro_f1"])
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    ):
        fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
        for i, row in enumerate(summaries):
            folds = [x["macro_f1"] for x in cv["folds"] if x["name"] == row["name"]]
            ax.scatter(
                folds,
                [i] * len(folds),
                color="#8a969e",
                s=24,
                label="Five fold values" if i == 0 else None,
            )
            ax.scatter(
                row["mean_macro_f1"],
                i,
                color="#005b96" if row["eligible"] else "#a45a00",
                marker="D",
                s=50,
                label="Fold mean" if i == 0 else None,
            )
        ax.set(
            yticks=range(len(summaries)),
            yticklabels=[LABELS[x["name"]] for x in summaries],
            xlim=(0, 1),
            xlabel="Macro-F1 · customer-disjoint development folds",
            title="Development comparison before holdout scoring",
        )
        ax.grid(axis="x", alpha=0.2)
        ax.legend(loc="lower right")
        export(fig, "development-comparison")

        cm = np.asarray(report["confusion_matrix"])
        normalized = cm / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
        im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}\n{normalized[i, j]:.1%}",
                    ha="center",
                    va="center",
                    color="white" if normalized[i, j] > 0.55 else "#142332",
                )
        ax.set(
            xticks=range(3),
            yticks=range(3),
            xticklabels=report["classes"],
            yticklabels=report["classes"],
            xlabel="Predicted category",
            ylabel="Actual category",
            title=f"Customer holdout · {report['holdout_rows']:,} observations",
        )
        fig.colorbar(im, ax=ax, label="Proportion within actual class")
        export(fig, "holdout-confusion")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
        for key, name, color, marker in [
            ("reliability_raw", "Uncalibrated", "#005b96", "o"),
            ("reliability_temperature", "OOF temperature", "#a45a00", "^"),
        ]:
            bins = [x for x in report[key]["bins"] if x["count"]]
            axes[0].plot(
                [x["confidence"] for x in bins],
                [x["accuracy"] for x in bins],
                marker=marker,
                color=color,
                label=name,
            )
            axes[1].plot(
                [(x["lower"] + x["upper"]) / 2 for x in report[key]["bins"]],
                [x["count"] for x in report[key]["bins"]],
                marker=marker,
                color=color,
                label=name,
            )
        axes[0].plot(
            [0, 1], [0, 1], ls="--", color="#555555", label="Perfect agreement"
        )
        axes[0].set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Mean top-label probability",
            ylabel="Observed top-label accuracy",
            title="Holdout reliability · ten fixed bins",
        )
        axes[1].set(
            xlim=(0, 1),
            ylim=(0, None),
            xlabel="Top-label probability bin",
            ylabel="Observations",
            title="Support behind the reliability points",
        )
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
        export(fig, "holdout-reliability")

        keys = ["accuracy", "macro_f1", "balanced_accuracy"]
        names = ["Accuracy", "Macro-F1", "Balanced accuracy"]
        fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
        for i, key in enumerate(keys):
            estimate = report["selected"][key]
            lo, hi = report["bootstrap"]["intervals"][key]
            ax.errorbar(
                estimate,
                i,
                xerr=np.asarray([[estimate - lo], [hi - estimate]]),
                fmt="o",
                color="#005b96",
                capsize=5,
                label="Selected model · 95% cluster percentile interval"
                if i == 0
                else None,
            )
            ax.scatter(
                report["prior_baseline"][key],
                i,
                marker="s",
                color="#a45a00",
                label="Prior baseline point estimate" if i == 0 else None,
            )
        ax.set(
            yticks=range(3),
            yticklabels=names,
            xlim=(0, 1),
            xlabel="Score",
            title="Unseen-customer holdout · fixed-model uncertainty",
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
        export(fig, "holdout-uncertainty")
    manifest = {
        "source_files": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [DATA / "cv-results.json", DATA / "holdout-results.json"]
        },
        "figures": {
            f.name: {
                "bytes": f.stat().st_size,
                "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
            }
            for f in OUT.iterdir()
            if f.suffix in [".png", ".svg"]
        },
        "uncertainty": "2000 customer-cluster bootstrap replicates; fixed-model percentile intervals",
        "alt_text": {
            "development-comparison": "Eight development variants with individual fold macro-F1 points and means; sensitivity variants excluded from selection.",
            "holdout-confusion": "Three-category confusion matrix with counts and percentages normalized within each actual class.",
            "holdout-reliability": "Raw and temperature-scaled top-label calibration bins, with observation counts in a companion panel.",
            "holdout-uncertainty": "Selected model point estimates and 95% customer-cluster intervals compared with baseline point estimates.",
        },
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf8"
    )
    print("Exported four figures in PNG/SVG with source hashes and alt text.")


if __name__ == "__main__":
    main()
