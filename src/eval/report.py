"""Figures et tableau de synthèse à partir de outputs/results.csv."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PRETTY = {
    "A_img_scratch": "A image seule, sans pré-entr.",
    "B_img_imagenet": "B image seule, ImageNet",
    "C_img_simclr": "C image seule, +SimCLR",
    "D_tab_only": "D tabulaire seul",
    "E_concat_naive": "E concaténation naïve",
    "F_ours_img": "F SSL multimodal -> image",
    "G_ours_both": "G SSL multimodal -> img+tab",
}


def plot_label_efficiency(results_csv: str | Path, out_png: str | Path,
                          metric: str = "test_auroc") -> None:
    """Courbe AUROC vs fraction de labels — la figure qui vend le projet."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(results_csv)
    df = df[~df["method"].astype(str).str.startswith("abl_")]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m, g in df.groupby("method"):
        g = g.sort_values("label_fraction")
        style = dict(marker="o", lw=2.4) if m.startswith(("F_", "G_")) else dict(
            marker="s", lw=1.4, alpha=0.75, ls="--")
        ax.plot(g["label_fraction"] * 100, g[metric], label=PRETTY.get(m, m), **style)
    ax.set_xscale("log")
    ax.set_xlabel("Fraction des étiquettes utilisées (%)")
    ax.set_ylabel(metric.replace("test_", "").upper())
    ax.set_title("Efficacité en labels — le SSL multimodal gagne surtout en régime peu annoté")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[report] {out_png}")


def plot_pretrain_curves(log_jsonl: str | Path, out_png: str | Path) -> None:
    """Perte contrastive et qualité de retrieval au fil des époques."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(l) for l in Path(log_jsonl).read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    recalls = [c for c in df.columns if c.startswith("R@") and c.endswith("_i2t")]
    # un log SimCLR n'a pas de retrieval croisé : on n'affiche qu'un panneau
    # plutôt qu'un axe vide et trompeur
    n_panels = 2 if recalls else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(11 if recalls else 6, 4),
                             squeeze=False)
    axes = axes[0]
    axes[0].plot(df["epoch"], df["loss"], label="perte contrastive (train)")
    axes[0].set_xlabel("époque"); axes[0].set_ylabel("perte"); axes[0].grid(alpha=0.3)
    axes[0].legend(); axes[0].set_title("Perte contrastive")
    if recalls:
        for c in recalls:
            axes[1].plot(df["epoch"], df[c], marker="o", ms=3, label=c)
        if "chance_R@1" in df:
            axes[1].axhline(df["chance_R@1"].iloc[-1], color="k", ls=":",
                            label="hasard R@1")
        axes[1].set_xlabel("époque"); axes[1].set_ylabel("recall")
        axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
        axes[1].set_title("Retrieval croisé image → tabulaire (val)")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[report] {out_png}")


def plot_embedding_umap(z_img: np.ndarray, z_tab: np.ndarray, labels: np.ndarray,
                        out_png: str | Path, n: int = 400) -> None:
    """
    Projette les deux modalités dans le même plan 2D.
    Si l'alignement a fonctionné, les paires (image_i, tableau_i) se superposent.
    Repli sur une ACP si umap-learn n'est pas installé.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(n, len(z_img))
    Z = np.concatenate([z_img[:n], z_tab[:n]])
    try:
        import umap
        P = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(Z)
        name = "UMAP"
    except Exception:
        Zc = Z - Z.mean(0)
        U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
        P = Zc @ Vt[:2].T
        name = "ACP"

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(P[:n, 0], P[:n, 1], c=labels[:n], cmap="coolwarm", marker="o",
               s=22, alpha=0.8, label="image")
    ax.scatter(P[n:, 0], P[n:, 1], c=labels[:n], cmap="coolwarm", marker="^",
               s=22, alpha=0.8, label="tabulaire")
    for i in range(0, n, max(n // 40, 1)):
        ax.plot([P[i, 0], P[n + i, 0]], [P[i, 1], P[n + i, 1]], color="gray",
                lw=0.4, alpha=0.5)
    ax.set_title(f"Espace latent partagé ({name}) — traits = paires réelles")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[report] {out_png}")


def markdown_table(results_csv: str | Path, out_md: Optional[str | Path] = None) -> str:
    df = pd.read_csv(results_csv)
    keep = ["method", "label_fraction", "modalities", "test_auroc", "test_auprc",
            "test_balanced_accuracy"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["method"] = df["method"].map(lambda m: PRETTY.get(m, m))
    md = df.to_markdown(index=False, floatfmt=".4f")
    if out_md:
        Path(out_md).write_text(md, encoding="utf-8")
        print(f"[report] {out_md}")
    return md


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs")
    a = ap.parse_args()
    d = Path(a.out_dir)
    if (d / "results.csv").exists():
        plot_label_efficiency(d / "results.csv", d / "figures" / "label_efficiency.png")
        markdown_table(d / "results.csv", d / "results.md")
    log = d / "pretrain" / "multimodal" / "log.jsonl"
    if log.exists():
        plot_pretrain_curves(log, d / "figures" / "pretrain_curves.png")


if __name__ == "__main__":
    main()
