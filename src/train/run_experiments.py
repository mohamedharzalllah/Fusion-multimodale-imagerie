"""
ÉTAPE 5 — Le plan d'expériences complet : notre méthode contre les baselines.

Sans ce fichier, le projet ne prouve rien. Une AUC de 0.85 n'a de sens que
comparée à ce qu'on obtient SANS le pré-entraînement multimodal.

Baselines produites
-------------------
  A  img_scratch      image seule, init aléatoire, supervisé
  B  img_imagenet     image seule, init ImageNet, supervisé
  C  img_simclr       image seule, SSL SimCLR unimodal -> fine-tuning
  D  tab_only         tabulaire seul, supervisé (plafond de l'info clinique)
  E  concat_naive     image + tabulaire concaténés, supervisé de bout en bout
                      (= "au-delà de la concaténation naïve" : c'est CE point
                        de comparaison que la méthode doit dépasser)
  F  ours_img         SSL multimodal -> fine-tuning IMAGE SEULE
                      (protocole Hager et al. : au test, pas de tableau)
  G  ours_both        SSL multimodal -> fine-tuning image + tabulaire

Ablations optionnelles (--ablations)
------------------------------------
  abl_naive_tab_enc   encodeur tabulaire = MLP naïf au lieu du FT-Transformer
  abl_no_corruption   augmentation tabulaire désactivée (corruption_rate = 0)
  abl_missing_XX      robustesse : XX % de manquants injectés au test
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..data.datamodule import build_datasets
from ..utils import Cfg, load_config, set_seed
from .finetune import finetune
from .pretrain import pretrain


def _cfg_with(cfg: Cfg, **paths) -> Cfg:
    """Copie profonde de la config avec des overrides 'a.b.c'."""
    new = copy.deepcopy(dict(cfg))
    for k, v in paths.items():
        node = new
        parts = k.split("__")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = v
    return Cfg(new)


def run_all(cfg: Cfg, fractions: List[float] | None = None,
            ablations: bool = False, verbose: bool = True) -> pd.DataFrame:
    set_seed(cfg.seed)
    fractions = fractions or list(cfg.finetune.label_fractions)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Un SEUL bundle de données partagé par toutes les expériences :
    # même split, mêmes prétraitements -> la comparaison est équitable.
    bundle = build_datasets(cfg, seed=cfg.seed)

    # --- Figer les embeddings sémantiques UNE fois pour toutes ---------------
    # `semantic_field_emb: auto` recalculerait la matrice à chaque construction
    # de modèle (1 pretrain + 21 finetunes). Si le backend bascule en cours de
    # route (sentence-transformers 384-d -> repli hachage 128-d), le chargement
    # du checkpoint casse au milieu d'un run de plusieurs heures. On matérialise
    # donc la matrice sur disque et on remplace la config par ce chemin.
    if cfg.model.tabular.get("semantic_field_emb", None) in (True, "auto", "hashing"):
        from ..data.semantic import build_semantic_matrix
        backend = "hashing" if cfg.model.tabular["semantic_field_emb"] == "hashing" else "auto"
        sem = build_semantic_matrix(bundle.schema, backend=backend, verbose=verbose)
        sem_path = out_dir / "semantic_emb.npy"
        np.save(sem_path, sem)
        cfg = _cfg_with(cfg, model__tabular__semantic_field_emb=str(sem_path))
        if verbose:
            print(f"[semantic] matrice figée {sem.shape} -> {sem_path}")

    if verbose:
        print(f"[data] train={len(bundle.train_ds)} val={len(bundle.val_ds)} "
              f"test={len(bundle.test_ds)} | {bundle.schema.n_features} features "
              f"tabulaires ({bundle.schema.n_num} num + {bundle.schema.n_cat} cat) "
              f"| {bundle.n_classes} classes")
        miss = bundle.train_ds.mask.mean()
        print(f"[data] taux global de valeurs manquantes : {miss:.1%}")

    results: List[Dict] = []

    # ================== Pré-entraînements ==================
    if verbose:
        print("\n=== SSL 1/2 : contrastif MULTIMODAL (image <-> tabulaire) ===")
    mm = pretrain(cfg, bundle=bundle, verbose=verbose,
                  mode="multimodal", tag="multimodal")

    if verbose:
        print("\n=== SSL 2/2 : SimCLR IMAGE seule (baseline) ===")
    sc = pretrain(cfg, bundle=bundle, verbose=verbose,
                  mode="simclr_image", tag="simclr")

    # ================== Fine-tunings ==================
    plan = [
        # (tag, ckpt, modalities, cfg_overrides)
        ("A_img_scratch",  None,                 "image",   {"model__image__pretrained": False}),
        ("B_img_imagenet", None,                 "image",   {"model__image__pretrained": True}),
        ("C_img_simclr",   sc["best_ckpt"],      "image",   {}),
        ("D_tab_only",     None,                 "tabular", {}),
        # E doit être VRAIMENT naïve : MLP sur [valeurs imputées | one-hot | masque],
        # pas le FT-Transformer qu'on propose. Sinon la comparaison
        # "au-delà de la concaténation naïve" ne mesure que l'effet du SSL.
        ("E_concat_naive", None,                 "both",
         {"model__tabular__arch": "naive_mlp",
          "model__tabular__semantic_field_emb": None}),
        ("F_ours_img",     mm["best_ckpt"],      "image",   {}),
        ("G_ours_both",    mm["best_ckpt"],      "both",    {}),
    ]

    for frac in fractions:
        for tag, ckpt, mods, ov in plan:
            c = _cfg_with(cfg, **ov) if ov else cfg
            if verbose:
                print(f"\n--- {tag} @ {frac:.0%} des labels ---")
            r = finetune(c, bundle=bundle, pretrained_ckpt=ckpt, modalities=mods,
                         label_fraction=frac, tag=f"{tag}_f{frac}", verbose=verbose)
            r["method"] = tag
            results.append(r)

    # ================== Ablations ==================
    if ablations:
        if verbose:
            print("\n=== ABLATIONS ===")
        # (1) encodeur tabulaire naïf
        c1 = _cfg_with(cfg, model__tabular__arch="naive_mlp")
        b1 = build_datasets(c1, seed=cfg.seed)
        p1 = pretrain(c1, bundle=b1, verbose=verbose, tag="abl_naive_tab")
        r = finetune(c1, bundle=b1, pretrained_ckpt=p1["best_ckpt"], modalities="image",
                     label_fraction=1.0, tag="abl_naive_tab_enc", verbose=verbose)
        r["method"] = "abl_naive_tab_enc"
        results.append(r)

        # (2) sans augmentation tabulaire
        c2 = _cfg_with(cfg, pretrain__tabular_corruption_rate=0.0)
        b2 = build_datasets(c2, seed=cfg.seed)
        p2 = pretrain(c2, bundle=b2, verbose=verbose, tag="abl_no_corruption")
        r = finetune(c2, bundle=b2, pretrained_ckpt=p2["best_ckpt"], modalities="image",
                     label_fraction=1.0, tag="abl_no_corruption", verbose=verbose)
        r["method"] = "abl_no_corruption"
        results.append(r)

        # (3) robustesse : on cache progressivement les métadonnées AU TEST
        rob, base_r = missingness_robustness(cfg, bundle, mm["best_ckpt"],
                                             rates=(0.0, 0.25, 0.5, 0.75, 1.0),
                                             verbose=verbose)
        results.append(base_r)
        rob.to_csv(out_dir / "robustness_missing.csv", index=False)
        if verbose:
            print("\nRobustesse aux métadonnées manquantes au test "
                  "(modèle SSL multimodal, fine-tuning image+tab) :")
            print(rob.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ================== Tableau final ==================
    df = pd.DataFrame(results)
    cols = ["method", "label_fraction", "modalities", "test_auroc",
            "test_auroc_lo", "test_auroc_hi", "test_auprc",
            "test_balanced_accuracy", "test_macro_f1", "test_ece", "val_auroc"]
    df = df[[c for c in cols if c in df.columns]].sort_values(
        ["label_fraction", "test_auroc"], ascending=[True, False])
    df.to_csv(out_dir / "results.csv", index=False)
    # NaN n'est pas du JSON valide (RFC 8259) : on l'écrit en null
    summary = {"multimodal": mm, "simclr": {**sc, "best_r10": None}}
    (out_dir / "pretrain_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    if verbose:
        _print_summary(df, out_dir, mm)
    return df


def _print_summary(df: pd.DataFrame, out_dir: Path, mm: Dict) -> None:
    """
    Sortie finale lisible. La version précédente déversait le DataFrame brut :
    24 lignes x 11 colonnes tronquées sur plusieurs blocs, illisible.
    Ici : un tableau AUROC par méthode x fraction, puis les verdicts avec IC.
    """
    from ..eval.report import PRETTY

    print("\n" + "=" * 78)
    print("AUROC (test) — lignes = méthode, colonnes = fraction d'étiquettes")
    print("=" * 78)
    piv = df.pivot_table(index="method", columns="label_fraction",
                         values="test_auroc")
    piv = piv.reindex([m for m in PRETTY if m in piv.index]
                      + [m for m in piv.index if m not in PRETTY])
    piv.index = [PRETTY.get(m, m) for m in piv.index]
    piv.columns = [f"{c:.0%}" for c in piv.columns]
    print(piv.to_string(float_format=lambda v: f"{v:.4f}", na_rep="   -  "))

    print(f"\n-> détail complet : {out_dir / 'results.csv'}")
    cap = mm.get("max_r10")
    if cap and np.isfinite(mm.get("best_r10", float("nan"))):
        print(f"-> alignement SSL : R@10 = {mm['best_r10']:.3f} "
              f"sur un plafond de {cap:.3f} "
              f"({mm['best_r10'] / cap:.0%} du maximum atteignable)")
    _print_deltas(df)


def missingness_robustness(cfg, bundle, ssl_ckpt: str,
                           rates=(0.0, 0.25, 0.5, 0.75, 1.0),
                           verbose: bool = True) -> pd.DataFrame:
    """
    Ablation "valeurs manquantes" — argument central du sujet de thèse.

    On fine-tune UNE fois en image+tabulaire, puis on ré-évalue le MÊME modèle
    en masquant au test une fraction croissante des métadonnées cliniques.
    C'est la situation réelle : à l'inférence, le dossier clinique est souvent
    incomplet ou absent.

    Le résultat attendu, et ce qui rend l'approche défendable : la courbe doit
    dégringoler moins vite qu'avec une concaténation naïve, parce que
    l'encodeur image a déjà absorbé une partie de l'information tabulaire
    pendant le pré-entraînement contrastif.
    """
    import torch

    from ..data.semantic import resolve_semantic_emb
    from ..models.multimodal import build_downstream_model
    from ..utils import get_device
    from .finetune import finetune, predict_and_score

    device = get_device(cfg.device)
    base_result = finetune(cfg, bundle=bundle, pretrained_ckpt=ssl_ckpt,
                           modalities="both", label_fraction=1.0,
                           tag="abl_missing_base", verbose=verbose)
    base_result["method"] = "abl_missing_base"

    sem = resolve_semantic_emb(cfg, bundle.schema, verbose=False)
    model = build_downstream_model(cfg, bundle.schema, bundle.n_classes, "both",
                                   None, sem).to(device)
    state = torch.load(Path(cfg.out_dir) / "finetune" / "abl_missing_base" / "best.pt",
                       map_location=device, weights_only=False)
    model.load_state_dict(state["model"])

    original_mask = bundle.test_ds.mask.copy()
    original_num = bundle.test_ds.x_num.copy()
    original_cat = bundle.test_ds.x_cat.copy()
    n_num = original_num.shape[1]
    # indice MISSING de chaque colonne catégorielle (dernière catégorie)
    miss_idx = np.array([c - 1 for c in bundle.schema.cat_cardinalities], dtype=np.int64)

    rows = []
    try:
        for p in rates:
            rng = np.random.default_rng(cfg.seed)
            extra = rng.random(original_mask.shape) < p
            bundle.test_ds.mask = original_mask | extra
            # On neutralise AUSSI la valeur elle-même, pas seulement le masque.
            # Avec le FT-Transformer, torch.where(mask, [MISSING], t) écrase déjà
            # le token ; mais le MLP naïf fait un one-hot de x_cat et laisserait
            # donc fuiter la catégorie prétendument masquée — ce qui rendrait la
            # courbe de robustesse faussement flatteuse pour la baseline.
            bundle.test_ds.x_num = np.where(extra[:, :n_num], 0.0,
                                            original_num).astype(np.float32)
            bundle.test_ds.x_cat = np.where(extra[:, n_num:], miss_idx[None, :],
                                            original_cat).astype(np.int64)
            loader = bundle.loader("test", cfg.finetune.batch_size, shuffle=False,
                                   num_workers=0)
            m = predict_and_score(model, loader, device)
            rows.append({"extra_missing_rate": p,
                         "effective_missing_rate": float(bundle.test_ds.mask.mean()),
                         **{f"test_{k}": v for k, v in m.items()}})
    finally:
        bundle.test_ds.mask = original_mask
        bundle.test_ds.x_num = original_num
        bundle.test_ds.x_cat = original_cat
    return pd.DataFrame(rows), base_result


def _print_deltas(df: pd.DataFrame) -> None:
    """
    Verdicts avec intervalles de confiance.

    Deux comparaisons portent une conclusion, et deux seulement :
      F vs B / C : l'apport du SSL multimodal à modalité de test identique (image)
      G vs E     : l'apport du SSL quand le tabulaire est disponible au test
    On affiche l'écart ET le chevauchement des IC95, parce qu'un écart de 0.02
    sur 700 lésions de test ne veut rien dire.
    """
    pairs = [("F_ours_img", ("B_img_imagenet", "C_img_simclr", "A_img_scratch")),
             ("G_ours_both", ("E_concat_naive", "D_tab_only"))]
    has_ci = {"test_auroc_lo", "test_auroc_hi"} <= set(df.columns)

    for ours_name, bases in pairs:
        if ours_name not in set(df["method"]):
            continue
        print(f"\n{ours_name} contre ses références :")
        for frac, g in df.groupby("label_fraction"):
            g = g.set_index("method")
            if ours_name not in g.index:
                continue
            o = g.loc[ours_name]
            head = f"  {frac:>5.0%} labels | {o.test_auroc:.4f}"
            if has_ci and np.isfinite(o.test_auroc_lo):
                head += f" [{o.test_auroc_lo:.3f}-{o.test_auroc_hi:.3f}]"
            parts = []
            for b in bases:
                if b not in g.index:
                    continue
                r = g.loc[b]
                d = o.test_auroc - r.test_auroc
                verdict = ""
                if has_ci and np.isfinite(o.test_auroc_lo) and np.isfinite(r.test_auroc_lo):
                    overlap = not (o.test_auroc_lo > r.test_auroc_hi
                                   or r.test_auroc_lo > o.test_auroc_hi)
                    verdict = " (IC~)" if overlap else " (signif.)"
                parts.append(f"vs {b.split('_', 1)[1]}: {d:+.4f}{verdict}")
            print(head + "  | " + "  ".join(parts))
    print("\n  (IC~) = intervalles de confiance chevauchants -> non concluant")


def main():
    ap = argparse.ArgumentParser(description="Plan d'expériences complet")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--fractions", nargs="*", type=float, default=None)
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    import yaml as _y
    overrides = {k: _y.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
    cfg = load_config(args.config, overrides)
    run_all(cfg, fractions=args.fractions, ablations=args.ablations)


if __name__ == "__main__":
    main()
