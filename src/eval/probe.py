"""
Sonde image → tabulaire : quelles colonnes l'image peut-elle réellement prédire ?

À quoi ça sert
--------------
L'InfoNCE image ↔ tabulaire demande d'apparier une image à SA ligne tabulaire.
Pour que ce soit possible, il faut que la ligne soit (a) identifiable — pas de
doublons, ce que vérifie `src/data/identifiability.py` — et (b) **prédictible
depuis l'image**. Le point (b) est celui qu'on oublie, et c'est en général le
vrai plafond.

Sur CBIS-DDSM avec `--kind roi`, l'entrée est un patch recadré autour de la
lésion. Or :

  * `mass_shape`, `mass_margins`, `calc_type`, `calc_distribution`, `subtlety`
    décrivent la lésion montrée -> **visibles** ;
  * `left_or_right_breast`, `image_view` sont perdus au recadrage -> invisibles ;
  * `breast_density` décrit le sein entier, pas le patch -> invisible ;
  * `n_abnormalities` est un compte à l'échelle de l'examen -> invisible.

Chaque colonne invisible ajoute de l'entropie que la perte contrastive doit
« deviner ». Elle ne le peut pas : cette part est du bruit irréductible dans la
cible, et elle borne l'alignement atteignable.

Ce module mesure (b) empiriquement : on gèle l'encodeur image, on extrait les
embeddings une fois, puis on ajuste une régression logistique multinomiale par
colonne et on compare sa *balanced accuracy* à celle du classifieur majoritaire.
Une colonne dont le gain est nul est invisible — donc à retirer du vecteur, ou
au moins à ne pas espérer aligner.

    python -m src.eval.probe --config configs/cbis_kaggle.yaml \
        --ckpt outputs/pretrain/multimodal/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from ..data.datamodule import build_datasets
from ..data.semantic import resolve_semantic_emb
from ..models.multimodal import build_pretrain_model
from ..utils import get_device, load_config, set_seed


@torch.no_grad()
def extract_image_features(encoder: nn.Module, loader, device) -> np.ndarray:
    encoder.eval()
    feats = []
    for batch in loader:
        feats.append(encoder(batch["image"].to(device)).float().cpu().numpy())
    return np.concatenate(feats)


def fit_and_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                    n_classes: int, epochs: int = 300, lr: float = 0.05,
                    wd: float = 1e-3, device="cpu") -> np.ndarray:
    """
    Régression logistique multinomiale (plein batch, Adam) ajustée sur le train,
    logits renvoyés sur le test.

    Les poids de classe sont indispensables : sans eux, la sonde se contente de
    prédire la classe majoritaire et la balanced accuracy reste au niveau du
    hasard pour une raison triviale, ce qui masquerait le vrai signal.
    """
    Xa = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    ya = torch.as_tensor(y_train, dtype=torch.long, device=device)
    head = nn.Linear(X_train.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    w = torch.as_tensor(counts.sum() / (n_classes * np.maximum(counts, 1)),
                        dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=w)
    for _ in range(epochs):
        opt.zero_grad()
        lossf(head(Xa), ya).backward()
        opt.step()
    with torch.no_grad():
        Xb = torch.as_tensor(X_test, dtype=torch.float32, device=device)
        return head(Xb).cpu().numpy()


def _balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


def probe_tabular_from_image(cfg, ckpt: str | None = None,
                             bundle=None, verbose: bool = True) -> List[Dict]:
    """
    Pour chaque colonne tabulaire : la balanced accuracy d'une sonde linéaire sur
    les embeddings image gelés, contre celle du classifieur majoritaire.

    `bundle` : passez celui déjà construit pour éviter de tout recharger.
    """
    device = get_device(cfg.device)
    set_seed(cfg.seed)
    if bundle is None:
        bundle = build_datasets(cfg, seed=cfg.seed)
    schema = bundle.schema

    sem = resolve_semantic_emb(cfg, schema, verbose=False)
    model = build_pretrain_model(cfg, schema, sem).to(device)
    if ckpt:
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state)
        if verbose:
            print(f"[probe] encodeur image chargé depuis {ckpt}")
    elif verbose:
        print("[probe] encodeur image NON entraîné "
              "(mesure la prédictibilité intrinsèque du patch)")

    # shuffle=False ET drop_last=False : la sonde indexe ensuite ces features
    # avec des masques construits sur le dataset, donc l'alignement doit être
    # ligne à ligne. Avec le drop_last par défaut du split train, les dernières
    # lignes du dataset disparaîtraient (2276 -> 2176 avec un batch de 128).
    tr = bundle.loader("train", 128, shuffle=False, drop_last=False,
                       num_workers=cfg.data.num_workers)
    te = bundle.loader("test", 128, shuffle=False, drop_last=False,
                       num_workers=cfg.data.num_workers)
    if verbose:
        print("[probe] extraction des embeddings image…")
    Xtr = extract_image_features(model.image_encoder, tr, device)
    Xte = extract_image_features(model.image_encoder, te, device)

    # garde-fou : si l'alignement casse un jour, on veut une erreur claire ici,
    # pas un IndexError obscur ni — pire — des features silencieusement décalées
    n_tr, n_te = len(bundle.train_ds.labels), len(bundle.test_ds.labels)
    if len(Xtr) != n_tr or len(Xte) != n_te:
        raise RuntimeError(
            f"features désalignées : {len(Xtr)} embeddings train pour {n_tr} "
            f"lignes, {len(Xte)} pour {n_te}. Le DataLoader a-t-il été construit "
            f"avec drop_last=True ou shuffle=True ?")

    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    rows: List[Dict] = []
    n_num = schema.n_num

    # --- colonnes catégorielles : classification directe
    for j, col in enumerate(schema.cat_cols):
        ytr = bundle.train_ds.x_cat[:, j]
        yte = bundle.test_ds.x_cat[:, j]
        card = schema.cat_cardinalities[j]
        keep_tr = ~bundle.train_ds.mask[:, n_num + j]
        keep_te = ~bundle.test_ds.mask[:, n_num + j]
        if keep_tr.sum() < 30 or keep_te.sum() < 10:
            continue
        Xa, ya = Xtr[keep_tr], ytr[keep_tr]
        Xb, yb = Xte[keep_te], yte[keep_te]
        if len(np.unique(ya)) < 2 or len(np.unique(yb)) < 2:
            continue
        lg = fit_and_predict(Xa, ya, Xb, card, device=device)
        bacc = _balanced_accuracy(yb, lg.argmax(1))
        maj = _balanced_accuracy(yb, np.full_like(yb, np.bincount(ya).argmax()))
        rows.append({"column": col, "kind": "cat", "n_classes": int(card),
                     "n_observed_test": int(keep_te.sum()),
                     "probe_bacc": bacc, "majority_bacc": maj,
                     "gain": bacc - maj})

    # --- colonnes numériques : discrétisées en terciles observés
    for j, col in enumerate(schema.num_cols):
        keep_tr = ~bundle.train_ds.mask[:, j]
        keep_te = ~bundle.test_ds.mask[:, j]
        if keep_tr.sum() < 60:
            continue
        v_tr = bundle.train_ds.x_num[keep_tr, j]
        v_te = bundle.test_ds.x_num[keep_te, j]
        edges = np.quantile(v_tr, [1 / 3, 2 / 3])
        ytr = np.digitize(v_tr, edges)
        yte = np.digitize(v_te, edges)
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2 or keep_te.sum() < 10:
            continue
        lg = fit_and_predict(Xtr[keep_tr], ytr, Xte[keep_te], 3, device=device)
        bacc = _balanced_accuracy(yte, lg.argmax(1))
        maj = _balanced_accuracy(yte, np.full_like(yte, np.bincount(ytr).argmax()))
        rows.append({"column": col, "kind": "num(terciles)", "n_classes": 3,
                     "n_observed_test": int(keep_te.sum()),
                     "probe_bacc": bacc, "majority_bacc": maj,
                     "gain": bacc - maj})

    rows.sort(key=lambda r: -r["gain"])
    if verbose:
        print(f"\n{'colonne':24s} {'type':14s} {'cl.':>4s} {'sonde':>7s} "
              f"{'major.':>7s} {'gain':>7s}  visible ?")
        print("-" * 78)
        for r in rows:
            visible = "OUI" if r["gain"] > 0.05 else ("~" if r["gain"] > 0.02 else "NON")
            print(f"{r['column']:24s} {r['kind']:14s} {r['n_classes']:4d} "
                  f"{r['probe_bacc']:7.3f} {r['majority_bacc']:7.3f} "
                  f"{r['gain']:+7.3f}  {visible}")
        inv = [r["column"] for r in rows if r["gain"] <= 0.02]
        print("\nLecture : une colonne à gain nul n'est pas observable dans l'image.")
        print("L'InfoNCE doit alors la deviner — impossible — et cette entropie")
        print("devient du bruit irréductible qui borne l'alignement atteignable.")
        if inv:
            print(f"\nColonnes non observables ici : {inv}")
            print("-> essayez `data.tabular_view: roi_visible` pour les retirer.")
    return rows


def _predict(Xa, ya, Xb, n_classes, device):
    Xat = torch.as_tensor(Xa, dtype=torch.float32, device=device)
    yat = torch.as_tensor(ya, dtype=torch.long, device=device)
    head = nn.Linear(Xa.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=0.05, weight_decay=1e-3)
    counts = np.bincount(ya, minlength=n_classes).astype(np.float32)
    w = torch.as_tensor(counts.sum() / (n_classes * np.maximum(counts, 1)),
                        dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=w)
    for _ in range(300):
        opt.zero_grad()
        lossf(head(Xat), yat).backward()
        opt.step()
    with torch.no_grad():
        return head(torch.as_tensor(Xb, dtype=torch.float32,
                                    device=device)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="Sonde image -> colonnes tabulaires")
    ap.add_argument("--config", default="configs/cbis_kaggle.yaml")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint SSL ; sans lui, encodeur non entraîné")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    import yaml as _y
    ov = {k: _y.safe_load(v) for k, v in (s.split("=", 1) for s in a.set)}
    cfg = load_config(a.config, ov)
    rows = probe_tabular_from_image(cfg, ckpt=a.ckpt)
    import pandas as pd
    out = Path(cfg.out_dir) / "probe_tabular_from_image.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
