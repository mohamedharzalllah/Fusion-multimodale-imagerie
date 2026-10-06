"""
Métriques d'évaluation, implémentées en numpy pur (aucune dépendance sklearn,
pour que le projet tourne tel quel sur un Colab minimal).

Deux familles :

1. Qualité de la tâche aval  : AUROC, AUPRC, balanced accuracy, F1 macro.
   -> On privilégie AUROC/AUPRC à l'accuracy : la malignité est déséquilibrée,
      et prédire "toujours bénin" donnerait déjà ~75 % d'accuracy.

2. Qualité de l'ALIGNEMENT multimodal : retrieval croisé image <-> tabulaire.
   -> C'est la métrique diagnostique du pré-entraînement : si R@10 ne décolle
      pas au-dessus du hasard, la perte contrastive n'a rien appris et il est
      inutile d'aller plus loin.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Courbes ROC / PR
# ---------------------------------------------------------------------------
def _rank_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """AUROC par la statistique de Mann-Whitney (gère les ex-aequo)."""
    pos, neg = y_true == 1, y_true == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # moyenne des rangs pour les valeurs identiques
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    return _rank_auc(np.asarray(y_true), np.asarray(score, dtype=float))


def auprc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Average precision (aire sous la courbe précision-rappel)."""
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    n_pos = (y_true == 1).sum()
    if n_pos == 0:
        return float("nan")
    recall = tp / n_pos
    d_recall = np.diff(np.concatenate([[0.0], recall]))
    return float((precision * d_recall).sum())


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for c in np.unique(y_true):
        m = y_true == c
        recalls.append((y_pred[m] == c).mean())
    return float(np.mean(recalls))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = []
    for c in np.unique(y_true):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1s.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1s))


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray,
                               n_bins: int = 10) -> float:
    """ECE : écart moyen entre confiance annoncée et exactitude réelle."""
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def classification_report(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    """Rapport complet à partir des logits bruts."""
    y_true = np.asarray(y_true)
    logits = np.asarray(logits, dtype=float)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob = e / e.sum(axis=1, keepdims=True)
    pred = prob.argmax(axis=1)
    n_classes = prob.shape[1]

    out = {
        "accuracy": float((pred == y_true).mean()),
        "balanced_accuracy": balanced_accuracy(y_true, pred),
        "macro_f1": macro_f1(y_true, pred),
        "ece": expected_calibration_error(y_true, prob),
    }
    if n_classes == 2:
        out["auroc"] = auroc(y_true, prob[:, 1])
        out["auprc"] = auprc(y_true, prob[:, 1])
    else:
        # macro one-vs-rest
        aus = [auroc((y_true == c).astype(int), prob[:, c]) for c in range(n_classes)]
        aps = [auprc((y_true == c).astype(int), prob[:, c]) for c in range(n_classes)]
        out["auroc"] = float(np.nanmean(aus))
        out["auprc"] = float(np.nanmean(aps))
    return out


# ---------------------------------------------------------------------------
# Alignement multimodal
# ---------------------------------------------------------------------------
def retrieval_metrics(z_img: np.ndarray, z_tab: np.ndarray,
                      ks: Sequence[int] = (1, 5, 10)) -> Dict[str, float]:
    """
    Recall@K du retrieval croisé sur un ensemble de N paires.

    Pour chaque image, on classe les N vecteurs tabulaires par similarité
    cosinus ; le succès = le vrai partenaire est dans le top-K.
    Hasard = K/N. Un R@10 de 0.9 sur N=500 (hasard 0.02) signifie que
    l'espace latent est réellement partagé entre les deux modalités.
    """
    a = z_img / (np.linalg.norm(z_img, axis=1, keepdims=True) + 1e-8)
    b = z_tab / (np.linalg.norm(z_tab, axis=1, keepdims=True) + 1e-8)
    sim = a @ b.T
    n = sim.shape[0]
    truth = np.arange(n)
    out: Dict[str, float] = {}
    for direction, s in (("i2t", sim), ("t2i", sim.T)):
        order = np.argsort(-s, axis=1)
        rank = np.argmax(order == truth[:, None], axis=1)
        for k in ks:
            out[f"R@{k}_{direction}"] = float((rank < k).mean())
        out[f"medr_{direction}"] = float(np.median(rank) + 1)
    out["chance_R@1"] = 1.0 / n
    return out


def bootstrap_ci(y_true: np.ndarray, score: np.ndarray, fn=auroc,
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Intervalle de confiance bootstrap — indispensable sur petits jeux de test."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(fn(y_true[idx], score[idx]))
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
