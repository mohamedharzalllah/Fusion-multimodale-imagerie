"""
ÉTAPE 4 — Fine-tuning sur la tâche aval (malignité ou BI-RADS).

Deux protocoles, tous deux standard en SSL :
  * linear_probe : encodeur GELÉ, on n'entraîne qu'une couche linéaire.
    -> mesure la qualité *intrinsèque* des représentations apprises.
  * full         : tout est dégelé, LR plus faible sur l'encodeur que sur la tête.
    -> mesure le gain pratique du pré-entraînement.

Et le point clé de la démonstration : la **courbe d'efficacité en labels**.
On refait le fine-tuning avec 1 %, 10 % puis 100 % des étiquettes. Le bénéfice
du SSL multimodal est toujours maximal dans le régime peu-annoté — c'est
précisément l'argument du sujet de thèse.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from ..data.datamodule import build_datasets
from ..data.semantic import resolve_semantic_emb
from ..eval.metrics import auroc, bootstrap_ci, classification_report
from ..models.multimodal import build_downstream_model
from ..utils import (AverageMeter, JsonLogger, cosine_warmup_scheduler,
                     get_device, load_config, save_checkpoint, set_seed)


def _subset_indices(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Sous-échantillon stratifié : conserve la prévalence de chaque classe."""
    if fraction >= 1.0:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        keep.append(idx[:max(2, int(round(fraction * len(idx))))])
    out = np.concatenate(keep)
    rng.shuffle(out)
    return out


def finetune(cfg, bundle=None, pretrained_ckpt: Optional[str] = None,
             modalities: Optional[str] = None, protocol: Optional[str] = None,
             label_fraction: float = 1.0, tag: str = "run",
             semantic_emb=None, verbose: bool = True) -> Dict[str, float]:
    device = get_device(cfg.device)
    set_seed(cfg.seed)
    modalities = modalities or cfg.finetune.modalities
    protocol = protocol or cfg.finetune.protocol
    out_dir = Path(cfg.out_dir) / "finetune" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonLogger(out_dir / "log.jsonl")

    if bundle is None:
        bundle = build_datasets(cfg, seed=cfg.seed)

    train_labels = bundle.train_ds.labels
    sub = _subset_indices(train_labels, label_fraction, cfg.seed)
    train_loader = bundle.loader("train", cfg.finetune.batch_size,
                                 num_workers=cfg.data.num_workers, subset_idx=sub)
    val_loader = bundle.loader("val", cfg.finetune.batch_size, shuffle=False,
                               num_workers=cfg.data.num_workers)
    test_loader = bundle.loader("test", cfg.finetune.batch_size, shuffle=False,
                                num_workers=cfg.data.num_workers)

    # ---------------- modèle ----------------
    if semantic_emb is None:
        semantic_emb = resolve_semantic_emb(cfg, bundle.schema, verbose=False)
    model = build_downstream_model(cfg, bundle.schema, bundle.n_classes, modalities,
                                   pretrained_ckpt, semantic_emb).to(device)
    if protocol == "linear_probe":
        model.freeze_encoders()

    # LR différencié : la tête est neuve, l'encodeur est déjà pré-entraîné
    head_params = list(model.head.parameters())
    enc_params = [p for n, p in model.named_parameters()
                  if not n.startswith("head") and p.requires_grad]
    groups = [{"params": head_params, "lr": cfg.finetune.head_lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": cfg.finetune.lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.finetune.weight_decay)

    steps = max(len(train_loader), 1)
    scheduler = cosine_warmup_scheduler(optimizer, warmup_steps=steps,
                                        total_steps=cfg.finetune.epochs * steps)
    weights = torch.tensor(bundle.class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    use_amp = bool(cfg.finetune.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    if verbose:
        print(f"[finetune:{tag}] modalities={modalities} protocol={protocol} "
              f"labels={len(sub)}/{len(train_labels)} ({label_fraction:.0%}) "
              f"ckpt={'oui' if pretrained_ckpt else 'non'}")

    best_auc, best_state = -1.0, None
    for epoch in range(cfg.finetune.epochs):
        model.train()
        meter = AverageMeter()
        for batch in train_loader:
            img = batch["image"].to(device, non_blocking=True)
            x_num = batch["x_num"].to(device, non_blocking=True)
            x_cat = batch["x_cat"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(img, x_num, x_cat, mask)
                loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # cf. pretrain.py : on n'avance le scheduler que si l'optimizer a
            # réellement fait un pas (AMP saute les pas à gradients inf/nan)
            if scaler.get_scale() >= scale_before:
                scheduler.step()
            meter.update(loss.item(), y.shape[0])

        val = predict_and_score(model, val_loader, device)
        logger.log(epoch=epoch, train_loss=meter.avg, **{f"val_{k}": v for k, v in val.items()})
        if verbose:
            print(f"  ep{epoch:03d} loss={meter.avg:.4f} val_auroc={val['auroc']:.4f} "
                  f"val_bacc={val['balanced_accuracy']:.4f}")
        # AUROC peut être NaN (split de validation mono-classe, fréquent à
        # label_fraction=1 %). NaN > x est toujours faux : sans repli,
        # best_state resterait None et on testerait la dernière époque.
        score = val["auroc"]
        if not np.isfinite(score):
            score = val["balanced_accuracy"]
            if verbose and epoch == 0:
                print("  [avert.] AUROC non définie en validation -> "
                      "sélection sur balanced_accuracy")
        if score > best_auc:
            best_auc = score
            best_state = copy.deepcopy(model.state_dict())

    # ---------------- test avec le meilleur checkpoint validation ----------------
    if best_state is not None:
        model.load_state_dict(best_state)
    y_true, logits = predict_raw(model, test_loader, device)
    test = classification_report(y_true, logits)
    # IC bootstrap : sur un test de quelques centaines d'exemples, un écart
    # d'AUROC de 0.02 entre deux méthodes est dans le bruit. Sans IC, on
    # sur-interprète systématiquement le classement.
    prob = _softmax(logits)
    if bundle.n_classes == 2:
        lo, hi = bootstrap_ci(y_true, prob[:, 1], auroc, n_boot=500, seed=cfg.seed)
    else:
        lo, hi = float("nan"), float("nan")
    result = {"tag": tag, "modalities": modalities, "protocol": protocol,
              "label_fraction": label_fraction, "pretrained": bool(pretrained_ckpt),
              "val_auroc": best_auc, "test_auroc_lo": lo, "test_auroc_hi": hi,
              **{f"test_{k}": v for k, v in test.items()}}
    logger.log(**result)
    save_checkpoint({"model": model.state_dict(), "result": result}, out_dir / "best.pt")
    if verbose:
        ci = "" if np.isnan(lo) else f" [IC95 {lo:.3f}-{hi:.3f}]"
        print(f"[finetune:{tag}] TEST auroc={test['auroc']:.4f}{ci} "
              f"auprc={test['auprc']:.4f} bacc={test['balanced_accuracy']:.4f}")
    return result


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@torch.no_grad()
def predict_and_score(model, loader, device) -> Dict[str, float]:
    model.eval()
    L, Y = [], []
    for batch in loader:
        logits = model(batch["image"].to(device), batch["x_num"].to(device),
                       batch["x_cat"].to(device), batch["mask"].to(device))
        L.append(logits.float().cpu().numpy())
        Y.append(batch["label"].numpy())
    return classification_report(np.concatenate(Y), np.concatenate(L))


@torch.no_grad()
def predict_raw(model, loader, device):
    """Renvoie (y_true, logits) — utile pour les IC bootstrap et les courbes."""
    model.eval()
    L, Y = [], []
    for batch in loader:
        logits = model(batch["image"].to(device), batch["x_num"].to(device),
                       batch["x_cat"].to(device), batch["mask"].to(device))
        L.append(logits.float().cpu().numpy())
        Y.append(batch["label"].numpy())
    return np.concatenate(Y), np.concatenate(L)


def main():
    ap = argparse.ArgumentParser(description="Fine-tuning sur la tâche aval")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=None, help="checkpoint SSL (facultatif)")
    ap.add_argument("--modalities", default=None, choices=["image", "tabular", "both"])
    ap.add_argument("--protocol", default=None, choices=["linear_probe", "full"])
    ap.add_argument("--label-fraction", type=float, default=1.0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    import yaml as _y
    overrides = {k: _y.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
    cfg = load_config(args.config, overrides)
    finetune(cfg, pretrained_ckpt=args.ckpt, modalities=args.modalities,
             protocol=args.protocol, label_fraction=args.label_fraction, tag=args.tag)


if __name__ == "__main__":
    main()
