"""
ÉTAPE 3 — Pré-entraînement auto-supervisé contrastif image <-> tabulaire.

Aucun label de malignité n'est utilisé ici. Le signal d'apprentissage vient
uniquement de l'appariement naturel "cette image appartient à ce patient".

Boucle, pour chaque batch de B paires :
  1. vue image      = augmentation forte de l'image
  2. vue tabulaire  = corruption de la ligne clinique (marginales empiriques)
  3. embeddings projetés z_I, z_T normalisés L2
  4. InfoNCE bidirectionnel : la diagonale de z_I z_T^T doit dominer sa ligne
     ET sa colonne
  5. (optionnel) terme SimCLR intra-image, pondéré par `intra_image_weight`

On suit `acc_i2t` (taux de fois où la bonne colonne est retrouvée dans le
batch) : c'est le meilleur indicateur en direct que l'alignement progresse.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data.datamodule import build_datasets
from ..data.identifiability import format_report, identifiability_report
from ..data.semantic import resolve_semantic_emb
from ..eval.metrics import retrieval_metrics
from ..losses.contrastive import (CLIPLoss, NTXentLoss, build_duplicate_mask,
                                  tabular_similarity)
from ..models.multimodal import build_pretrain_model
from ..utils import (AverageMeter, JsonLogger, cosine_warmup_scheduler,
                     count_params, get_device, load_config, save_checkpoint,
                     save_config, set_seed)


def pretrain(cfg, bundle=None, semantic_emb=None, verbose: bool = True,
             mode: str = "multimodal", tag: str = "multimodal"):
    """
    mode = "multimodal"    -> InfoNCE image <-> tabulaire  (la méthode du projet)
    mode = "simclr_image"  -> SimCLR image seule           (baseline SSL unimodale)
    """
    device = get_device(cfg.device)
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir) / "pretrain" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonLogger(out_dir / "log.jsonl")

    # ---------------- données ----------------
    if bundle is None:
        bundle = build_datasets(cfg, seed=cfg.seed)
    train_loader = bundle.loader("train", cfg.pretrain.batch_size,
                                 num_workers=cfg.data.num_workers, mode="pretrain")
    val_loader = bundle.loader("val", cfg.pretrain.batch_size, shuffle=False,
                               num_workers=cfg.data.num_workers, mode="supervised")

    # ---------------- modèle ----------------
    if semantic_emb is None:
        semantic_emb = resolve_semantic_emb(cfg, bundle.schema, verbose=verbose)
    model = build_pretrain_model(cfg, bundle.schema, semantic_emb).to(device)
    use_soft = bool(cfg.pretrain.get("soft_targets", False))
    criterion = CLIPLoss(temperature=cfg.pretrain.temperature,
                         learnable=cfg.pretrain.learnable_temperature,
                         lambda_i2t=cfg.pretrain.lambda_i2t,
                         soft_targets=use_soft,
                         soft_temperature=float(
                             cfg.pretrain.get("soft_temperature", 0.25))).to(device)
    ntxent = NTXentLoss(cfg.pretrain.temperature)

    # --- Diagnostic d'identifiabilité, AVANT d'entraîner -------------------
    # Il donne le plafond de R@1 imposé par les lignes tabulaires dupliquées.
    # Sans lui on compare le R@1 mesuré au hasard 1/N, ce qui fait croire à un
    # succès là où le modèle est peut-être déjà au maximum du possible — ou
    # l'inverse.
    ident = None
    if mode != "simclr_image":
        ident = identifiability_report(
            bundle.train_ds.x_num, bundle.train_ds.x_cat, bundle.train_ds.mask,
            batch_size=cfg.pretrain.batch_size,
            eval_size=min(len(bundle.val_ds), 2000), ks=cfg.eval.retrieval_k,
            seed=cfg.seed)
        if verbose:
            print(format_report(ident, cfg.pretrain.batch_size,
                                min(len(bundle.val_ds), 2000)))
            if not use_soft and ident["entropy_bits"] < 0.8 * ident["bits_needed_batch"]:
                print("  [note] soft_targets est DÉSACTIVÉ alors que le vecteur "
                      "tabulaire n'identifie pas le patient.")
            print()

    params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.pretrain.lr,
                                  weight_decay=cfg.pretrain.weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    scheduler = cosine_warmup_scheduler(
        optimizer,
        warmup_steps=cfg.pretrain.warmup_epochs * steps_per_epoch,
        total_steps=cfg.pretrain.epochs * steps_per_epoch)
    use_amp = bool(cfg.pretrain.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    if verbose:
        print(f"[pretrain] device={device} | image_enc={count_params(model.image_encoder):,} "
              f"params | tab_enc={count_params(model.tabular_encoder):,} params")
        print(f"[pretrain] {len(bundle.train_ds)} paires d'entraînement, "
              f"{steps_per_epoch} pas/époque")

    best_score, best_path = -1.0, out_dir / "best.pt"
    for epoch in range(cfg.pretrain.epochs):
        model.train()
        meters = {k: AverageMeter() for k in
                  ("loss", "loss_i2t", "loss_t2i", "acc_i2t", "acc_t2i")}
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            img = batch["image"].to(device, non_blocking=True)
            x_num = batch["x_num"].to(device, non_blocking=True)
            x_cat = batch["x_cat"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            # calculés hors autocast (booléens / float32) et seulement quand la
            # perte croisée en a besoin
            dup = sim = None
            if mode != "simclr_image":
                if use_soft:
                    sim = tabular_similarity(x_num, x_cat, mask)
                else:
                    dup = build_duplicate_mask(x_num, x_cat)

            with torch.amp.autocast(device.type, enabled=use_amp):
                if mode == "simclr_image":
                    # baseline SSL unimodale : deux vues de la MÊME image.
                    # On n'appelle PAS l'encodeur tabulaire du tout — sinon ses
                    # BatchNorm se mettraient à jour et la baseline ne serait
                    # plus honnêtement "image seule".
                    z1 = model.proj_image(model.encode_image(img))
                    z2 = model.proj_image(model.encode_image(
                        batch["image2"].to(device, non_blocking=True)))
                    loss = ntxent(z1, z2)
                    stats = {"loss_i2t": loss.item(), "loss_t2i": 0.0,
                             "acc_i2t": 0.0, "acc_t2i": 0.0,
                             "temperature": cfg.pretrain.temperature}
                else:
                    out = model(img, x_num, x_cat, mask)
                    loss, stats = criterion(out["z_image"], out["z_tabular"],
                                            duplicate_mask=dup,
                                            tabular_similarity=sim)
                    if cfg.pretrain.intra_image_weight > 0:
                        z2 = model.proj_image(model.encode_image(
                            batch["image2"].to(device, non_blocking=True)))
                        loss = loss + cfg.pretrain.intra_image_weight * ntxent(
                            out["z_image"], z2)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # En AMP, `scaler.step` SAUTE l'optimizer quand les gradients
            # contiennent des inf/nan (fréquent aux premiers pas) et réduit
            # alors l'échelle. Avancer le scheduler dans ce cas décale le
            # planning de LR et déclenche l'avertissement PyTorch
            # « lr_scheduler.step() before optimizer.step() ».
            if scaler.get_scale() >= scale_before:
                scheduler.step()

            bs = img.shape[0]
            meters["loss"].update(loss.item(), bs)
            for k in ("loss_i2t", "loss_t2i", "acc_i2t", "acc_t2i"):
                meters[k].update(stats[k], bs)

            if verbose and cfg.pretrain.log_every and step % cfg.pretrain.log_every == 0:
                print(f"  ep{epoch:03d} step{step:04d} "
                      f"loss={meters['loss'].avg:.4f} acc_i2t={meters['acc_i2t'].avg:.3f} "
                      f"tau={stats['temperature']:.3f}")

        # ---------------- validation ----------------
        # En mode multimodal on suit le retrieval croisé. En mode SimCLR il n'y
        # a pas d'alignement croisé à mesurer : on retient simplement la
        # dernière époque, et on ne rapporte PAS de faux R@10.
        val_stats = ({} if mode == "simclr_image"
                     else evaluate_alignment(model, val_loader, device,
                                             ks=cfg.eval.retrieval_k))
        rec = {"epoch": epoch, "time_s": time.time() - t0,
               **{k: m.avg for k, m in meters.items()}, **val_stats}
        logger.log(**rec)
        if verbose:
            # R@10 est affiché sur son PLAFOND, pas seul : « 0.13 » ne veut rien
            # dire, « 0.13/0.42 » situe immédiatement où en est l'alignement.
            extra = ""
            if mode != "simclr_image":
                r10 = val_stats.get("R@10_i2t", float("nan"))
                cap = (ident or {}).get("max_R@10_eval")
                cap_s = f"/{cap:.2f}" if cap else ""
                extra = (f" | val R@1={val_stats.get('R@1_i2t', float('nan')):.3f} "
                         f"R@10={r10:.3f}{cap_s}")
            print(f"[pretrain:{tag}] ep{epoch:03d} loss={rec['loss']:.4f} "
                  f"acc_i2t={rec['acc_i2t']:.3f}{extra} ({rec['time_s']:.1f}s)")

        # critère de sélection : le plus grand K disponible dans retrieval_k
        key = (float(epoch) if mode == "simclr_image"
               else max((v for k, v in val_stats.items()
                         if k.startswith("R@") and k.endswith("_i2t")), default=0.0))
        if key > best_score:
            best_score = key
            save_checkpoint({"model": model.state_dict(), "epoch": epoch,
                             "cfg": dict(cfg), "val": val_stats}, best_path)

    save_checkpoint({"model": model.state_dict(), "epoch": cfg.pretrain.epochs - 1,
                     "cfg": dict(cfg)}, out_dir / "last.pt")
    save_config(cfg, out_dir / "config.yaml")
    best_r10 = float("nan") if mode == "simclr_image" else best_score
    cap10 = (ident or {}).get("max_R@10_eval")
    if verbose:
        if mode == "simclr_image":
            label = "dernière époque conservée"
        else:
            frac = f" ({best_score / cap10:.0%} du plafond)" if cap10 else ""
            label = (f"meilleur R@10 (val) = {best_score:.3f}"
                     + (f" / plafond {cap10:.3f}{frac}" if cap10 else ""))
        print(f"[pretrain:{tag}] terminé — {label} -> {best_path}")
    return {"best_ckpt": str(best_path), "last_ckpt": str(out_dir / "last.pt"),
            "best_r10": best_r10, "max_r10": cap10, "mode": mode,
            "soft_targets": use_soft,
            "identifiability": {k: float(v) for k, v in (ident or {}).items()}}


@torch.no_grad()
def evaluate_alignment(model, loader, device, ks=(1, 5, 10), max_samples: int = 2000):
    """Encode le split de validation et mesure le retrieval croisé."""
    model.eval()
    Z_i, Z_t = [], []
    seen = 0
    for batch in loader:
        img = batch["image"].to(device)
        out = model(img, batch["x_num"].to(device), batch["x_cat"].to(device),
                    batch["mask"].to(device))
        Z_i.append(F.normalize(out["z_image"], dim=-1).float().cpu().numpy())
        Z_t.append(F.normalize(out["z_tabular"], dim=-1).float().cpu().numpy())
        seen += img.shape[0]
        if seen >= max_samples:
            break
    z_i, z_t = np.concatenate(Z_i), np.concatenate(Z_t)
    ks = [k for k in ks if k < len(z_i)] or [1]
    return retrieval_metrics(z_i, z_t, ks)


def main():
    ap = argparse.ArgumentParser(description="Pré-entraînement contrastif multimodal")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[],
                    help="overrides, ex: pretrain.epochs=5 model.image.name=resnet50")
    args = ap.parse_args()
    overrides = {}
    for s in args.set:
        k, v = s.split("=", 1)
        overrides[k] = _cast(v)
    cfg = load_config(args.config, overrides)
    pretrain(cfg)


def _cast(v: str):
    import yaml as _y
    return _y.safe_load(v)


if __name__ == "__main__":
    main()
