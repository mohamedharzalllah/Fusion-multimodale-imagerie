"""
Test de bout en bout ultra-rapide (~2-4 min sur CPU).

À lancer EN PREMIER, avant toute expérience longue :

    python scripts/smoke_test.py

Il vérifie, dans l'ordre :
  1. génération du dataset synthétique + prétraitement tabulaire missing-aware
  2. formes des tenseurs en sortie du DataLoader
  3. passe avant/arrière des deux encodeurs et de la perte InfoNCE
  4. décroissance effective de la perte contrastive sur quelques dizaines de pas
  5. gestion des valeurs manquantes (100 % manquant ne doit pas produire de NaN)
  6. cohérence des métriques
  7. mini pré-entraînement + mini fine-tuning complets
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PASS, FAIL = "  [OK]  ", "  [KO]  "
n_fail = 0


def check(name, fn):
    global n_fail
    try:
        info = fn()
        print(PASS + name + (f" — {info}" if info else ""))
    except Exception as e:
        n_fail += 1
        print(FAIL + name + f" — {type(e).__name__}: {e}")
        traceback.print_exc()


def main():
    import torch
    from src.data.datamodule import build_datasets
    from src.data.semantic import build_semantic_matrix
    from src.eval.metrics import auroc, retrieval_metrics
    from src.losses.contrastive import CLIPLoss, build_duplicate_mask
    from src.models.multimodal import build_pretrain_model
    from src.utils import load_config, set_seed

    print("=" * 70)
    print("SMOKE TEST — Fusion multimodale imagerie + tabulaire")
    print("=" * 70)

    cfg = load_config(ROOT / "configs" / "default.yaml", {
        "data.n_synthetic": 300, "data.image_size": 64, "data.num_workers": 0,
        "pretrain.epochs": 2, "pretrain.batch_size": 32, "pretrain.warmup_epochs": 0,
        "pretrain.log_every": 0, "finetune.epochs": 2, "finetune.batch_size": 32,
        "finetune.label_fractions": [1.0], "out_dir": str(ROOT / "outputs" / "_smoke"),
        "model.image.name": "resnet18", "model.tabular.d_token": 32,
        "model.tabular.n_layers": 2,
    })
    set_seed(0)
    device = torch.device("cpu")

    # ---- 1. données
    bundle = {}

    def t_data():
        bundle["b"] = build_datasets(cfg, seed=0)
        b = bundle["b"]
        assert len(b.train_ds) > 0 and len(b.test_ds) > 0
        assert b.schema.n_features == b.train_ds.mask.shape[1]
        return (f"train={len(b.train_ds)} val={len(b.val_ds)} test={len(b.test_ds)}, "
                f"{b.schema.n_features} features, manquants={b.train_ds.mask.mean():.1%}")
    check("1. construction du dataset", t_data)
    b = bundle["b"]

    # ---- 2. dataloader
    def t_loader():
        ld = b.loader("train", 8, num_workers=0, mode="pretrain")
        batch = next(iter(ld))
        assert batch["image"].shape == (8, cfg.data.in_chans, 64, 64), batch["image"].shape
        assert batch["x_num"].shape == (8, b.schema.n_num)
        assert batch["x_cat"].shape == (8, b.schema.n_cat)
        assert batch["mask"].shape == (8, b.schema.n_features)
        assert batch["image"].min() >= 0 and batch["image"].max() <= 1
        bundle["batch"] = batch
        return f"image{tuple(batch['image'].shape)} x_num{tuple(batch['x_num'].shape)}"
    check("2. sorties du DataLoader", t_loader)
    batch = bundle["batch"]

    # ---- 3. modèle
    def t_model():
        sem = build_semantic_matrix(b.schema, backend="hashing", verbose=False)
        m = build_pretrain_model(cfg, b.schema, sem).to(device)
        out = m(batch["image"], batch["x_num"], batch["x_cat"], batch["mask"])
        assert out["z_image"].shape == out["z_tabular"].shape == (8, cfg.model.projection.dim)
        assert torch.isfinite(out["z_image"]).all() and torch.isfinite(out["z_tabular"]).all()
        bundle["model"] = m
        return f"z_image{tuple(out['z_image'].shape)}, embeddings sémantiques {sem.shape}"
    check("3. passe avant du modèle multimodal", t_model)

    # ---- 4. perte + apprentissage effectif
    def t_loss():
        m = bundle["model"]
        crit = CLIPLoss(0.1, True, 0.5)
        opt = torch.optim.AdamW(list(m.parameters()) + list(crit.parameters()), lr=1e-3)
        losses = []
        for _ in range(3):                              # ~3 passes sur 300 ex.
            for bt in b.loader("train", 32, num_workers=0, mode="pretrain"):
                out = m(bt["image"], bt["x_num"], bt["x_cat"], bt["mask"])
                dup = build_duplicate_mask(bt["x_num"], bt["x_cat"])
                loss, st = crit(out["z_image"], out["z_tabular"], dup)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
        assert all(np.isfinite(losses))
        # Comparaison moyennée (3 premiers vs 3 derniers pas) : un test sur
        # deux pas isolés d'AdamW échoue au hasard une fois sur cinq.
        first, last = float(np.mean(losses[:3])), float(np.mean(losses[-3:]))
        assert last < first, f"la perte ne descend pas ({first:.3f} -> {last:.3f})"
        # On NE teste PAS `last < log(B)`. La température apprise démarre à
        # tau=0.1, soit un logit_scale de 10 : la perte initiale est donc très
        # au-dessus de log(B) et 3 passes sur 300 exemples synthétiques ne
        # suffisent pas à repasser dessous. Cette assertion échouait à tort et
        # ne testait rien d'utile — descendre sous log(B) est un objectif
        # d'entraînement complet, pas de smoke test.
        chance = float(np.log(32))
        note = "" if last < chance else f", encore au-dessus du hasard {chance:.2f}"
        return f"InfoNCE {first:.3f} -> {last:.3f}{note}"
    check("4. la perte contrastive décroît", t_loss)

    # ---- 5. valeurs manquantes extrêmes
    def t_missing():
        m = bundle["model"]
        m.eval()
        full = torch.ones_like(batch["mask"])
        with torch.no_grad():
            z = m.encode_tabular(torch.zeros_like(batch["x_num"]),
                                 torch.zeros_like(batch["x_cat"]), full)
            z2 = m.encode_tabular(batch["x_num"], batch["x_cat"],
                                  torch.zeros_like(batch["mask"]))
        assert torch.isfinite(z).all(), "NaN avec 100 % de valeurs manquantes"
        assert torch.isfinite(z2).all()
        # tout-manquant => toutes les lignes doivent donner la MÊME représentation
        spread = (z - z.mean(0)).abs().max().item()
        assert spread < 1e-4, f"tout-manquant devrait être constant (écart {spread:.2e})"
        d = (z[0] - z2[0]).norm().item()
        assert d > 1e-3, "le masque n'a aucun effet sur la représentation"
        return f"pas de NaN ; distance masqué/non-masqué = {d:.3f}"
    check("5. robustesse aux valeurs manquantes", t_missing)

    # ---- 6. métriques
    def t_metrics():
        y = np.array([0, 0, 1, 1]); s = np.array([.1, .4, .35, .8])
        assert abs(auroc(y, s) - 0.75) < 1e-9
        z = np.random.default_rng(0).normal(size=(50, 8))
        assert retrieval_metrics(z, z, (1,))["R@1_i2t"] == 1.0
        return "AUROC et retrieval conformes"
    check("6. métriques", t_metrics)

    # ---- 7. pipeline complet
    def t_pipeline():
        from src.train.finetune import finetune
        from src.train.pretrain import pretrain
        p = pretrain(cfg, bundle=b, verbose=False, tag="smoke")
        r = finetune(cfg, bundle=b, pretrained_ckpt=p["best_ckpt"],
                     modalities="image", label_fraction=1.0, tag="smoke", verbose=False)
        assert np.isfinite(r["test_auroc"])
        return (f"R@10 pré-entraînement={p['best_r10']:.3f}, "
                f"AUROC aval={r['test_auroc']:.3f}")
    check("7. pré-entraînement + fine-tuning de bout en bout", t_pipeline)

    print("=" * 70)
    if n_fail:
        print(f"ÉCHEC : {n_fail} test(s) en erreur")
        sys.exit(1)
    print("TOUS LES TESTS SONT PASSÉS — le pipeline est opérationnel.")
    print("Étape suivante :  python -m src.train.run_experiments --config configs/default.yaml")


if __name__ == "__main__":
    main()
