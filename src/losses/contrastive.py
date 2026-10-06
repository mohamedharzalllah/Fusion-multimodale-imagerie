"""
Pertes contrastives.

`CLIPLoss` : InfoNCE bidirectionnel entre deux jeux d'embeddings appariés.
Pour un batch de B paires (image_i, tableau_i) :

        z^I, z^T  in R^{B x d},  normalisés L2
        logits    = (z^I @ z^T.T) / tau
        L_{I->T}  = CE(logits,   [0..B-1])      # retrouver le bon tableau
        L_{T->I}  = CE(logits.T, [0..B-1])      # retrouver la bonne image
        L         = lambda * L_{I->T} + (1-lambda) * L_{T->I}

La diagonale = les vraies paires (positifs), tout le reste du batch = négatifs.
Le modèle est donc forcé à encoder, dans l'image, l'information qui identifie
le patient dans le tableau (âge, densité, type de lésion...) et réciproquement.

Subtilité propre au tabulaire (soulignée par Hager et al.) : deux patients
distincts peuvent avoir des vecteurs cliniques identiques (surtout après
discrétisation). Ce sont des **faux négatifs** : la perte les pousse à
s'éloigner alors qu'ils sont sémantiquement identiques. `mask_duplicates`
retire ces paires du dénominateur.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    """
    InfoNCE bidirectionnel, avec deux traitements des faux positifs/négatifs.

    `duplicate_mask` (doublons EXACTS) : les paires hors-diagonale identiques
    sont mises à -inf, donc retirées du dénominateur.

    `soft_targets` : quand le vecteur tabulaire n'identifie pas le patient — le
    cas de CBIS-DDSM, cf. `src/data/identifiability.py` — la cible one-hot est
    tout simplement fausse. On la remplace alors par une cible **pondérée par la
    similarité tabulaire** : si trois lignes du batch sont identiques, la masse
    de probabilité est répartie entre elles au lieu d'exiger de les distinguer.
    C'est une cross-entropie à cible douce, qui redevient exactement l'InfoNCE
    classique quand toutes les lignes sont distinctes.
    """

    def __init__(self, temperature: float = 0.1, learnable: bool = True,
                 lambda_i2t: float = 0.5, soft_targets: bool = False,
                 soft_temperature: float = 0.25):
        super().__init__()
        self.lambda_i2t = lambda_i2t
        self.soft_targets = soft_targets
        self.soft_temperature = soft_temperature
        logit_scale = torch.tensor(np.log(1.0 / temperature), dtype=torch.float32)
        if learnable:
            self.logit_scale = nn.Parameter(logit_scale)
        else:
            self.register_buffer("logit_scale", logit_scale)

    @property
    def temperature(self) -> float:
        return float(1.0 / self.logit_scale.exp().item())

    def forward(self, z_img: torch.Tensor, z_tab: torch.Tensor,
                duplicate_mask: torch.Tensor | None = None,
                tabular_similarity: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, dict]:
        z_img = F.normalize(z_img, dim=-1)
        z_tab = F.normalize(z_tab, dim=-1)

        # borne le scale pour éviter la divergence (astuce OpenCLIP)
        scale = self.logit_scale.clamp(max=np.log(100.0)).exp()
        logits = scale * z_img @ z_tab.t()                     # (B, B)
        B = logits.shape[0]
        target = torch.arange(B, device=logits.device)

        if self.soft_targets and tabular_similarity is not None:
            # cible douce : masse répartie sur les lignes tabulairement proches,
            # la vraie paire gardant toujours le poids maximal
            q = _soft_target(tabular_similarity, self.soft_temperature)
            loss_i2t = _soft_cross_entropy(logits, q)
            loss_t2i = _soft_cross_entropy(logits.t(), q.t())
        else:
            if duplicate_mask is not None:
                logits = logits.masked_fill(duplicate_mask, float("-inf"))
            loss_i2t = F.cross_entropy(logits, target)
            loss_t2i = F.cross_entropy(logits.t(), target)

        loss = self.lambda_i2t * loss_i2t + (1 - self.lambda_i2t) * loss_t2i

        with torch.no_grad():
            acc_i2t = (logits.argmax(dim=1) == target).float().mean()
            acc_t2i = (logits.argmax(dim=0) == target).float().mean()
        return loss, {"loss_i2t": loss_i2t.item(), "loss_t2i": loss_t2i.item(),
                      "acc_i2t": acc_i2t.item(), "acc_t2i": acc_t2i.item(),
                      "temperature": float(1.0 / scale.item())}


def _soft_target(sim: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Distribution cible (B, B) à partir d'une similarité tabulaire dans [0, 1].

    La diagonale est forcée à 1 (la vraie paire reste toujours le mode), puis on
    applique un softmax de température `tau` : petit tau -> cible presque
    one-hot, grand tau -> masse largement partagée avec les lignes similaires.
    """
    sim = sim.float().clone()
    sim.fill_diagonal_(1.0)
    return F.softmax(sim / max(tau, 1e-6), dim=1)


def _soft_cross_entropy(logits: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return -(q * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


class NTXentLoss(nn.Module):
    """SimCLR classique : deux vues augmentées de la MÊME modalité."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.t = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B = z1.shape[0]
        z = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)      # (2B, d)
        sim = z @ z.t() / self.t
        # exclut la similarité d'un élément avec lui-même (hors-place : évite
        # les soucis d'autograd et reste compatible AMP)
        eye = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(eye, float("-inf"))
        target = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
        return F.cross_entropy(sim, target)


def build_duplicate_mask(x_num: torch.Tensor, x_cat: torch.Tensor,
                         tol: float = 1e-6) -> torch.Tensor:
    """
    True là où deux lignes tabulaires DIFFÉRENTES du batch sont identiques
    (faux négatifs). La diagonale reste False : ce sont les vrais positifs.
    """
    B = x_cat.shape[0] if x_cat.numel() else x_num.shape[0]
    same_num = torch.ones(B, B, dtype=torch.bool, device=x_num.device)
    if x_num.shape[1]:
        # float32 explicite : cdist n'est pas fiable en half sous AMP
        xn = x_num.detach().float()
        same_num = (torch.cdist(xn, xn, p=1) < tol)
    same_cat = torch.ones(B, B, dtype=torch.bool, device=x_num.device)
    if x_cat.shape[1]:
        same_cat = (x_cat.unsqueeze(1) == x_cat.unsqueeze(0)).all(dim=-1)
    dup = same_num & same_cat
    dup.fill_diagonal_(False)
    return dup


def tabular_similarity(x_num: torch.Tensor, x_cat: torch.Tensor,
                       mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Similarité tabulaire dans [0, 1] : fraction de features qui coïncident.

    Simple et volontairement grossière — c'est exactement la quantité qui décide
    si deux lignes sont interchangeables du point de vue de la perte. Les
    catégorielles se comparent par égalité, les numériques (déjà standardisées)
    par une exponentielle de leur écart. Les features manquantes des deux côtés
    comptent comme identiques : deux dossiers également incomplets *sont* peu
    distinguables.
    """
    B = x_cat.shape[0] if x_cat.numel() else x_num.shape[0]
    device = x_num.device
    agree = torch.zeros(B, B, device=device)
    n_feat = 0

    if x_cat.numel() and x_cat.shape[1]:
        eq = (x_cat.unsqueeze(1) == x_cat.unsqueeze(0)).float().sum(-1)
        agree = agree + eq
        n_feat += x_cat.shape[1]

    if x_num.numel() and x_num.shape[1]:
        xn = x_num.detach().float()
        d = (xn.unsqueeze(1) - xn.unsqueeze(0)).abs()
        agree = agree + torch.exp(-d).sum(-1)      # 1 si égal, ->0 si éloigné
        n_feat += x_num.shape[1]

    if n_feat == 0:
        return torch.eye(B, device=device)
    return (agree / n_feat).clamp(0.0, 1.0)
