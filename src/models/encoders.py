"""
Les deux encodeurs + les têtes de projection.

  ImageEncoder    : CNN/ViT (timm), sortie = vecteur de dimension d_img
  TabularEncoder  : deux variantes
                      - MLP naïf (baseline, "concaténation naïve")
                      - FT-Transformer missing-aware (proposition du projet)
  ProjectionHead  : MLP 2 couches vers l'espace contrastif partagé
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 1. Encodeur image
# ===========================================================================
class ImageEncoder(nn.Module):
    """Backbone timm sans tête de classification (num_classes=0)."""

    def __init__(self, name: str = "resnet18", pretrained: bool = False,
                 in_chans: int = 1, image_size: int = 128):
        super().__init__()
        import timm
        kwargs = dict(pretrained=pretrained, num_classes=0, in_chans=in_chans)
        # les ViT timm exigent une taille d'image compatible avec le patch size
        if "vit" in name or "swin" in name or "deit" in name:
            kwargs["img_size"] = image_size
        self.backbone = timm.create_model(name, **kwargs)
        self.out_dim: int = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ===========================================================================
# 2. Encodeur tabulaire
# ===========================================================================
class FeatureTokenizer(nn.Module):
    """
    Transforme une ligne tabulaire hétérogène en une séquence de tokens.

    Pour chaque feature j on produit un vecteur de dimension d :

        numérique   :  t_j = x_j * W_j + b_j
        catégoriel  :  t_j = Embedding_j[k]
        MANQUANTE   :  t_j = m_j            <- embedding appris, PROPRE à j

    puis on ajoute un *field embedding* e_j (identité de la colonne) et,
    optionnellement, un embedding **sémantique** issu du texte décrivant la
    colonne ("age of the patient in years..."). Ce dernier point est ce qui
    permet de parler de "sémantique des données tabulaires" : deux colonnes de
    sens proche démarrent avec des représentations proches, et un modèle
    entraîné sur un dataset peut être transféré vers un autre schéma tabulaire.
    """

    def __init__(self, n_num: int, cat_cardinalities: List[int], d_token: int,
                 semantic_emb: Optional[np.ndarray] = None):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(cat_cardinalities)
        self.n_features = n_num + self.n_cat
        self.d_token = d_token

        # --- numériques : un poids et un biais par colonne
        if n_num:
            self.num_weight = nn.Parameter(torch.empty(n_num, d_token))
            self.num_bias = nn.Parameter(torch.empty(n_num, d_token))
            nn.init.normal_(self.num_weight, std=d_token ** -0.5)
            nn.init.normal_(self.num_bias, std=d_token ** -0.5)

        # --- catégorielles : une table d'embedding par colonne
        #     (le dernier indice de chaque table est déjà le marqueur MISSING,
        #      mais on le neutralise pour utiliser un embedding MISSING unifié)
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(c, d_token) for c in cat_cardinalities])
        for emb in self.cat_embeddings:
            nn.init.normal_(emb.weight, std=d_token ** -0.5)

        # --- embedding "valeur manquante", un par feature
        self.missing_emb = nn.Parameter(torch.randn(self.n_features, d_token) * d_token ** -0.5)

        # --- field embedding (identité de colonne)
        self.field_emb = nn.Parameter(torch.randn(self.n_features, d_token) * d_token ** -0.5)

        # --- embedding sémantique optionnel (figé, projeté)
        self.semantic_proj = None
        if semantic_emb is not None:
            sem = torch.as_tensor(np.asarray(semantic_emb), dtype=torch.float32)
            assert sem.shape[0] == self.n_features, (
                f"semantic_emb doit avoir {self.n_features} lignes, reçu {sem.shape[0]}")
            self.register_buffer("semantic_emb", F.normalize(sem, dim=-1))
            self.semantic_proj = nn.Linear(sem.shape[1], d_token)

        # --- token [CLS] : agrège toute la ligne
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * d_token ** -0.5)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """(B, n_num), (B, n_cat), (B, n_feat) -> (B, 1 + n_feat, d)"""
        B = mask.shape[0]
        toks = []
        if self.n_num:
            toks.append(x_num.unsqueeze(-1) * self.num_weight + self.num_bias)
        if self.n_cat:
            toks.append(torch.stack(
                [emb(x_cat[:, j]) for j, emb in enumerate(self.cat_embeddings)], dim=1))
        t = torch.cat(toks, dim=1) if toks else torch.zeros(B, 0, self.d_token,
                                                            device=mask.device)

        # substitution des features manquantes par leur embedding dédié
        t = torch.where(mask.unsqueeze(-1), self.missing_emb.unsqueeze(0).expand(B, -1, -1), t)

        # identité + sémantique de la colonne
        t = t + self.field_emb.unsqueeze(0)
        if self.semantic_proj is not None:
            t = t + self.semantic_proj(self.semantic_emb).unsqueeze(0)

        return torch.cat([self.cls_token.expand(B, -1, -1), t], dim=1)


class FTTransformerTabularEncoder(nn.Module):
    """
    Transformer sur la séquence de tokens de features.

    Pourquoi un Transformer plutôt qu'un MLP sur le vecteur concaténé ?
      * l'attention modélise explicitement les **interactions entre colonnes**
        (ex. "densité D" + "marges spiculées" pèse plus que la somme des deux) ;
      * elle permet d'**ignorer proprement** un token manquant plutôt que de le
        remplacer par une moyenne trompeuse ;
      * elle rend le modèle **agnostique au nombre de colonnes**, condition pour
        transférer entre schémas tabulaires différents.
    """

    def __init__(self, n_num: int, cat_cardinalities: List[int], d_token: int = 64,
                 n_layers: int = 3, n_heads: int = 4, dropout: float = 0.1,
                 semantic_emb: Optional[np.ndarray] = None,
                 attend_missing: bool = True):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num, cat_cardinalities, d_token, semantic_emb)
        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=4 * d_token,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        # enable_nested_tensor=False : avec norm_first=True le chemin « nested
        # tensor » est de toute façon désactivé par PyTorch, mais il émet un
        # UserWarning à chaque construction. On le désactive explicitement.
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers,
                                                 enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_token)
        self.out_dim = d_token
        # si False, les tokens manquants sont masqués dans l'attention
        # (variante "drop") plutôt que représentés par [MISSING] (variante "learn")
        self.attend_missing = attend_missing

    def forward(self, x_num, x_cat, mask) -> torch.Tensor:
        t = self.tokenizer(x_num, x_cat, mask)
        key_padding = None
        if not self.attend_missing:
            cls_col = torch.zeros(mask.shape[0], 1, dtype=torch.bool, device=mask.device)
            key_padding = torch.cat([cls_col, mask], dim=1)
            # garde-fou : si TOUTES les features sont manquantes, la ligne
            # n'aurait plus que le [CLS] comme clé visible — c'est valide, mais
            # on désactive alors le masquage pour éviter les NaN d'attention.
            all_missing = mask.all(dim=1, keepdim=True)
            key_padding = key_padding & ~all_missing
        h = self.transformer(t, src_key_padding_mask=key_padding)
        return self.norm(h[:, 0])          # représentation [CLS]


class NaiveMLPTabularEncoder(nn.Module):
    """
    Baseline "concaténation naïve" : on empile [valeurs numériques imputées à 0,
    one-hot des catégorielles, masque binaire] et on passe un MLP.
    C'est le point de comparaison que le projet doit battre.
    """

    def __init__(self, n_num: int, cat_cardinalities: List[int], d_token: int = 64,
                 n_layers: int = 3, dropout: float = 0.1, **_):
        super().__init__()
        self.n_num = n_num
        self.cards = cat_cardinalities
        in_dim = n_num + sum(cat_cardinalities) + n_num + len(cat_cardinalities)
        dims = [in_dim] + [4 * d_token] * (n_layers - 1) + [d_token]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1])]
            if i < len(dims) - 2:
                layers += [nn.BatchNorm1d(dims[i + 1]), nn.GELU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)
        self.out_dim = d_token

    def forward(self, x_num, x_cat, mask) -> torch.Tensor:
        feats = [x_num]
        for j, c in enumerate(self.cards):
            feats.append(F.one_hot(x_cat[:, j], num_classes=c).float())
        feats.append(mask.float())
        return self.net(torch.cat(feats, dim=1))


def build_tabular_encoder(arch: str, n_num: int, cat_cardinalities: List[int],
                          **kwargs) -> nn.Module:
    if arch == "ft_transformer":
        return FTTransformerTabularEncoder(n_num, cat_cardinalities, **kwargs)
    if arch == "naive_mlp":
        kwargs.pop("n_heads", None)
        kwargs.pop("semantic_emb", None)
        kwargs.pop("attend_missing", None)
        return NaiveMLPTabularEncoder(n_num, cat_cardinalities, **kwargs)
    raise ValueError(f"arch tabulaire inconnue : {arch}")


# ===========================================================================
# 3. Tête de projection
# ===========================================================================
class ProjectionHead(nn.Module):
    """MLP 2 couches vers l'espace contrastif (SimCLR / CLIP)."""

    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
