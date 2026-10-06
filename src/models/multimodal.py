"""
Assemblage : modèle de pré-entraînement contrastif + modèle de tâche aval.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .encoders import ImageEncoder, ProjectionHead, build_tabular_encoder


# ===========================================================================
# Modèle SSL
# ===========================================================================
class MultimodalContrastiveModel(nn.Module):
    """
        image  --[ImageEncoder]--> h_I --[ProjHead_I]--> z_I  ┐
                                                              ├─ InfoNCE
        tabular--[TabEncoder]----> h_T --[ProjHead_T]--> z_T  ┘

    Les têtes de projection sont jetées après le pré-entraînement ; seuls les
    encodeurs sont transférés (comme dans SimCLR/CLIP).
    """

    def __init__(self, image_cfg: dict, tabular_cfg: dict, proj_cfg: dict,
                 n_num: int, cat_cardinalities: List[int],
                 semantic_emb: Optional[np.ndarray] = None,
                 image_size: int = 128, in_chans: int = 1):
        super().__init__()
        self.image_encoder = ImageEncoder(
            name=image_cfg["name"], pretrained=image_cfg.get("pretrained", False),
            in_chans=in_chans, image_size=image_size)

        tab_kwargs = dict(
            d_token=tabular_cfg.get("d_token", 64),
            n_layers=tabular_cfg.get("n_layers", 3),
            n_heads=tabular_cfg.get("n_heads", 4),
            dropout=tabular_cfg.get("dropout", 0.1),
            semantic_emb=semantic_emb,
            attend_missing=tabular_cfg.get("attend_missing", True),
        )
        self.tabular_encoder = build_tabular_encoder(
            tabular_cfg.get("arch", "ft_transformer"), n_num, cat_cardinalities,
            **tab_kwargs)

        self.proj_image = ProjectionHead(self.image_encoder.out_dim,
                                         proj_cfg["hidden"], proj_cfg["dim"])
        self.proj_tabular = ProjectionHead(self.tabular_encoder.out_dim,
                                           proj_cfg["hidden"], proj_cfg["dim"])

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image)

    def encode_tabular(self, x_num, x_cat, mask) -> torch.Tensor:
        return self.tabular_encoder(x_num, x_cat, mask)

    def forward(self, image, x_num, x_cat, mask) -> Dict[str, torch.Tensor]:
        h_i = self.encode_image(image)
        h_t = self.encode_tabular(x_num, x_cat, mask)
        return {"h_image": h_i, "h_tabular": h_t,
                "z_image": self.proj_image(h_i),
                "z_tabular": self.proj_tabular(h_t)}


# ===========================================================================
# Modèle aval
# ===========================================================================
class DownstreamModel(nn.Module):
    """
    Classifieur pour la tâche aval. `modalities` contrôle ce qui est disponible
    au moment du test :

      * "image"   : protocole de Hager et al. — en clinique, l'image est toujours
                    là, les métadonnées structurées souvent non. On évalue donc
                    ce que le pré-entraînement multimodal a *injecté* dans
                    l'encodeur image.
      * "tabular" : baseline tabulaire seule.
      * "both"    : fusion tardive des deux représentations.
    """

    def __init__(self, image_encoder: Optional[nn.Module],
                 tabular_encoder: Optional[nn.Module],
                 n_classes: int, modalities: str = "image",
                 head_hidden: int = 0, dropout: float = 0.0):
        super().__init__()
        assert modalities in ("image", "tabular", "both")
        self.modalities = modalities
        self.image_encoder = image_encoder if modalities in ("image", "both") else None
        self.tabular_encoder = tabular_encoder if modalities in ("tabular", "both") else None

        dim = 0
        if self.image_encoder is not None:
            dim += self.image_encoder.out_dim
        if self.tabular_encoder is not None:
            dim += self.tabular_encoder.out_dim

        if head_hidden:
            self.head = nn.Sequential(
                nn.Linear(dim, head_hidden), nn.BatchNorm1d(head_hidden),
                nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Linear(head_hidden, n_classes))
        else:
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_classes))

    def features(self, image, x_num, x_cat, mask) -> torch.Tensor:
        parts = []
        if self.image_encoder is not None:
            parts.append(self.image_encoder(image))
        if self.tabular_encoder is not None:
            parts.append(self.tabular_encoder(x_num, x_cat, mask))
        return torch.cat(parts, dim=1)

    def forward(self, image, x_num, x_cat, mask) -> torch.Tensor:
        return self.head(self.features(image, x_num, x_cat, mask))

    def freeze_encoders(self) -> None:
        """Protocole 'linear probe' : on n'entraîne que la tête."""
        for enc in (self.image_encoder, self.tabular_encoder):
            if enc is not None:
                enc.eval()
                for p in enc.parameters():
                    p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        # en linear probe on garde les encodeurs en eval (BatchNorm figée)
        for enc in (self.image_encoder, self.tabular_encoder):
            if enc is not None and not any(p.requires_grad for p in enc.parameters()):
                enc.eval()
        return self


# ===========================================================================
# Fabriques
# ===========================================================================
def build_pretrain_model(cfg, schema, semantic_emb=None) -> MultimodalContrastiveModel:
    return MultimodalContrastiveModel(
        image_cfg=dict(cfg.model.image),
        tabular_cfg=dict(cfg.model.tabular),
        proj_cfg=dict(cfg.model.projection),
        n_num=schema.n_num,
        cat_cardinalities=schema.cat_cardinalities,
        semantic_emb=semantic_emb,
        image_size=cfg.data.image_size,
        in_chans=cfg.data.in_chans,
    )


def build_downstream_model(cfg, schema, n_classes: int, modalities: str,
                           pretrained_ckpt: str | None = None,
                           semantic_emb=None) -> DownstreamModel:
    """Construit le modèle aval et y charge éventuellement les poids SSL."""
    base = build_pretrain_model(cfg, schema, semantic_emb)
    if pretrained_ckpt:
        state = torch.load(pretrained_ckpt, map_location="cpu", weights_only=False)
        sd = state["model"] if "model" in state else state
        missing, unexpected = base.load_state_dict(sd, strict=False)
        # Garde-fou indispensable : avec strict=False, un checkpoint incompatible
        # (d_token différent, autre arch tabulaire, embeddings sémantiques
        # activés d'un côté seulement) se charge SILENCIEUSEMENT en ne
        # transférant rien. On croirait alors évaluer le SSL alors qu'on évalue
        # une initialisation aléatoire.
        enc_missing = [k for k in missing if k.startswith(("image_encoder.",
                                                           "tabular_encoder."))]
        if unexpected or enc_missing:
            raise RuntimeError(
                f"Checkpoint SSL incompatible avec la config actuelle.\n"
                f"  clés inattendues  : {len(unexpected)} (ex. {unexpected[:3]})\n"
                f"  encodeurs manquants : {len(enc_missing)} (ex. {enc_missing[:3]})\n"
                f"Vérifiez model.image.name, model.tabular.arch/d_token/n_layers "
                f"et model.tabular.semantic_field_emb.")
        n_loaded = sum(1 for k in sd if k.startswith(("image_encoder.",
                                                      "tabular_encoder.")))
        print(f"[downstream] {n_loaded} tenseurs d'encodeur chargés depuis "
              f"{pretrained_ckpt}")
    ft = cfg.get("finetune", {})
    return DownstreamModel(base.image_encoder, base.tabular_encoder,
                           n_classes=n_classes, modalities=modalities,
                           head_hidden=int(ft.get("head_hidden", 0)),
                           dropout=float(ft.get("head_dropout", 0.1)))
