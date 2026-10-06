"""Utilitaires transverses : seed, device, config, logging, checkpoints."""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Fixe toutes les sources d'aléa (python, numpy, torch, cuda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


# ---------------------------------------------------------------------------
# Config : dict imbriqué accessible par attribut (cfg.model.image.name)
# ---------------------------------------------------------------------------
class Cfg(dict):
    """Dictionnaire récursif accessible par point."""

    def __getattr__(self, item):
        try:
            v = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        return Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, key, value):
        self[key] = value


def load_config(path: str | Path, overrides: Dict[str, Any] | None = None) -> Cfg:
    """Charge un YAML et applique des overrides `a.b.c=valeur`."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for k, v in overrides.items():
            node = cfg
            parts = k.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = v
    return Cfg(cfg)


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(state: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


# ---------------------------------------------------------------------------
# Suivi de métriques
# ---------------------------------------------------------------------------
@dataclass
class AverageMeter:
    """Moyenne glissante d'une quantité scalaire."""

    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)

    def reset(self) -> None:
        self.total, self.count = 0.0, 0


class JsonLogger:
    """Journalise les métriques ligne par ligne dans un .jsonl."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, **kwargs) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(kwargs, default=float) + "\n")


# ---------------------------------------------------------------------------
# Scheduler : warmup linéaire + décroissance cosinus
# ---------------------------------------------------------------------------
def cosine_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    import math

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
