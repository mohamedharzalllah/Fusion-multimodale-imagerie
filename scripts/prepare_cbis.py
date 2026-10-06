"""
Prépare CBIS-DDSM (version Kaggle `awsaf49/cbis-ddsm-breast-cancer-image-dataset`)
en un manifeste + un cache d'images redimensionnées.

    # --root est optionnel : les emplacements Kaggle usuels sont auto-détectés
    python scripts/prepare_cbis.py --out /kaggle/working/cbis_cache --size 256 --kind roi

    # ou explicitement
    python scripts/prepare_cbis.py \
        --root /kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset \
        --out  /kaggle/working/cbis_cache \
        --size 256 --kind roi

Pourquoi une étape de préparation séparée ?
------------------------------------------
1. **La jointure est non triviale.** Les CSV de description pointent vers les
   chemins DICOM d'origine (`Mass-Training_P_00001_LEFT_CC/1.3.6.../000000.dcm`)
   alors que les images sont des JPEG rangés par `SeriesInstanceUID`. Le pont est
   `dicom_info.csv`, où `PatientID` vaut exactement le premier segment du chemin
   CSV. On fait cette jointure UNE fois, pas à chaque epoch.

2. **Les dossiers ambigus.** Certains dossiers de série contiennent DEUX jpeg :
   le patch recadré ET le masque ROI. Les distinguer demande de regarder les
   pixels. On le fait une fois ici (test de binarité de l'histogramme), pas
   10 000 fois par epoch.

3. **La vitesse.** Les mammographies complètes font ~3000x5000 px. Kaggle donne
   2 vCPU : décoder ces JPEG à la volée sature le CPU et le GPU attend. On décode
   une fois, en profitant de `PIL.Image.draft()` qui redimensionne directement
   dans le domaine DCT du JPEG (5 à 10x plus rapide qu'un decode complet suivi
   d'un resize).

Aucune profondeur d'arborescence n'est supposée : la localisation des dossiers et
la résolution des images passent par `src/data/cbis_paths.py`.

Sortie : `<out>/cbis_manifest.csv` + `<out>/images/<id>.png`
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.cbis_paths import (ImageIndex, discover_root,  # noqa: E402
                                 find_dirs)

Image.MAX_IMAGE_PIXELS = None          # les mammographies dépassent la limite par défaut

CASE_FILES = [("mass", "train"), ("mass", "test"), ("calc", "train"), ("calc", "test")]

# quel `SeriesDescription` chercher, et quelle colonne de chemin utiliser
KIND_SPEC = {
    "full": ("full", "image_file_path"),
    "roi":  ("crop", "cropped_image_file_path"),
    "mask": ("mask", "roi_mask_file_path"),
}


# ---------------------------------------------------------------------------
# Lecture des CSV de cas
# ---------------------------------------------------------------------------
def load_case_csvs(csv_dir: Path) -> pd.DataFrame:
    frames = []
    for kind, split in CASE_FILES:
        p = csv_dir / f"{kind}_case_description_{split}_set.csv"
        if not p.exists():
            hits = list(csv_dir.glob(f"{kind}*_{split}_set.csv"))
            if not hits:
                print(f"  [avert.] {p.name} absent")
                continue
            p = hits[0]
        d = pd.read_csv(p)
        # Normalisation AVANT concaténation — c'est essentiel : 'breast density'
        # (fichiers calc) et 'breast_density' (fichiers mass) doivent devenir la
        # même colonne. Si l'on concatène d'abord, on obtient deux colonnes
        # homonymes et tout accès `df["breast_density"]` renvoie un DataFrame 2D.
        d.columns = [c.strip().lower().replace(" ", "_") for c in d.columns]
        if d.columns.duplicated().any():
            dup = d.columns[d.columns.duplicated()].tolist()
            raise ValueError(f"{p.name} : colonnes homonymes après "
                             f"normalisation : {dup}")
        d["abnormality_kind"] = kind
        d["split"] = split
        frames.append(d)
        print(f"  {p.name:44s} {len(d):5d} lignes")
    if not frames:
        raise FileNotFoundError(f"aucun CSV de cas dans {csv_dir}")
    df = pd.concat(frames, ignore_index=True)

    for c in ("mass_shape", "mass_margins", "calc_type", "calc_distribution",
              "breast_density", "subtlety", "abnormality_id", "assessment",
              "pathology", "patient_id", "left_or_right_breast", "image_view"):
        if c not in df.columns:
            df[c] = np.nan
    df["abnormality_type"] = df["abnormality_kind"].str.upper()
    return df


# ---------------------------------------------------------------------------
# Jointure via dicom_info.csv
# ---------------------------------------------------------------------------
def build_path_map(csv_dir: Path, index: ImageIndex, want: str) -> Dict[str, List[str]]:
    """{PatientID -> fichiers image de la série}, filtré sur SeriesDescription."""
    dpath = csv_dir / "dicom_info.csv"
    if not dpath.exists():
        hits = list(csv_dir.glob("dicom*info*.csv"))
        if not hits:
            raise FileNotFoundError(
                f"dicom_info.csv absent de {csv_dir} : c'est le pont "
                "indispensable entre les CSV de cas (chemins DICOM) et les JPEG "
                "(rangés par SeriesInstanceUID).")
        dpath = hits[0]
    dinfo = pd.read_csv(dpath, low_memory=False)

    pid_col = _pick(dinfo, ["PatientID", "patient_id"])
    desc_col = _pick(dinfo, ["SeriesDescription", "series_description"])
    ip_col = _pick(dinfo, ["image_path", "image path", "ImagePath", "file_path"])
    missing = [n for n, c in (("PatientID", pid_col), ("SeriesDescription", desc_col),
                              ("image_path", ip_col)) if c is None]
    if missing:
        raise KeyError(f"colonnes absentes de {dpath.name} : {missing}\n"
                       f"colonnes présentes : {list(dinfo.columns)}")

    sel = dinfo[dinfo[desc_col].astype(str).str.lower().str.contains(want, na=False)]
    print(f"  dicom_info : {len(sel)} séries dont SeriesDescription contient "
          f"{want!r} (sur {len(dinfo)})")
    if sel.empty:
        vals = dinfo[desc_col].value_counts().head(10)
        raise ValueError(
            f"aucune série ne correspond à {want!r}. Valeurs de {desc_col} :\n"
            + "\n".join(f"  {v!r}: {n}" for v, n in vals.items()))

    out: Dict[str, List[str]] = {}
    n_unresolved = 0
    for pid, ip in zip(sel[pid_col].astype(str), sel[ip_col].astype(str)):
        files = index.resolve(ip)
        if files:
            out.setdefault(pid, []).extend(files)
        else:
            n_unresolved += 1
    if n_unresolved:
        print(f"  [avert.] {n_unresolved} séries de dicom_info non retrouvées "
              f"sur le disque")
    for pid in out:
        out[pid] = sorted(set(out[pid]))
    return out


def _pick(df: pd.DataFrame, names) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


# ---------------------------------------------------------------------------
# Désambiguïsation crop / masque
# ---------------------------------------------------------------------------
def binariness(path: str, probe: int = 96) -> float:
    """
    Fraction de pixels quasi-noirs ou quasi-blancs, sur une vignette.

    Un **masque ROI** est binaire : cette fraction vaut ~1.0. Un **patch de
    tissu** a un histogramme étalé, la fraction est bien plus basse. C'est ce qui
    permet de séparer les deux jpeg d'un même dossier de série, là où
    `SeriesDescription` ne suffit pas.
    """
    try:
        im = Image.open(path)
        im.draft("L", (probe, probe))          # décodage partiel : très rapide
        a = np.asarray(im.convert("L").resize((probe, probe), Image.BILINEAR))
    except Exception:
        return 0.0
    return float(((a <= 12) | (a >= 243)).mean())


def choose_file(paths: List[str], want: str) -> Optional[str]:
    """Choisit le bon jpeg parmi ceux d'un dossier de série."""
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]
    scored = sorted(((binariness(p), p) for p in paths), key=lambda t: t[0])
    if want == "mask":
        return scored[-1][1]          # le plus binaire = le masque
    return scored[0][1]               # le moins binaire = le patch de tissu


# ---------------------------------------------------------------------------
# Cache d'images
# ---------------------------------------------------------------------------
def cache_image(src: str, dst: Path, size: int) -> bool:
    if dst.exists():
        return True
    try:
        im = Image.open(src)
        # draft : PIL décode directement à une échelle réduite (JPEG uniquement).
        im.draft("L", (size * 2, size * 2))
        im = im.convert("L")
        w, h = im.size
        scale = size / max(w, h)
        if scale < 1.0:
            im = im.resize((max(int(w * scale), 1), max(int(h * scale), 1)),
                           Image.BILINEAR)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, format="PNG", optimize=False, compress_level=1)
        return True
    except Exception as e:
        print(f"  [skip] {Path(src).name}: {e}")
        return False


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="racine du dataset ; auto-détectée si omise")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256,
                    help="côté long après redimensionnement")
    ap.add_argument("--kind", default="roi", choices=["roi", "full", "mask"],
                    help="roi = patch recadré (rapide) ; full = mammographie complète")
    ap.add_argument("--limit", type=int, default=0, help="0 = tout")
    ap.add_argument("--no-cache-images", action="store_true",
                    help="ne construit que le manifeste, pointant vers les JPEG d'origine")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"PRÉPARATION CBIS-DDSM  kind={a.kind}  size={a.size}")
    print("=" * 78)

    root = discover_root(a.root)
    csv_dir, img_dir = find_dirs(root)
    print(f"racine : {root}")
    print(f"csv    : {csv_dir}")
    print(f"images : {img_dir}")

    print("\nIndexation des images (un seul parcours, profondeur quelconque)…")
    index = ImageIndex(img_dir)
    print(f"  {index.summary()}")
    if index.n_files == 0:
        print("ERREUR : aucune image trouvée. Lancez scripts/inspect_cbis.py.")
        sys.exit(1)

    print("\nLecture des CSV de cas :")
    df = load_case_csvs(csv_dir)
    print(f"  total : {len(df)} lignes, {df['patient_id'].nunique()} patients\n")

    want, pathcol = KIND_SPEC[a.kind]
    if pathcol not in df.columns:
        print(f"ERREUR : colonne {pathcol} absente. Colonnes : {sorted(df.columns)}")
        sys.exit(1)

    print(f"Jointure via dicom_info.csv (SeriesDescription ~ {want!r}) :")
    pmap = build_path_map(csv_dir, index, want)
    print(f"  {len(pmap)} PatientID résolus\n")

    # clé = premier segment du chemin CSV (= PatientID de dicom_info)
    df["_key"] = (df[pathcol].astype(str).str.strip()
                  .str.replace("\\", "/", regex=False).str.split("/").str[0]
                  .str.strip())
    hit = df["_key"].isin(pmap)
    print(f"Résolution : {hit.mean():.1%} des lignes ont une série correspondante")
    if hit.mean() < 0.5:
        print("\n  Taux anormalement bas. Exemples de clés non résolues :")
        for k in df.loc[~hit, "_key"].head(5):
            print(f"    {k!r}")
        print("  Exemples de PatientID disponibles :")
        for k in list(pmap)[:5]:
            print(f"    {k!r}")
        print("\n  -> relancez scripts/inspect_cbis.py et envoyez la sortie.")
    df = df[hit].reset_index(drop=True)
    if df.empty:
        print("ERREUR : aucune ligne résolue, arrêt.")
        sys.exit(1)
    if a.limit:
        df = df.head(a.limit)

    # ---- fichier retenu par ligne, avec désambiguïsation ----
    print(f"\nSélection du bon jpeg par série ({len(df)} lignes)…")
    n_ambig = 0
    src_paths = []
    for i, k in enumerate(df["_key"]):
        paths = pmap[k]
        if len(paths) > 1:
            n_ambig += 1
        src_paths.append(choose_file(paths, want))
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(df)}")
    df["src_path"] = src_paths
    print(f"  {n_ambig} lignes issues d'un dossier multi-images "
          f"(désambiguïsé par test de binarité)")
    df = df[df["src_path"].notna()].reset_index(drop=True)

    # ---- cache d'images ----
    if a.no_cache_images:
        df["image_path"] = df["src_path"]
        print("\n(cache d'images désactivé : le manifeste pointe vers les JPEG d'origine)")
    else:
        print(f"\nConstruction du cache {a.size} px dans {out / 'images'} …")
        img_out = out / "images"
        kept = []
        for i, r in df.iterrows():
            uid = f"{r['_key']}__{Path(r['src_path']).stem}"
            dst = img_out / f"{uid}.png"
            kept.append(str(dst) if cache_image(r["src_path"], dst, a.size) else None)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(df)}")
        df["image_path"] = kept
        df = df[df["image_path"].notna()].reset_index(drop=True)

    # ---- nettoyage des métadonnées ----
    # Dans CBIS-DDSM, 0 code « non évalué » et non une valeur basse, pour la
    # densité comme pour la subtilité (échelles 1-4 et 1-5). Le laisser à 0
    # fausse la standardisation et fait croire au modèle à une valeur observée.
    df["breast_density"] = pd.to_numeric(df["breast_density"], errors="coerce")
    df.loc[df["breast_density"] == 0, "breast_density"] = np.nan
    df["subtlety"] = pd.to_numeric(df["subtlety"], errors="coerce")
    df.loc[df["subtlety"] == 0, "subtlety"] = np.nan
    df["abnormality_id"] = pd.to_numeric(df["abnormality_id"], errors="coerce")
    df["assessment"] = pd.to_numeric(df["assessment"], errors="coerce")

    for c in ("mass_shape", "mass_margins", "calc_type", "calc_distribution",
              "left_or_right_breast", "image_view"):
        df[c] = (df[c].astype(str).str.strip().str.upper()
                 .replace({"N/A": np.nan, "NAN": np.nan, "": np.nan, "NONE": np.nan}))

    df["n_abnormalities"] = df.groupby("_key")["abnormality_id"].transform("count")

    df["pathology"] = df["pathology"].astype(str).str.upper().str.strip()
    df["malignancy"] = (df["pathology"] == "MALIGNANT").astype(int)
    df["birads"] = df["assessment"].fillna(0).astype(int)

    # ---- écriture ----
    manifest = out / "cbis_manifest.csv"
    keep_cols = ["patient_id", "split", "abnormality_type", "abnormality_id",
                 "n_abnormalities", "breast_density", "subtlety",
                 "left_or_right_breast", "image_view",
                 "mass_shape", "mass_margins", "calc_type", "calc_distribution",
                 "assessment", "pathology", "malignancy", "birads",
                 "image_path", "src_path", "_key"]
    df[[c for c in keep_cols if c in df.columns]].to_csv(manifest, index=False)

    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"]
    print("\n" + "=" * 78)
    print(f"MANIFESTE : {manifest}   ({len(df)} lignes)")
    print("=" * 78)
    print(f"patients      : {df['patient_id'].nunique()}")
    print(f"split officiel: train={len(tr)} test={len(te)}")
    print(f"malignité     : {df['malignancy'].mean():.1%} de positifs "
          f"(train {tr['malignancy'].mean():.1%} | test {te['malignancy'].mean():.1%})")
    print(f"pathology     : {dict(df['pathology'].value_counts())}")
    print("\nvaleurs manquantes par colonne tabulaire :")
    for c in ["breast_density", "subtlety", "mass_shape", "mass_margins",
              "calc_type", "calc_distribution"]:
        if c in df:
            print(f"  {c:22s} {df[c].isna().mean():6.1%}  "
                  f"(cardinalité {df[c].nunique()})")
    print("\nLa manquance de mass_* / calc_* est STRUCTURELLE : c'est exactement")
    print("le phénomène que l'encodeur missing-aware doit exploiter.")
    print("\nÉtape suivante :")
    print("  python -m src.train.run_experiments --config configs/cbis_kaggle.yaml \\")
    print(f"      --set data.root={out}")


if __name__ == "__main__":
    main()
