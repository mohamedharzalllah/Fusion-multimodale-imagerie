"""
À LANCER EN PREMIER. Ne modifie rien, n'écrit rien : il inspecte.

    python scripts/inspect_cbis.py                       # racine auto-détectée
    python scripts/inspect_cbis.py --root /kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset

Le dataset `awsaf49/cbis-ddsm-breast-cancer-image-dataset` a une particularité :
les CSV de description pointent vers des chemins **DICOM** d'origine, alors que
les images sont des JPEG rangés par `SeriesInstanceUID`. Le pont entre les deux
est `dicom_info.csv`. Ce script vérifie que ce pont fonctionne AVANT de lancer
quoi que ce soit de long, et affiche tout ce dont on a besoin pour adapter le
code si la structure diffère.

Aucune profondeur d'arborescence n'est supposée.

Il répond à huit questions :
  1. où sont les CSV et les images, et à quelle profondeur ?
  2. quels sont les vrais noms de colonnes (ils diffèrent entre mass et calc) ?
  3. quelles valeurs prend `SeriesDescription` dans dicom_info.csv ?
  4. la clé de jointure (1er segment du chemin CSV = PatientID DICOM) marche-t-elle ?
  5. quel pourcentage de lignes se résout en un fichier existant ?
  6. combien de dossiers de série contiennent PLUSIEURS images (masque + crop) ?
  7. distribution des cibles, des manquants, et des cardinalités catégorielles
  8. les patients se chevauchent-ils entre train et test officiels ?
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.cbis_paths import (ImageIndex, discover_root,  # noqa: E402
                                 find_dirs)

CASE_FILES = [("mass", "train"), ("mass", "test"), ("calc", "train"), ("calc", "test")]


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="racine du dataset ; auto-détectée si omise")
    ap.add_argument("--sample", type=int, default=500,
                    help="nb de lignes échantillonnées pour tester la résolution")
    a = ap.parse_args()

    # ---------------------------------------------------------------- 1
    hr("1. LOCALISATION ET ARBORESCENCE")
    try:
        root = discover_root(a.root)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    print(f"racine détectée : {root}")
    for p in sorted(root.iterdir())[:20]:
        extra = ""
        if p.is_dir():
            try:
                extra = f"  ({sum(1 for _ in p.iterdir())} entrées)"
            except OSError:
                pass
        print(f"  {'DIR ' if p.is_dir() else 'FILE'} {p.name}{extra}")

    try:
        csv_dir, img_dir = find_dirs(root)
    except FileNotFoundError as e:
        print(f"\nERREUR : {e}")
        sys.exit(1)
    print(f"\n-> dossier CSV    : {csv_dir}")
    print(f"-> dossier images : {img_dir}")

    print("\nIndexation des images (un seul parcours)…")
    index = ImageIndex(img_dir)
    print(f"  {index.summary()}")
    if index.n_files == 0:
        print("ERREUR : aucun fichier image trouvé.")
        sys.exit(1)
    ex_folder, ex_files = next(iter(index.by_folder.items()))
    print(f"  exemple de dossier : {ex_folder}")
    print(f"    -> {[Path(f).name for f in ex_files][:4]}")
    print(f"    chemin relatif : "
          f"{Path(ex_files[0]).relative_to(img_dir)}")

    # ---------------------------------------------------------------- 2
    hr("2. COLONNES DES CSV DE CAS (attention : mass et calc diffèrent)")
    frames = {}
    for kind, split in CASE_FILES:
        p = csv_dir / f"{kind}_case_description_{split}_set.csv"
        if not p.exists():
            hits = list(csv_dir.glob(f"{kind}*_{split}_set.csv"))
            if not hits:
                print(f"  MANQUANT : {p.name}")
                continue
            p = hits[0]
        d = pd.read_csv(p)
        frames[(kind, split)] = d
        print(f"\n{p.name}  ({len(d)} lignes)")
        for c in d.columns:
            print(f"    {c!r}")

    if frames:
        allcols = {k: set(v.columns) for k, v in frames.items()}
        mass = allcols.get(("mass", "train"), set())
        calc = allcols.get(("calc", "train"), set())
        print("\n  -> colonnes SEULEMENT dans mass :", sorted(mass - calc))
        print("  -> colonnes SEULEMENT dans calc :", sorted(calc - mass))
        print("\n  (le piège classique : 'breast_density' dans mass vs "
              "'breast density' dans calc)")

    # ---------------------------------------------------------------- 3
    hr("3. dicom_info.csv")
    dpath = csv_dir / "dicom_info.csv"
    if not dpath.exists():
        hits = list(csv_dir.glob("dicom*info*.csv"))
        dpath = hits[0] if hits else dpath
    if not dpath.exists():
        print("ABSENT — la jointure est impossible sans ce fichier")
        dinfo = None
    else:
        dinfo = pd.read_csv(dpath, low_memory=False)
        print(f"{dpath.name} : {len(dinfo)} lignes, {len(dinfo.columns)} colonnes")
        print("colonnes :")
        for c in dinfo.columns:
            print(f"    {c!r}")
        desc_col = _pick(dinfo, ["SeriesDescription", "series_description"])
        if desc_col:
            print(f"\nvaleurs de {desc_col} :")
            for v, n in dinfo[desc_col].value_counts(dropna=False).items():
                print(f"    {str(v)!r:36s} {n}")
        for col in ("image_path", "PatientID"):
            c = _pick(dinfo, [col])
            if c:
                print(f"\nexemples de {c} :")
                for v in dinfo[c].dropna().head(3):
                    print(f"    {v!r}")

    # ---------------------------------------------------------------- 4-5
    hr("4-5. JOINTURE ET RÉSOLUTION DES CHEMINS")
    if not frames:
        print("pas de CSV de cas -> impossible de tester")
        sys.exit(1)

    # Normaliser AVANT de concaténer, sinon 'breast_density' (mass) et
    # 'breast density' (calc) produisent DEUX colonnes homonymes et tout accès
    # renvoie un DataFrame 2D.
    norm = []
    for (kind, split), d in frames.items():
        d = d.copy()
        d.columns = [c.strip().lower().replace(" ", "_") for c in d.columns]
        d["split"] = split
        d["abnormality_kind"] = kind
        norm.append(d)
    df = pd.concat(norm, ignore_index=True)
    assert not df.columns.duplicated().any(), \
        f"colonnes dupliquées : {df.columns[df.columns.duplicated()].tolist()}"
    print("colonnes normalisées :", sorted(c for c in df.columns
                                           if c not in ("split", "abnormality_kind")))

    pid_col = _pick(dinfo, ["PatientID"]) if dinfo is not None else None
    for pathcol in ("image_file_path", "cropped_image_file_path", "roi_mask_file_path"):
        if pathcol not in df.columns:
            print(f"\n{pathcol} : ABSENTE")
            continue
        vals = df[pathcol].dropna().astype(str)
        print(f"\n{pathcol}  ({len(vals)} valeurs non nulles)")
        for v in vals.head(2):
            print(f"    {v.strip()!r}")
        keys = vals.str.strip().str.split("/").str[0].str.strip()
        print(f"  1er segment (clé de jointure) : {keys.iloc[0]!r}")
        print(f"  {keys.nunique()} clés distinctes")
        if pid_col:
            hit = keys.isin(set(dinfo[pid_col].astype(str)))
            print(f"  -> {hit.mean():.1%} des clés présentes dans "
                  f"dicom_info.{pid_col}")

    # résolution effective
    if dinfo is not None and pid_col:
        desc_col = _pick(dinfo, ["SeriesDescription", "series_description"])
        ip_col = _pick(dinfo, ["image_path", "ImagePath", "file_path"])
        if desc_col and ip_col:
            print(f"\nrésolution réelle sur disque (échantillon de {a.sample}) :")
            for want, label, col in [("full", "full mammogram", "image_file_path"),
                                     ("crop", "cropped (ROI)", "cropped_image_file_path"),
                                     ("mask", "ROI mask", "roi_mask_file_path")]:
                if col not in df.columns:
                    continue
                sub = dinfo[dinfo[desc_col].astype(str).str.lower()
                            .str.contains(want, na=False)]
                mapping = {}
                for pid, ip in zip(sub[pid_col].astype(str), sub[ip_col].astype(str)):
                    files = index.resolve(ip)
                    if files:
                        mapping.setdefault(pid, files)
                keys = (df[col].dropna().astype(str).str.strip()
                        .str.split("/").str[0].str.strip().head(a.sample))
                ok = sum(1 for k in keys if k in mapping)
                tot = max(len(keys), 1)
                print(f"  {label:16s} séries dicom={len(sub):5d}  "
                      f"résolues sur disque={len(mapping):5d}  "
                      f"lignes résolues={ok / tot:6.1%}")
                if ok / tot < 0.5 and len(sub):
                    print(f"      exemple de clé CSV non résolue : "
                          f"{next((k for k in keys if k not in mapping), None)!r}")
                    print(f"      exemple de PatientID dicom      : "
                          f"{next(iter(mapping), None)!r}")

    # ---------------------------------------------------------------- 6
    hr("6. DOSSIERS À PLUSIEURS IMAGES (masque et crop mêlés)")
    counts = Counter(len(v) for v in index.by_folder.values())
    for n in sorted(counts):
        print(f"  {counts[n]:6d} dossiers contiennent {n} image(s)")
    multi = [(k, [Path(f).name for f in v])
             for k, v in index.by_folder.items() if len(v) > 1][:3]
    for name, files in multi:
        print(f"  exemple : {name} -> {files}")
    if sum(c for n, c in counts.items() if n > 1):
        print("\n  Ces dossiers mélangent le patch recadré et le masque ROI.")
        print("  prepare_cbis.py les sépare par un test de binarité de")
        print("  l'histogramme (un masque est quasi noir-et-blanc pur).")

    # ---------------------------------------------------------------- 7
    hr("7. CIBLES, MANQUANTS, CARDINALITÉS")
    if "pathology" in df.columns:
        print("pathology :")
        for v, n in df["pathology"].value_counts(dropna=False).items():
            print(f"    {str(v)!r:28s} {n}")
        mal = (df["pathology"].astype(str).str.upper().str.strip() == "MALIGNANT").mean()
        print(f"  -> prévalence malignité : {mal:.1%}")
    if "assessment" in df.columns:
        print("\nassessment (BI-RADS) :",
              dict(df["assessment"].value_counts().sort_index()))
    if "breast_density" in df.columns:
        print("breast_density :",
              dict(df["breast_density"].value_counts(dropna=False).sort_index()))
        z = int((df["breast_density"] == 0).sum())
        if z:
            print(f"  ATTENTION : {z} lignes ont breast_density=0 (non évalué) "
                  "-> converti en NaN")

    cat_cols = ["left_or_right_breast", "image_view", "abnormality_type",
                "mass_shape", "mass_margins", "calc_type", "calc_distribution"]
    print("\ncolonnes catégorielles :")
    for c in cat_cols:
        if c not in df.columns:
            continue
        s = df[c]
        print(f"  {c:22s} manquants={s.isna().mean():6.1%}  "
              f"cardinalité={s.nunique()}")
        print(f"      top-3 : {dict(s.value_counts().head(3))}")
        if s.nunique() > 15:
            print("      -> forte cardinalité (valeurs composées type "
                  "'PLEOMORPHIC-FINE_LINEAR_BRANCHING') : la queue sera "
                  "regroupée en 'OTHER'")

    # ---------------------------------------------------------------- 8
    hr("8. CHEVAUCHEMENT DE PATIENTS ENTRE TRAIN ET TEST OFFICIELS")
    if "patient_id" in df.columns:
        print(f"patients distincts : {df['patient_id'].nunique()} "
              f"pour {len(df)} lignes "
              f"({len(df) / max(df['patient_id'].nunique(), 1):.1f} lignes/patient)")
        ptr = set(df.loc[df["split"] == "train", "patient_id"])
        pte = set(df.loc[df["split"] == "test", "patient_id"])
        inter = ptr & pte
        print(f"train={len(ptr)} patients | test={len(pte)} patients | "
              f"chevauchement={len(inter)}")
        if inter:
            n_rows = int(df.loc[df["split"] == "train", "patient_id"]
                         .isin(inter).sum())
            print(f"  {len(inter)} patients ({n_rows} lignes de train) sont dans "
                  f"les DEUX splits officiels.")
            print("  Cause : CBIS-DDSM sépare masses et calcifications "
                  "indépendamment.")
            print("  -> `enforce_group_disjoint: true` les retire du train.")
        else:
            print("  Aucun chevauchement : le split officiel est utilisable tel quel.")

    hr("CONCLUSION")
    print("Si les huit sections sont conformes, lancez :")
    print(f"  python scripts/prepare_cbis.py --root {root} \\")
    print("      --out /kaggle/working/cbis_cache --size 256 --kind roi")


def _pick(df, names):
    if df is None:
        return None
    for n in names:
        if n in df.columns:
            return n
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


if __name__ == "__main__":
    main()
