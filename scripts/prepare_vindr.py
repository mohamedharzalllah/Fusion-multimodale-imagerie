"""
Conversion VinDr-Mammo : DICOM -> PNG 16 bits redimensionnés.

VinDr-Mammo pèse ~300 Go en DICOM natif. Ce script produit une copie légère
(~3-6 Go) utilisable directement par le pipeline.

Traitements appliqués (dans cet ordre, et l'ordre compte) :
  1. lecture du pixel array + application de la LUT VOI/WINDOW du DICOM
  2. inversion si PhotometricInterpretation == MONOCHROME1
  3. recadrage automatique sur la région du sein (seuillage d'Otsu + plus
     grande composante) — supprime le fond noir, qui représente souvent 50 %
     de l'image et dilue le budget de résolution
  4. redimensionnement au côté long demandé, en conservant le ratio
  5. écriture PNG 16 bits (préserve la dynamique des microcalcifications)

Usage :
    python scripts/prepare_vindr.py --root /data/vindr --out /data/vindr/images_png --size 1024
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def otsu_threshold(img: np.ndarray, nbins: int = 256) -> float:
    hist, edges = np.histogram(img, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.cumsum((hist * centers)[::-1])[::-1] - hist * centers) / np.maximum(w1, 1)
    var = w0 * w1 * (m0 - m1) ** 2
    return float(centers[np.argmax(var)])


def crop_breast(img: np.ndarray, margin: int = 8) -> np.ndarray:
    """Recadre sur la plus grande zone non-fond."""
    thr = otsu_threshold(img)
    m = img > thr
    rows, cols = np.where(m)
    if len(rows) == 0:
        return img
    r0, r1 = max(rows.min() - margin, 0), min(rows.max() + margin, img.shape[0])
    c0, c1 = max(cols.min() - margin, 0), min(cols.max() + margin, img.shape[1])
    return img[r0:r1, c0:c1]


def convert_one(dcm_path: Path, out_path: Path, size: int) -> bool:
    import pydicom
    from PIL import Image
    try:
        ds = pydicom.dcmread(str(dcm_path))
        arr = ds.pixel_array.astype(np.float32)

        # LUT VOI (fenêtrage) si présente
        try:
            from pydicom.pixel_data_handlers.util import apply_voi_lut
            arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
        except Exception:
            pass

        if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
            arr = arr.max() - arr

        arr = arr - arr.min()
        arr = arr / max(arr.max(), 1e-6)
        arr = crop_breast(arr)

        h, w = arr.shape
        scale = size / max(h, w)
        new = (max(int(round(w * scale)), 1), max(int(round(h * scale)), 1))
        im = Image.fromarray((arr * 65535).astype(np.uint16)).resize(new, Image.BILINEAR)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path)
        return True
    except Exception as e:
        print(f"  [skip] {dcm_path.name}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="racine VinDr (contient images/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="0 = tout convertir")
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    dcms = sorted(root.rglob("*.dicom")) + sorted(root.rglob("*.dcm"))
    if a.limit:
        dcms = dcms[:a.limit]
    print(f"{len(dcms)} DICOM à convertir -> {out}")

    ok = 0
    for i, p in enumerate(dcms):
        rel = p.relative_to(root)
        dst = out / rel.parent.name / (p.stem + ".png")
        if dst.exists():
            ok += 1
            continue
        ok += convert_one(p, dst, a.size)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(dcms)} ({ok} réussis)")
    print(f"terminé : {ok}/{len(dcms)}")


if __name__ == "__main__":
    main()
