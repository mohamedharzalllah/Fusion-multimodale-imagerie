"""
Écrit tout le projet EN CLAIR dans une cellule de `notebooks/kaggle_cbis.ipynb`.

    python scripts/embed_sources.py

Le notebook devient alors totalement autonome : ni dépôt GitHub, ni Kaggle
Dataset de code, ni upload. On l'importe sur Kaggle, on ajoute le dataset
d'images, et on lance.

Pourquoi en clair plutôt qu'en base64
-------------------------------------
Une charge base64 fonctionne mais produit 140 Ko de chaîne opaque : illisible en
revue, impossible à corriger sur place, et invisible à un `git diff`. Ici chaque
fichier apparaît tel quel dans le notebook, relisible et modifiable directement
dans la cellule.

Le prix : un `FILES` d'environ 220 Ko de source. C'est une seule cellule, que
Kaggle et Jupyter replient.

Garde-fous
----------
Les sources sont insérées dans des littéraux `r'''...'''`. Deux conditions
doivent tenir, et sont VÉRIFIÉES avant écriture — un échec ici produirait un
notebook syntaxiquement cassé, ou pire, silencieusement tronqué :

  * aucun fichier ne contient la séquence ``'''`` (elle fermerait le littéral) ;
  * aucun fichier ne se termine par ``\\`` (interdit en fin de raw string).

Après génération, le script relit la cellule, ré-exécute l'écriture dans un
dossier temporaire et compare les fichiers aux originaux octet par octet.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "kaggle_cbis.ipynb"

# Uniquement ce qui est nécessaire à l'exécution sur Kaggle.
# Volontairement exclus : README.md (30 Ko de markdown), requirements.txt
# (les dépendances sont déjà là), prepare_vindr.py et run_all.sh (hors sujet ici).
def essential_files() -> list[Path]:
    files = sorted(p for p in (ROOT / "src").rglob("*.py")
                   if "__pycache__" not in p.parts)
    files += [ROOT / "configs" / "default.yaml", ROOT / "configs" / "cbis_kaggle.yaml"]
    files += [ROOT / "scripts" / n for n in
              ("inspect_cbis.py", "prepare_cbis.py", "smoke_test.py")]
    return [f for f in files if f.exists()]


def check(files: list[Path]) -> None:
    bad = []
    for p in files:
        t = p.read_text(encoding="utf-8")
        if "'''" in t:
            bad.append(f"{p.relative_to(ROOT)} : contient ''' (fermerait le littéral)")
        if t.rstrip("\n").endswith("\\"):
            bad.append(f"{p.relative_to(ROOT)} : finit par \\ (interdit en raw string)")
    if bad:
        raise SystemExit("Sources incompatibles avec l'insertion en clair :\n  "
                         + "\n  ".join(bad))


HEADER = '''# ============================================================================
# LE PROJET ENTIER EST DANS CETTE CELLULE, EN CLAIR.
#
# Aucun dépôt GitHub, aucun Kaggle Dataset de code, aucun upload : cette cellule
# écrit src/, configs/ et scripts/ dans /kaggle/working/fusion-multimodale.
#
# Le seul dataset à attacher est celui des IMAGES :
#     awsaf49/cbis-ddsm-breast-cancer-image-dataset
#
# Pour modifier le code : éditez directement les chaînes ci-dessous, ré-exécutez
# cette cellule, et les fichiers sont réécrits. En local, régénérez la cellule
# avec `python scripts/embed_sources.py`.
# ============================================================================
import importlib, os, sys, textwrap
from pathlib import Path

FILES = {}
'''

FOOTER = '''
# --- écriture sur disque -----------------------------------------------------
REPO.mkdir(parents=True, exist_ok=True)
for _rel, _txt in FILES.items():
    _p = REPO / _rel
    _p.parent.mkdir(parents=True, exist_ok=True)
    _p.write_text(_txt, encoding="utf-8")

os.chdir(REPO)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# --- purge du cache d'import -------------------------------------------------
# Réécrire les fichiers ne recharge RIEN : Python garde les modules déjà importés
# dans sys.modules, et `from src.x import y` renverrait encore l'ancienne
# version. Piège classique de Jupyter, et pire que bruyant : un décalage discret
# entre le code sur disque et celui en mémoire ferait tirer des conclusions sur
# du vieux code, sans la moindre erreur.
_stale = [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]
for _m in _stale:
    del sys.modules[_m]
importlib.invalidate_caches()

_missing = [r for r in FILES if not (REPO / r).exists()]
assert not _missing, f"fichiers non écrits : {_missing}"
print(f"projet écrit dans {REPO}")
print(f"  {len(FILES)} fichiers, "
      f"{sum(len(t) for t in FILES.values()) / 1024:.0f} Ko de source")
print(" ", sorted(p.name for p in REPO.iterdir()))
if _stale:
    print(f"\\n  {len(_stale)} modules src.* purgés du cache d'import.")
    print("  ATTENTION : les objets DÉJÀ construits (bundle, model, cfg…) portent")
    print("  encore les anciennes classes. Reconstruisez-les, ou faites Run All —")
    print("  c'est la seule garantie de cohérence.")

# --- le dataset d'IMAGES (le seul à attacher) --------------------------------
from src.data.cbis_paths import discover_root, find_dirs, ImageIndex
ROOT_DS = discover_root(DATASET)
CSV_DIR, IMG_DIR = find_dirs(ROOT_DS)
print(f"\\nIMAGES : {ROOT_DS}")
print(f"         csv    : {CSV_DIR}")
print(f"         images : {IMG_DIR}")
print("        ", ImageIndex(IMG_DIR).summary().replace("\\n", "\\n         "))
'''


def build_cell(files: list[Path]) -> str:
    parts = [HEADER]
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        txt = p.read_text(encoding="utf-8")
        parts.append(f"\n# {'-' * 74}\n# {rel}\n# {'-' * 74}\n"
                     f"FILES[{rel!r}] = r'''{txt}'''\n")
    parts.append(FOOTER)
    return "".join(parts)


def verify(cell: str, files: list[Path]) -> None:
    """Ré-exécute l'écriture dans un dossier jetable et compare aux originaux."""
    body = cell.split("# --- écriture sur disque")[0]
    ns: dict = {}
    exec(compile(body, "<cell>", "exec"), ns)
    written = ns["FILES"]
    assert len(written) == len(files), (
        f"{len(written)} entrées dans FILES pour {len(files)} fichiers")
    tmp = Path(tempfile.mkdtemp())
    try:
        diffs = []
        for rel, txt in written.items():
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp / rel).write_text(txt, encoding="utf-8")
            if (tmp / rel).read_bytes() != (ROOT / rel).read_bytes():
                diffs.append(rel)
        if diffs:
            raise SystemExit(f"round-trip KO sur : {diffs}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    files = essential_files()
    check(files)
    cell = build_cell(files)
    compile(cell, "<cell>", "exec")          # la cellule doit être valide
    verify(cell, files)

    nb = json.loads(NB.read_text(encoding="utf-8"))
    targets = [i for i, c in enumerate(nb["cells"])
               if any(k in "".join(c["source"])
                      for k in ("_find_code", "_find_source", "FILES[", "PROJECT_ZIP_B64"))]
    if len(targets) != 1:
        raise SystemExit(f"attendu 1 cellule d'amorçage, trouvé {len(targets)}")
    i = targets[0]
    nb["cells"][i]["source"] = cell.splitlines(keepends=True)
    nb["cells"][i]["outputs"] = []
    nb["cells"][i]["execution_count"] = None
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    total = sum(len(p.read_text(encoding="utf-8")) for p in files)
    print(f"{len(files)} fichiers, {total / 1024:.0f} Ko de source -> 1 cellule")
    print("round-trip octet par octet : OK")
    print(f"notebook : {NB} ({NB.stat().st_size / 1024:.0f} Ko, "
          f"{len(nb['cells'])} cellules)")


if __name__ == "__main__":
    main()
