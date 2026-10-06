#!/usr/bin/env bash
# Reproduit l'intégralité des résultats du projet.
#
#   bash scripts/run_all.sh            # dataset synthétique (aucun téléchargement)
#   bash scripts/run_all.sh cbis /data/cbis
#   bash scripts/run_all.sh vindr /data/vindr
set -euo pipefail
cd "$(dirname "$0")/.."

SOURCE="${1:-synthetic}"
ROOT="${2:-}"

case "$SOURCE" in
  synthetic) EXTRA="data.source=synthetic" ;;
  cbis)      EXTRA="data.source=cbis_ddsm data.root=$ROOT data.image_size=224 model.image.name=resnet50" ;;
  vindr)     EXTRA="data.source=vindr data.root=$ROOT data.image_size=224 model.image.name=resnet50" ;;
  *) echo "source inconnue : $SOURCE (attendu : synthetic | cbis | vindr)"; exit 1 ;;
esac

echo "==> 1/3  Smoke test"
python scripts/smoke_test.py

echo "==> 2/3  Plan d'expériences complet (source=$SOURCE)"
python -m src.train.run_experiments \
  --config configs/default.yaml \
  --ablations \
  --set $EXTRA

echo "==> 3/3  Figures et tableau"
python -m src.eval.report --out-dir outputs

echo
echo "Terminé. Résultats :"
echo "  outputs/results.csv"
echo "  outputs/results.md"
echo "  outputs/robustness_missing.csv"
echo "  outputs/figures/label_efficiency.png"
echo "  outputs/figures/pretrain_curves.png"
