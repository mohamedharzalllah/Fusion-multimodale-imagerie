# Fusion multimodale imagerie + données tabulaires en apprentissage auto-supervisé

Implémentation pédagogique et exécutable d'une version simplifiée de **Hager et al., CVPR 2023,
*Best of Both Worlds: Multimodal Contrastive Learning with Tabular and Imaging Data***,
appliquée à la mammographie.

L'idée en une phrase : **apprendre un espace latent partagé entre l'image et le dossier clinique
du même patient, sans aucune étiquette de diagnostic**, puis transférer l'encodeur image ainsi
enrichi vers une tâche supervisée (malignité, BI-RADS) où les annotations sont rares.

---

## Sommaire

1. [Pourquoi ce projet](#1-pourquoi-ce-projet)
2. [Démarrage rapide](#2-démarrage-rapide)
3. [Les 7 étapes, expliquées](#3-les-7-étapes-expliquées)
4. [Choix du dataset](#4-choix-du-dataset)
5. [Structure du code](#5-structure-du-code)
6. [Résultats attendus et lecture](#6-résultats-attendus-et-lecture)
7. [Passer aux vraies données](#7-passer-aux-vraies-données)
8. [Pièges rencontrés](#8-pièges-rencontrés-et-comment-ils-sont-traités)
9. [Extensions](#9-extensions-pour-aller-vers-la-thèse)
10. [Références](#10-références)

---

## 1. Pourquoi ce projet

Le sujet de thèse insiste sur quatre points. Voici où chacun est traité dans le code :

| Mot-clé du sujet | Où c'est traité |
|---|---|
| **SSL multimodal** | `src/losses/contrastive.py` — InfoNCE bidirectionnel image ↔ tabulaire |
| **Sémantique des données tabulaires** | `src/data/semantic.py` + `FeatureTokenizer` — les noms/descriptions de colonnes sont encodés par un modèle de langage et injectés dans les tokens |
| **Valeurs manquantes** | `src/data/tabular.py` + embedding `[MISSING]` appris **par colonne** ; ablation de robustesse dédiée |
| **Au-delà de la concaténation naïve** | `FTTransformerTabularEncoder` (attention entre colonnes) vs baseline `NaiveMLPTabularEncoder`, comparés explicitement |

Le point non négociable du projet : **une AUROC seule ne prouve rien**. Tout est construit autour
d'une comparaison contrôlée entre 7 méthodes qui partagent exactement le même split, le même
prétraitement et le même budget d'entraînement.

---

## 2. Démarrage rapide

**Sur le vrai dataset CBIS-DDSM (Kaggle)** — c'est probablement ce que vous voulez.

Le notebook `notebooks/kaggle_cbis.ipynb` **contient tout le projet** : sa
deuxième cellule écrit les 29 fichiers de `src/`, `configs/` et `scripts/` dans
`/kaggle/working`. Ni dépôt GitHub, ni Dataset de code, ni upload.

1. Kaggle → **File → Import Notebook** → déposez `notebooks/kaggle_cbis.ipynb`
2. **Add Data** → `cbis-ddsm-breast-cancer-image-dataset` (le seul dataset requis)
3. **Settings** → Accelerator **GPU**, Internet **On**
4. **Run All**

Les sources sont écrites **en clair**, pas encodées : chaque fichier est
relisible et modifiable directement dans la cellule. Après une modification en
local, régénérez-la :

```bash
python scripts/embed_sources.py
```

Le script vérifie que chaque fichier ressort **identique octet pour octet**
avant d'écrire le notebook.

> **Le seul point de vigilance** : le notebook embarque un instantané du code.
> Si vous modifiez `src/` sans relancer `embed_sources.py`, le notebook fait
> tourner l'ancienne version sans rien signaler.

Sinon, les trois commandes de la [section 7](#7-passer-aux-vraies-données)
fonctionnent en local comme sur Kaggle.

**Sur données synthétiques**, sans rien télécharger :

```bash
pip install -r requirements.txt

# 1) Vérification que tout fonctionne (~2-4 min sur CPU)
python scripts/smoke_test.py

# 2) Expérience complète sur données synthétiques (~30-60 min sur GPU)
python -m src.train.run_experiments --config configs/default.yaml

# 3) Figures et tableau markdown
python -m src.eval.report --out-dir outputs
```

Sur Colab, ouvrir `notebooks/demo_end_to_end.ipynb` : il fait tout de bout en bout et affiche
les figures en ligne.

Le pipeline tourne **sans télécharger aucune donnée** : un générateur produit des
pseudo-mammographies dont l'apparence et les métadonnées cliniques dérivent des mêmes variables
latentes. Ce n'est pas un jouet vide — le benchmark a été calibré pour avoir la bonne difficulté :

| Prédicteur | AUROC de la malignité |
|---|---|
| Oracle sur les descripteurs visibles dans l'image | **0.92** |
| Métadonnées cliniques seules (régression logistique) | **0.69** |
| Pixels bruts moyennés en 8×8 (log-reg) | 0.51 |

Il y a donc du signal exploitable dans l'image (qu'il faut aller chercher), du signal
complémentaire dans le tableau, et pas de raccourci trivial.

---

## 3. Les 7 étapes, expliquées

### Étape 1 — Construire les paires (image, dossier clinique)

`src/data/synthetic.py`, `src/data/real_datasets.py`, `src/data/datamodule.py`

L'unité d'apprentissage n'est pas une image, c'est une **paire**. Le signal auto-supervisé vient
uniquement du fait que *cette image appartient à ce patient*. Aucun label de malignité n'est
utilisé avant l'étape 5.

Deux précautions qui décident de la validité de tout le reste :

- **Anti-fuite.** Les colonnes `pathology`, `assessment`, `breast_birads` sont retirées du vecteur
  tabulaire. Si on les laisse, le modèle tabulaire atteint 100 % d'AUC et la comparaison n'a plus
  de sens. Voir `banned` dans `build_datasets`.
- **Split par patient.** Une patiente a 2 à 4 vues (CC/MLO × gauche/droite). Les répartir entre
  train et test met la *même* patiente des deux côtés et gonfle artificiellement l'AUC.
  `stratified_split(..., groups=patient_id)` l'empêche.

### Étape 2 — Prétraiter le tabulaire sans mentir sur les manquants

`src/data/tabular.py`

L'approche habituelle impute par la moyenne. Problème : le modèle ne peut plus distinguer
« densité mammaire = valeur moyenne » de « densité mammaire non renseignée ». Or dans les données
réelles la manquance est **informative** — sur CBIS-DDSM, `mass_shape` n'est vide que pour les cas
de calcification.

On produit donc trois tenseurs :

```
x_num  (N, n_num)   valeurs standardisées, 0.0 aux positions manquantes
x_cat  (N, n_cat)   indices ; la dernière catégorie est réservée à MISSING
mask   (N, n_feat)  booléen, True = manquant
```

Les statistiques (moyenne, écart-type, vocabulaire catégoriel) sont ajustées **sur le train
seulement**. Une catégorie jamais vue au train est traitée comme manquante au test.

### Étape 3 — Deux encodeurs

`src/models/encoders.py`

**Image** : n'importe quel backbone `timm` (`resnet18` par défaut, `resnet50` ou `vit_small`
si vous avez le GPU), avec `in_chans=1` et sans tête de classification.

**Tabulaire** : un *FT-Transformer* missing-aware. Chaque colonne devient un token de dimension `d` :

```
numérique   : t_j = x_j · W_j + b_j
catégoriel  : t_j = Embedding_j[k]
MANQUANT    : t_j = m_j            <- embedding appris, PROPRE à la colonne j
puis        : t_j += field_emb_j + semantic_emb_j
```

Un token `[CLS]` est préfixé, un Transformer traite la séquence, et la sortie `[CLS]` est la
représentation du dossier.

Pourquoi pas un MLP sur le vecteur concaténé ? Trois raisons concrètes :

1. l'attention modélise explicitement les **interactions entre colonnes** (« densité D » + « marges
   spiculées » pèse plus que la somme des deux) ;
2. une colonne manquante est **remplacée par un vecteur appris et identifiable**, pas par une
   moyenne trompeuse ;
3. le modèle est **agnostique au nombre de colonnes** — condition nécessaire pour transférer un
   modèle de CBIS-DDSM vers VinDr-Mammo, dont le schéma diffère.

Le `semantic_emb_j` est l'embedding textuel de la *description* de la colonne
(« age of the patient in years at the time of the mammography exam »). C'est ce qui donne un sens
au token au-delà de sa position, et ce qui rend le transfert inter-schémas envisageable.

### Étape 4 — Pré-entraînement contrastif

`src/train/pretrain.py`, `src/losses/contrastive.py`

Pour un batch de `B` paires, avec `z_I` et `z_T` normalisés L2 :

```
logits   = (z_I @ z_T.T) / τ
L_{I→T}  = CE(logits,   diag)     retrouver le bon dossier pour chaque image
L_{T→I}  = CE(logits.T, diag)     retrouver la bonne image pour chaque dossier
L        = λ · L_{I→T} + (1-λ) · L_{T→I}
```

Trois détails qui font la différence entre « ça converge » et « ça n'apprend rien » :

**Augmentation tabulaire.** Sans elle, la perte s'effondre vers une solution triviale. On utilise
la *corruption* de SCARF : pour ~30 % des colonnes tirées au hasard, on remplace la valeur par
celle d'une autre ligne du train. On échantillonne donc dans la marginale empirique de chaque
colonne — la vue reste plausible colonne par colonne, mais les corrélations inter-colonnes sont
cassées. Point subtil : **si la valeur donneuse était manquante, la vue reste manquante**, sinon
on réintroduit exactement l'imputation trompeuse qu'on cherche à éviter.

**Augmentation image adaptée.** SimCLR standard est trop agressif ici : une lésion occupe quelques
dizaines de pixels, un crop à `scale=0.08` la fait disparaître et crée une paire positive fausse.
On garde `scale ≥ 0.6`, pas de flip vertical (l'orientation CC/MLO est informative), pas de jitter
de teinte (image en niveaux de gris).

**Faux négatifs.** Deux patients distincts peuvent avoir des vecteurs cliniques *identiques* après
discrétisation. La perte les pousse à s'éloigner alors qu'ils sont sémantiquement identiques.
`build_duplicate_mask` retire ces paires du dénominateur.

**Comment savoir si ça marche ?** On mesure à chaque époque le **retrieval croisé** sur la
validation : pour chaque image, le bon dossier est-il dans le top-K des `N` dossiers ?
Le hasard vaut `K/N`. Si R@10 ne décolle pas nettement au-dessus du hasard, inutile de passer à
l'étape suivante — l'alignement n'a pas eu lieu.

### Étape 5 — Fine-tuning sur la tâche aval

`src/train/finetune.py`

Les têtes de projection sont jetées, on ne garde que les encodeurs. Deux protocoles :

- `linear_probe` : encodeur gelé, une couche linéaire entraînée → mesure la qualité *intrinsèque*
  des représentations ;
- `full` : tout dégelé, LR plus faible sur l'encodeur que sur la tête neuve → mesure le gain pratique.

Et surtout trois régimes d'annotation : **1 %, 10 %, 100 %** des étiquettes. Le bénéfice du SSL
est toujours maximal en régime peu annoté — c'est précisément la promesse du sujet.

Le paramètre `modalities` contrôle ce qui est disponible **au test** :

- `image` — protocole de Hager et al. En clinique, l'image est toujours là ; les métadonnées
  structurées, souvent non. On mesure donc ce que le pré-entraînement a *injecté* dans l'encodeur
  image. C'est le résultat le plus intéressant à présenter.
- `both` — fusion tardive, quand le dossier est disponible.

### Étape 6 — Comparer honnêtement

`src/train/run_experiments.py`

| Code | Méthode | Ce qu'elle isole |
|---|---|---|
| A | Image seule, init aléatoire | plancher |
| B | Image seule, init ImageNet | apport d'un pré-entraînement générique |
| C | Image seule, SimCLR unimodal | apport du SSL **sans** le tabulaire |
| D | Tabulaire seul | plafond de l'information clinique |
| E | Concaténation naïve (MLP tabulaire), supervisé | apport de la fusion **sans** le SSL |
| **F** | **SSL multimodal → test image seule** | **la méthode du projet** |
| G | SSL multimodal → test image + tabulaire | idem, avec le dossier disponible |

Toutes partagent le même `DataBundle` : même split, même prétraitement, même budget.
**F vs C** isole l'apport du tabulaire dans le SSL. **F vs E** isole l'apport du SSL par rapport à
la fusion supervisée naïve. Ce sont les deux comparaisons à mettre en avant.

Ablations (`--ablations`) :

- `abl_naive_tab_enc` — MLP au lieu du FT-Transformer dans le SSL
- `abl_no_corruption` — augmentation tabulaire désactivée (attendu : effondrement)
- `robustness_missing.csv` — on masque 0 → 100 % des métadonnées **au test** et on trace la chute

### Étape 7 — Vérifier avant de conclure

`scripts/smoke_test.py`, `src/eval/metrics.py`

Le smoke test valide 7 points en quelques minutes : formes des tenseurs, décroissance effective de
la perte contrastive sous la valeur du hasard `log(B)`, absence de NaN quand 100 % des features
sont manquantes, effet mesurable du masque sur la représentation, cohérence des métriques.

Les métriques sont en numpy pur (pas de sklearn) : AUROC (Mann-Whitney avec rangs moyens sur les
ex-æquo), AUPRC, balanced accuracy, F1 macro, ECE, et un **intervalle de confiance bootstrap**.
Ce dernier n'est pas décoratif : sur quelques centaines d'exemples de test, un écart d'AUROC de
0.02 entre deux méthodes est dans le bruit.

---

## 4. Choix du dataset

Le sujet mentionne VinDr-Mammo et EMBED. Recommandation pratique, du plus léger au plus lourd :

| Dataset | Taille | Métadonnées tabulaires | Accès |
|---|---|---|---|
| **Synthétique** (fourni) | ~50 Mo | 14 colonnes, manquance structurelle | immédiat |
| **CBIS-DDSM** ← recommandé | ~6 Go (JPEG) | densité, subtilité, forme/marges de masse, type/distribution de calcifications, BI-RADS, pathologie | Kaggle / TCIA, ouvert |
| VinDr-Mammo | ~300 Go (DICOM) | densité, BI-RADS sein, catégories de findings + boîtes | PhysioNet, accréditation requise |
| EMBED | très volumineux | très riche, multi-ethnique | demande d'accès |

**CBIS-DDSM est le meilleur choix pour ce projet**, et pas seulement pour sa taille : ses
métadonnées ont exactement la **manquance structurelle** qui nous intéresse. `mass_shape` et
`mass_margins` n'existent que pour les masses, `calc_type` et `calc_distribution` que pour les
calcifications. Environ 45 % de chaque groupe de colonnes est vide, et cette manquance porte du
sens (MNAR). C'est le scénario idéal pour démontrer l'intérêt d'un encodeur missing-aware.

Le générateur synthétique reproduit délibérément cette structure.

---

## 5. Structure du code

```
configs/default.yaml            dataset synthétique, toute la config commentée
configs/cbis_kaggle.yaml        CBIS-DDSM sur Kaggle GPU (patchs ROI, split officiel)
src/
  utils.py                      seed, device, config, checkpoints, scheduler
  data/
    synthetic.py                générateur de pseudo-mammographies + métadonnées
    real_datasets.py            adaptateurs CBIS-DDSM et VinDr-Mammo
    cbis_paths.py               découverte des chemins, index d'images (0 hypothèse)
    tabular.py                  prétraitement missing-aware + corruption SCARF
    splits.py                   split stratifié, et par patient
    semantic.py                 embeddings textuels des noms de colonnes
    datamodule.py               Dataset PyTorch, augmentations, DataLoaders
  models/
    encoders.py                 ImageEncoder, FeatureTokenizer, FT-Transformer, MLP naïf
    multimodal.py               modèle SSL + modèle aval
  losses/contrastive.py         CLIPLoss (InfoNCE bidir.), NTXent, masque de doublons
  train/
    pretrain.py                 boucle SSL (multimodal ou SimCLR)
    finetune.py                 boucle aval + IC bootstrap
    run_experiments.py          plan d'expériences complet + ablations
  eval/
    metrics.py                  AUROC, AUPRC, ECE, retrieval R@K, bootstrap
    report.py                   figures et tableau markdown
scripts/
  smoke_test.py                 validation de bout en bout
  inspect_cbis.py               diagnostic du dataset Kaggle (à lancer en premier)
  prepare_cbis.py               jointure dicom_info + cache d'images redimensionnées
  prepare_vindr.py              DICOM → PNG (fenêtrage VOI, recadrage Otsu)
  embed_sources.py              réécrit tout le projet en clair dans le notebook
  run_all.sh                    reproduit tous les résultats
notebooks/
  kaggle_cbis.ipynb             ← à importer sur Kaggle (GPU) : contient le projet
  demo_end_to_end.ipynb         démo synthétique, prête pour Colab
RAPPORT_RESULTATS.md            résultats mesurés, ablations, limites
```

Tous les hyperparamètres passent par `configs/default.yaml`, surchargeable en ligne de commande :

```bash
python -m src.train.run_experiments \
  --set model.image.name=resnet50 data.image_size=224 pretrain.epochs=100 \
        model.tabular.semantic_field_emb=auto
```

---

## 6bis. Résultats mesurés sur CBIS-DDSM

Run complet : patchs ROI 224 px, ResNet18 **initialisé ImageNet**, split officiel,
2 276 lésions d'entraînement, 704 de test. **AUROC de malignité :**

| Méthode | 1 % | 10 % | 100 % |
|---|---|---|---|
| A image seule, sans pré-entraînement | 0,588 | 0,587 | 0,635 |
| B image seule, ImageNet | 0,584 | 0,624 | 0,729 |
| C image seule, + SimCLR | 0,616 | 0,669 | 0,749 |
| D tabulaire seul | 0,498 | 0,692 | 0,793 |
| E concaténation naïve | 0,586 | 0,735 | 0,827 |
| **F SSL multimodal → image seule** | **0,630** | **0,741** | 0,782 |
| **G SSL multimodal → image + tab** | **0,652** | **0,788** | **0,831** |

**Le résultat central** : F bat B de **+0,117 à 10 % des étiquettes**, avec des
intervalles de confiance qui ne se chevauchent pas — le seul écart
statistiquement significatif en faveur de la méthode. C'est exactement la
promesse du sujet : le bénéfice est maximal en régime peu annoté. À 100 %, F − B
tombe à +0,053 et l'écart n'est plus significatif.

**Le biais corrigé.** Un premier run avait `pretrained: false` et donnait
F = 0,662 contre B = 0,746 : le SSL partait de poids aléatoires face à ImageNet.
Le passage à `pretrained: true` fait passer la perte InfoNCE de 4,22 à 3,23 et
F de 0,662 à 0,782 — le verdict s'inverse.

**Ce qui limite l'alignement**, mesuré et non supposé. R@10 atteint 0,179 pour un
plafond de 1,0. Deux causes :

* **les lignes tabulaires se dupliquent** — 50,5 % de vecteurs distincts
  seulement, 70 % des lignes dans un groupe dupliqué, plafond de R@1 à 0,798.
  Chez Hager et al. le vecteur compte 75-120 features dont beaucoup de continues,
  donc une empreinte quasi unique ; 10 features discrètes ne suffisent pas ;
* **une colonne n'est pas lisible dans le patch** — la sonde
  `src/eval/probe.py` donne `left_or_right_breast` à gain −0,006. En revanche
  `breast_density` reste lisible (+0,162), contrairement à ce que j'avais supposé.

**Les ablations ne soutiennent pas tout.** L'augmentation tabulaire est justifiée
(−0,017 sans elle, et la perte s'effondre à 0,70 par mémorisation). Le
FT-Transformer, non : le MLP naïf fait 0,791 contre 0,782 et aligne mieux. Avec
10 features, l'attention entre colonnes n'a pas de quoi s'exprimer.

**La réserve la plus importante.** En vue `clinical_only` — densité, latéralité,
incidence, nombre de lésions — `D_tab_only` tombe à **0,492**, soit le hasard, et
la concaténation naïve passe sous l'image seule. Le bon score de la fusion en vue
`full` vient donc essentiellement des descripteurs morphologiques, qui sont des
annotations expertes portant sur la lésion montrée.

Détail complet, tableaux de robustesse et pistes : **`RAPPORT_RESULTATS.md`**.

---

## 6. Résultats attendus et lecture

Sur le dataset synthétique, l'ordre attendu est :

```
D_tab_only     <  A_img_scratch  <  B/C_img_*  <  E_concat_naive  <  F_ours_img  ≤  G_ours_both
```

avec un écart **F − C** qui se creuse quand la fraction d'étiquettes diminue. C'est la figure
`outputs/figures/label_efficiency.png`.

Comment lire les résultats sans se raconter d'histoires :

- Si **R@10 du retrieval reste au niveau du hasard**, le SSL n'a rien appris. Cherchez d'abord du
  côté de `tabular_corruption_rate` (trop bas → solution triviale, trop haut → paires détruites) et
  de la température.
- Si **F ≈ C**, le tabulaire n'apporte rien de plus que le SSL image. Soit les métadonnées sont
  vraiment peu informatives, soit elles fuient déjà dans l'image, soit le batch est trop petit
  (InfoNCE a besoin de négatifs : viser `batch_size ≥ 128`).
- Si **D est très élevé** (> 0.9), vérifiez qu'il ne reste pas une colonne qui fuit la cible.
- Si les intervalles de confiance de F et E se chevauchent largement, **ne concluez pas** :
  augmentez la taille du test ou faites plusieurs graines.

---

## 7. Passer aux vraies données

### CBIS-DDSM sur Kaggle — le chemin recommandé

Dataset : [`awsaf49/cbis-ddsm-breast-cancer-image-dataset`](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)

```
/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset/
├── csv/
│   ├── mass_case_description_{train,test}_set.csv
│   ├── calc_case_description_{train,test}_set.csv
│   ├── dicom_info.csv        <- le pont indispensable
│   └── meta.csv
└── jpeg/
    └── … / <SeriesInstanceUID>/1-1.jpg   (parfois aussi 1-2.jpg)
```

**Le dataset ne se lit pas directement**, et c'est le point à comprendre : les CSV
de description pointent vers les chemins **DICOM** d'origine
(`Mass-Training_P_00001_LEFT_CC/1.3.6.../000000.dcm`) alors que les images sont
des **JPEG** rangés par `SeriesInstanceUID`. Le pont est `dicom_info.csv`, dont
`PatientID` vaut exactement le premier segment du chemin CSV, et dont
`SeriesDescription` distingue *full mammogram* / *cropped* / *ROI mask*.

**Aucune hypothèse n'est faite sur les chemins.** Selon le mode de montage, la
racine est `/kaggle/input/<slug>/` ou `/kaggle/input/datasets/<user>/<slug>/`, et
le nombre de niveaux sous `jpeg/` varie. Tout passe donc par
`src/data/cbis_paths.py` : un seul `os.walk` construit un index
`{nom_de_dossier → fichiers}`, et la résolution se fait par le
`SeriesInstanceUID` — le nom du dossier terminal, quelle que soit la profondeur
au-dessus. Testé sur `jpeg/UID/`, `jpeg/X/UID/` et `jpeg/X/Y/UID/` : manifestes
identiques, 100 % de résolution dans les trois cas.

Trois commandes, dans cet ordre. `--root` est **optionnel** (auto-détection) :

```bash
# 1) inspection — n'écrit rien, vérifie que la jointure fonctionne
python scripts/inspect_cbis.py

# 2) préparation — jointure + cache d'images redimensionnées (~15-25 min)
python scripts/prepare_cbis.py --out /kaggle/working/cbis_cache \
    --size 256 --kind roi

# 3) expériences
python -m src.train.run_experiments --config configs/cbis_kaggle.yaml --ablations
```

Si l'auto-détection échoue, passez le chemin affiché dans le panneau Data de
Kaggle. Un pointage approximatif est toléré (sur `csv/`, sur `jpeg/`, ou un cran
trop haut) ; un chemin inexistant échoue immédiatement avec la commande à lancer
pour lister `/kaggle/input`.

```bash
python scripts/inspect_cbis.py \
    --root /kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
```

Ou bien : ouvrez **`notebooks/kaggle_cbis.ipynb`** directement sur Kaggle, il
enchaîne tout avec les figures.

`--kind roi` prend les patchs recadrés autour de la lésion (rapides, signal
dense). Pour la mammographie complète : `--kind full --size 512`.

#### Les cinq pièges de ce dataset, et comment ils sont traités

| Piège | Conséquence si ignoré | Traitement |
|---|---|---|
| `breast_density` (fichiers *mass*) vs `breast density` (fichiers *calc*) | après normalisation, **deux colonnes homonymes** → tout accès renvoie un DataFrame 2D et plante | normalisation des noms **avant** la concaténation, plus une assertion |
| Certaines séries contiennent 2 JPEG : le patch **et** le masque ROI | ~30 % des entrées d'entraînement sont des masques binaires ; le modèle apprend sur du vide | test de binarité de l'histogramme dans `prepare_cbis.py` (un masque est quasi noir-et-blanc pur) |
| `breast_density = 0` | 0 signifie « non évalué », pas « densité faible » ; traité comme une valeur numérique il fausse la standardisation | converti en `NaN`, donc pris en charge par le token `[MISSING]` |
| Le split officiel est fait **séparément** pour les masses et les calcifications | un patient peut être dans `mass_test` **et** `calc_train` → AUROC de test optimiste | `enforce_group_disjoint: true` retire ces patients du train (jamais du test) |
| `calc_type` a ~45 modalités composées (`PLEOMORPHIC-FINE_LINEAR_BRANCHING`), souvent à 1-2 occurrences | ces modalités identifient le patient : la perte contrastive apprend un raccourci au lieu de la sémantique | `min_category_count: 20`, la queue est regroupée en `OTHER` |
| La racine et la profondeur sous `jpeg/` varient selon le montage | tout chemin codé en dur casse au premier changement | `src/data/cbis_paths.py` : index par `os.walk`, résolution par nom de dossier terminal |

#### Une réserve à assumer sur CBIS-DDSM

Le tabulaire de CBIS contient deux natures d'information très différentes :

* des champs de **routine** — densité mammaire, latéralité, incidence, nombre
  d'anomalies ;
* des **descripteurs morphologiques** — `mass_shape`, `mass_margins`,
  `calc_type`, `calc_distribution`, `subtlety`. Ce sont des annotations de
  radiologue **portant sur la lésion montrée dans l'image**.

Les seconds sont légitimes (ils figurent dans le compte rendu) mais très
informatifs : attendez-vous à une baseline tabulaire seule autour de 0.75-0.85
d'AUROC. Ce n'est pas une fuite, mais ce n'est pas non plus le scénario le plus
convaincant.

D'où l'option `data.clinical_only: true`, qui restreint le vecteur tabulaire aux
champs de routine. Le tabulaire seul devient nettement plus faible, le gain du
SSL multimodal plus modeste — mais l'argument est bien plus solide.
**Faites tourner les deux** : la version `clinical_only` est celle à mettre en
avant dans une discussion scientifique, l'autre sert de borne supérieure.

### CBIS-DDSM téléchargé en local

Identique, en changeant les chemins :

```bash
python scripts/prepare_cbis.py --root /data/cbis-ddsm --out /data/cbis_cache \
    --size 256 --kind roi
python -m src.train.run_experiments --config configs/cbis_kaggle.yaml \
    --set data.root=/data/cbis_cache out_dir=outputs data.num_workers=8
```

### VinDr-Mammo

```bash
# 1) conversion DICOM -> PNG (fenêtrage VOI, recadrage Otsu sur le sein)
python scripts/prepare_vindr.py --root /data/vindr --out /data/vindr/images_png --size 1024

# 2) entraînement
python -m src.train.run_experiments \
  --set data.source=vindr data.root=/data/vindr data.image_size=224 data.target=birads
```

### Ajouter votre propre dataset

Écrivez une fonction qui renvoie
`(chemins_images, dataframe, num_cols, cat_cols, descriptions)` et branchez-la dans
`build_datasets`. Les descriptions textuelles des colonnes sont ce qui alimente les embeddings
sémantiques — écrivez-les en anglais et de façon explicite.

---

## 8. Pièges rencontrés (et comment ils sont traités)

Ces points ne sont pas théoriques : chacun a été identifié en revue de code sur ce projet.

| Piège | Symptôme | Traitement |
|---|---|---|
| Fuite du label via `assessment` / `pathology` | AUC tabulaire ≈ 1.0 | liste `banned` dans `build_datasets` |
| Même patiente en train et en test | AUC surestimée de plusieurs points | split par groupe |
| PNG 16 bits chargé via `.convert("L")` | 99 % des pixels saturent à 255, image blanche | rééchelonnage explicite dans `_load_pil` |
| Colonne 100 % manquante au train | `std = NaN` → perte NaN | remplacement de la valeur, pas seulement du test |
| Masque remis à `False` après corruption | on déclare « observé » un 0 qui n'existe pas | le masque suit la ligne donneuse |
| `load_state_dict(strict=False)` silencieux | on croit évaluer le SSL, on évalue du bruit | erreur levée si les encodeurs ne se chargent pas |
| AUROC = NaN en validation à 1 % de labels | `NaN > x` est faux → dernière époque conservée | repli sur `balanced_accuracy` |
| Baseline « naïve » utilisant le FT-Transformer | la comparaison phare ne mesure plus rien | override explicite `arch=naive_mlp` |
| RNG partagé entre workers du DataLoader | corruption identique dans tous les workers | RNG dérivé de `get_worker_info()` |
| BatchNorm sur un dernier batch de taille 1 | plantage en fin d'époque | `drop_last` conditionnel |

---

## 9. Extensions pour aller vers la thèse

Par ordre croissant d'ambition :

1. **Plusieurs graines** et report en moyenne ± écart-type, en plus des IC bootstrap.
2. **Pertes contrastives supervisées faibles** : utiliser la densité BI-RADS (disponible sans
   biopsie) pour définir des positifs supplémentaires.
3. **Reconstruction tabulaire** en tâche auxiliaire : prédire les colonnes masquées depuis
   l'image — force l'encodeur image à retenir explicitement l'information clinique.
4. **Attention croisée** entre tokens d'image et tokens de colonnes au lieu de la fusion tardive :
   permet d'inspecter *quelle colonne* regarde *quelle région* de l'image, ce qui est un argument
   d'interprétabilité clinique fort.
5. **Transfert inter-datasets** : pré-entraîner sur CBIS-DDSM, évaluer sur VinDr-Mammo. C'est ici
   que les embeddings sémantiques de colonnes deviennent indispensables, puisque les schémas
   tabulaires diffèrent. C'est probablement la contribution la plus originale accessible.
6. **Manquance MNAR modélisée explicitement** : apprendre un modèle du mécanisme de manquance
   plutôt que de le subir.

---

## 10. Références

- [1] Hager, Menten, Rueckert. *Best of Both Worlds: Multimodal Contrastive Learning with Tabular
  and Imaging Data*. CVPR 2023.
- [2] Chen, Kornblith, Norouzi, Hinton. *A Simple Framework for Contrastive Learning of Visual
  Representations* (SimCLR). ICML 2020.
- [3] Radford et al. *Learning Transferable Visual Models From Natural Language Supervision*
  (CLIP). ICML 2021.
- [4] Bahri, Jiang, Tay, Metzler. *SCARF: Self-Supervised Contrastive Learning using Random Feature
  Corruption*. ICLR 2022.
- [5] Gorishniy, Rubachev, Khrulkov, Babenko. *Revisiting Deep Learning Models for Tabular Data*
  (FT-Transformer). NeurIPS 2021.
- [6] Nguyen et al. *VinDr-Mammo: A large-scale benchmark dataset for computer-aided diagnosis in
  full-field digital mammography*. Scientific Data, 2023.
- [7] Lee et al. *A curated mammography data set for use in computer-aided detection and diagnosis
  research* (CBIS-DDSM). Scientific Data, 2017.
- [8] Jeong et al. *The EMory BrEast imaging Dataset (EMBED)*. Radiology: AI, 2023.
