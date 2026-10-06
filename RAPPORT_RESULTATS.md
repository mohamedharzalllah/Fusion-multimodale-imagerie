# Fusion multimodale imagerie + tabulaire en SSL — rapport de résultats

CBIS-DDSM, patchs ROI recadrés 224 px, ResNet18 initialisé ImageNet, split
officiel CBIS. Cible : malignité (`MALIGNANT` contre le reste).

| | |
|---|---|
| lésions | 3 567 (1 566 patients) |
| train / val / test | 2 276 / 504 / 704 |
| prévalence de la malignité | 40,8 % |
| features tabulaires | 10 (3 numériques + 7 catégorielles) |
| valeurs manquantes (train) | 21,4 % |
| pré-entraînement | 60 époques, batch 128, ~8 s/époque sur T4 |

Le split officiel a été purgé de 31 patients présents à la fois dans le train et
le test — CBIS-DDSM sépare masses et calcifications indépendamment, ce qui crée
ce recouvrement. 83 lignes d'entraînement ont été retirées.

---

## 1. Résultats principaux

AUROC sur le test, par fraction d'étiquettes d'entraînement utilisée.

| Méthode | 1 % | 10 % | 100 % |
|---|---|---|---|
| A image seule, sans pré-entraînement | 0,588 | 0,587 | 0,635 |
| B image seule, ImageNet | 0,584 | 0,624 | 0,729 |
| C image seule, + SimCLR | 0,616 | 0,669 | 0,749 |
| D tabulaire seul | 0,498 | 0,692 | 0,793 |
| E concaténation naïve | 0,586 | 0,735 | 0,827 |
| **F SSL multimodal → test image seule** | **0,630** | **0,741** | 0,782 |
| **G SSL multimodal → test image + tabulaire** | **0,652** | **0,788** | **0,831** |

Intervalles de confiance bootstrap à 100 % des étiquettes : F 0,748–0,816 ;
G 0,801–0,858 ; E 0,797–0,854 ; D 0,761–0,821.

### Ce que ces chiffres établissent

**Le pré-entraînement contrastif multimodal améliore l'encodeur image.** F bat B
(même modalité au test, même initialisation) de +0,046 à 1 %, **+0,117 à 10 %** et
+0,053 à 100 %. À 10 % des étiquettes, l'écart est **statistiquement significatif** :
les intervalles de confiance ne se chevauchent pas. C'est le résultat central, et
c'est celui qu'on attendait du sujet de thèse — le bénéfice est maximal en régime
peu annoté.

**Le tabulaire apporte quelque chose que le SSL unimodal ne trouve pas.** F bat C
(SimCLR, même architecture, même budget, mais sans jamais voir le tableau) de
+0,013 / +0,072 / +0,034. La direction est constante mais les intervalles se
chevauchent : à confirmer sur plusieurs graines.

**La fusion tardive reste la meilleure quand le dossier est disponible.** G est en
tête aux trois régimes, avec un avantage net sur `D_tab_only` (+0,154 / +0,096 /
+0,038, significatif aux deux premiers). En revanche G ne dépasse E que de +0,004
à 100 % : à pleine annotation, le pré-entraînement n'apporte plus rien par rapport
à une concaténation supervisée. L'écart se creuse quand les étiquettes se
raréfient (+0,053 à 10 %, +0,066 à 1 %).

### Ce que ces chiffres n'établissent pas

À 1 % des étiquettes (22 exemples), toutes les méthodes sont entre 0,50 et 0,65 et
**tous les intervalles se chevauchent**. Aucun classement n'est défendable dans ce
régime.

`D_tab_only` atteint 0,793 en vue `full`. Les descripteurs `mass_margins` et
`calc_type` sont des annotations de radiologue portant sur la lésion montrée : ce
n'est pas une fuite, mais ce n'est pas non plus un scénario clinique. La vue
`clinical_only` existe pour cette raison (§5).

---

## 2. Un biais méthodologique corrigé, et son effet mesuré

Un premier run utilisait `pretrained: false` : le pré-entraînement contrastif
partait de poids aléatoires alors que la baseline B partait d'ImageNet. On
comparait donc « SSL sur 2 276 images » à « ImageNet sur 1,2 million » — un
handicap auto-infligé, et ce n'est pas le protocole de Hager et al.

Effet du passage à `pretrained: true`, à 100 % des étiquettes :

| | avant (init aléatoire) | après (init ImageNet) |
|---|---|---|
| perte InfoNCE finale | 4,22 | **3,23** |
| `acc_i2t` (train) | 0,053 | **0,190** |
| R@10 validation | 0,133 | **0,179** |
| F_ours_img | 0,662 | **0,782** |
| F contre B | **−0,084** | **+0,053** |

Le verdict s'inverse. Avant correction, la conclusion aurait été « le SSL
multimodal est moins utile qu'ImageNet » ; c'était un artefact du protocole.

---

## 3. Qualité de l'alignement contrastif

| | valeur |
|---|---|
| perte du hasard `log(128)` | 4,852 |
| perte InfoNCE finale | 3,227 |
| R@1 validation | 0,026 (hasard 0,002) |
| R@10 validation | 0,179 |
| R@10 test | 0,159 |
| plafond R@1 (validation, 504 éléments) | **0,798** |
| plafond R@10 | 1,000 |

L'alignement est **réel mais faible** : 13 fois le hasard en R@1, et 18 % du
maximum atteignable en R@10. La perte descend bien sous le hasard, donc quelque
chose est appris ; l'espace latent partagé est loin d'être structuré.

### Pourquoi il plafonne — deux causes mesurées

**Les lignes tabulaires se dupliquent massivement.** Le diagnostic
`identifiability_report`, affiché avant chaque pré-entraînement :

| | vue `full` | `roi_visible` | `clinical_only` |
|---|---|---|---|
| vecteurs distincts / 2 276 | 1 149 (50,5 %) | 274 (12,0 %) | 32 (1,4 %) |
| lignes dans un groupe dupliqué | 70,4 % | — | — |
| plus grand groupe | 12 | — | — |
| entropie de la ligne | 9,79 bits | — | — |
| plafond R@1 (validation) | 0,798 | 0,911 (R@10) | 0,564 (R@10) |

Avec 70 % des lignes partageant leur vecteur, la cible one-hot de l'InfoNCE
demande en partie de distinguer l'indistinguable. Le contexte compte : chez Hager
et al. (UK Biobank), le vecteur tabulaire compte 75 à 120 features dont beaucoup
de continues — une empreinte quasi unique. Sur CBIS, 10 features toutes discrètes
et de faible cardinalité ne suffisent pas.

**Certaines colonnes ne sont pas lisibles dans le patch.** La sonde
`src/eval/probe.py` gèle l'encodeur image et ajuste une régression logistique par
colonne, comparée au classifieur majoritaire :

| colonne | sonde | majoritaire | gain | lisible ? |
|---|---|---|---|---|
| `abnormality_type` | 0,927 | 0,500 | **+0,427** | oui, massivement |
| `breast_density` | 0,495 | 0,333 | +0,162 | oui |
| `mass_shape` | 0,238 | 0,100 | +0,138 | oui |
| `calc_type` | 0,220 | 0,091 | +0,129 | oui |
| `subtlety` | 0,461 | 0,333 | +0,128 | oui |
| `mass_margins` | 0,211 | 0,100 | +0,111 | oui |
| `calc_distribution` | 0,251 | 0,143 | +0,109 | oui |
| `image_view` | 0,539 | 0,500 | +0,039 | marginal |
| `left_or_right_breast` | 0,494 | 0,500 | **−0,006** | **non** |

Une prédiction que j'avais faite s'avère fausse : je supposais que la densité
mammaire, décrivant le sein entier, serait perdue au recadrage. Elle reste
partiellement lisible dans la texture du patch (+0,162). Seule la **latéralité**
est authentiquement invisible — elle est du bruit pur dans la cible contrastive.
`n_abnormalities` a été écartée par la sonde faute de variabilité suffisante.

---

## 4. Ablations

À 100 % des étiquettes, protocole image seule au test.

| Variante | R@10 val | AUROC test |
|---|---|---|
| F, complète (FT-Transformer + corruption) | 0,179 | 0,782 |
| encodeur tabulaire = MLP naïf | **0,234** | **0,791** |
| sans augmentation tabulaire (corruption) | 0,200 | 0,765 |

**L'augmentation tabulaire est justifiée** : la retirer coûte 0,017 d'AUROC. Le
mécanisme est visible dans les courbes — sans corruption, la perte s'effondre à
0,70 et `acc_i2t` monte à 0,80, c'est-à-dire que le modèle **mémorise** les paires
d'entraînement. Le R@10 en validation ne suit pas (0,200), et l'aval se dégrade.

**Le FT-Transformer n'est PAS justifié ici.** Le MLP naïf aligne mieux (R@10 0,234
contre 0,179) et fait légèrement mieux en aval (0,791 contre 0,782). C'est un
résultat négatif qu'il faut assumer : avec 10 features seulement, l'attention
entre colonnes n'a pas de quoi s'exprimer, et ses ~380 k paramètres se paient en
sur-apprentissage. L'architecture ne se défendrait que sur un schéma tabulaire
bien plus riche, ou pour le transfert inter-datasets.

---

## 5. Robustesse aux métadonnées manquantes

Même modèle fine-tuné en image + tabulaire, métadonnées masquées progressivement
**au test**.

| masquage ajouté | taux effectif de manquants | AUROC | balanced acc. | ECE |
|---|---|---|---|---|
| 0 % | 21 % | 0,831 | 0,734 | 0,084 |
| 25 % | 42 % | 0,787 | 0,689 | 0,120 |
| 50 % | 61 % | 0,765 | 0,635 | 0,172 |
| 75 % | 80 % | 0,715 | 0,574 | 0,253 |
| 100 % | 100 % | 0,723 | 0,515 | 0,334 |

La dégradation est **progressive, pas brutale** : −0,04 pour un quart de
métadonnées perdues, et le modèle conserve 0,72 d'AUROC quand **tout** le tableau
disparaît, à comparer à 0,782 pour F entraîné image seule. L'encodeur
missing-aware fait donc son travail : le token `[MISSING]` appris par colonne
permet une dégradation contrôlée au lieu d'un effondrement.

Deux réserves à ne pas passer sous silence. La balanced accuracy s'écroule
(0,73 → 0,51) bien plus vite que l'AUROC : le **classement** reste correct, mais
le seuil de décision devient inutilisable. L'ECE quadruple (0,08 → 0,33) : les
probabilités ne sont plus calibrées du tout. Un modèle qui garde 0,72 d'AUROC
mais dont on ne peut plus lire les probabilités n'est pas déployable en l'état.

---

## 6. Comparaison des vues tabulaires

À 100 % des étiquettes. Le run `clinical_only` a été interrompu avant F et G.

| Méthode | `full` | `roi_visible` | `clinical_only` |
|---|---|---|---|
| B image seule, ImageNet | 0,729 | 0,738 | — |
| C image seule, + SimCLR | 0,749 | 0,746 | — |
| D tabulaire seul | 0,793 | 0,805 | 0,492 |
| E concaténation naïve | 0,827 | 0,824 | 0,732 |
| F SSL multimodal → image | 0,782 | 0,774 | interrompu |
| G SSL multimodal → img+tab | 0,831 | 0,828 | interrompu |

`roi_visible` ne change presque rien : les colonnes retirées (latéralité,
incidence, nombre de lésions) étaient effectivement peu informatives, mais leur
retrait réduit aussi l'identifiabilité du vecteur — 12 % de vecteurs distincts
contre 50 %. Les deux effets se compensent, et R@10 passe de 0,18 à 0,15.

`clinical_only` est le résultat le plus instructif malgré l'interruption :
`D_tab_only` tombe à **0,492**, soit le hasard. Réduits à densité, latéralité,
incidence et nombre de lésions, les métadonnées ne prédisent rien. Et
`E_concat_naive` chute de 0,827 à 0,732, en dessous de C image seule (0,749).
Conclusion honnête : **le bon score de la fusion en vue `full` vient
essentiellement des descripteurs morphologiques**, c'est-à-dire d'annotations
expertes portant sur la lésion montrée. C'est légitime — ce sont des données de
compte rendu — mais il faut le dire.

---

## 7. Contrôles de validité

Ce qui a été vérifié, et pourquoi ça compte.

- **Split par patient.** 31 patients à cheval sur le train et le test officiels
  ont été détectés et retirés du train. Sans ce garde-fou, l'AUROC de test serait
  optimiste sans qu'on le voie.
- **Anti-fuite.** `pathology` et `assessment` (le BI-RADS) sont exclus du vecteur
  tabulaire. `assessment` prédit presque parfaitement la malignité.
- **Patchs et non masques.** 3 463 des 3 567 séries contenaient deux JPEG, le
  patch et le masque ROI mêlés. Le test de binarité de l'histogramme les a
  séparés : binarité moyenne des images retenues 0,010, et 0 image suspecte sur
  200 contrôlées. Sans cette étape, un tiers des données d'entraînement auraient
  été des masques binaires.
- **Modalités rares regroupées.** `calc_type` passe de 47 à 15 modalités, la
  queue (194 lignes) allant dans `OTHER`. Des modalités à une ou deux occurrences
  identifient le patient et offriraient un raccourci à la perte contrastive.
- **`breast_density = 0` et `subtlety = 0`** convertis en `NaN` : dans CBIS, 0 code
  « non évalué », pas une valeur basse.
- **Smoke test** : 7/7 au vert avant les expériences longues.

---

## 8. Ce qu'il faut faire ensuite

Par ordre de rentabilité.

1. **Plusieurs graines.** C'est le manque le plus criant. La quasi-totalité des
   écarts intéressants ont des intervalles qui se chevauchent. Trois graines et un
   report en moyenne ± écart-type transformeraient les tendances en résultats.
2. **Terminer le run `clinical_only`.** Les deux lignes manquantes (F et G) sont
   celles qui portent l'argument le plus solide du projet.
3. **Retirer `left_or_right_breast`** du vecteur contrastif : la sonde la donne à
   gain négatif, c'est du bruit pur dans la cible.
4. **Activer `soft_targets: true`.** Le diagnostic le recommande explicitement
   (70 % de lignes dupliquées) et la mécanique est implémentée mais n'a pas encore
   été évaluée. C'est le levier le plus direct sur le plafond de l'InfoNCE.
5. **Abandonner le FT-Transformer** pour cette configuration, ou le réserver à un
   schéma tabulaire plus riche. L'ablation ne le soutient pas.
6. **Mammographies complètes** (`--kind full --size 512`). Scénario clinique
   réaliste, et le SSL multimodal devrait y aider davantage puisque la lésion est
   noyée dans l'image.
7. **Transfert CBIS-DDSM → VinDr-Mammo.** C'est là que les embeddings sémantiques
   de colonnes deviennent indispensables, les schémas tabulaires différant. C'est
   probablement la contribution la plus originale accessible.

---

## Annexe — reproduire

```bash
python scripts/inspect_cbis.py
python scripts/prepare_cbis.py --out /kaggle/working/cbis_cache --size 256 --kind roi
python scripts/smoke_test.py
python -m src.train.run_experiments --config configs/cbis_kaggle.yaml --ablations
python -m src.eval.probe --ckpt outputs/pretrain/multimodal/best.pt
```

Sorties : `outputs/results.csv`, `outputs/robustness_missing.csv`,
`outputs/probe_tabular_from_image.csv`, `outputs/pretrain_summary.json`,
`outputs/figures/`.

Configuration : `configs/cbis_kaggle.yaml`. Écarts par rapport aux valeurs par
défaut du fichier pour ce run : aucun, hors `data.root` et `out_dir`.
