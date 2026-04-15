# Optimisation des Ressources CPU par Machine Learning
### Projet de Fin d'Études (PFE) — Génie Logiciel & Systèmes d'Information

> Détection d'anomalies et prédiction de l'utilisation CPU sur des serveurs AWS EC2,  
> en suivant la méthodologie **CRISP-DM**.

---

## Présentation du projet

Ce projet vise à optimiser la gestion des ressources CPU de serveurs cloud (AWS EC2) en combinant :
- la **prédiction** de la charge CPU future (modèles ML supervisés)
- la **détection d'anomalies** (méthodes non supervisées + fusion avec la prédiction)

Les données proviennent du dataset public **Numenta Anomaly Benchmark (NAB)** — métriques CloudWatch AWS, disponibles sur [Kaggle](https://www.kaggle.com/datasets/boltzmannbrain/nab).

---

## Structure du dépôt

```
ml-pfe/
│
├── data/
│   └── realAWSCloudwatch/          # Données brutes AWS CloudWatch (5 serveurs EC2)
│
├── notebooks/
│   ├── phase1_data_exploration.ipynb     # Exploration & fusion multi-serveurs
│   ├── phase2_feature_engineering.ipynb  # Préparation des features (CRISP-DM)
│   ├── phase3_prediction.ipynb           # Modèles ML (RF, XGBoost, LightGBM)
│   └── phase4_detection.ipynb            # Détection d'anomalies (IF, LOF, SVM)
│
├── outputs/
│   ├── figures/                    # Graphiques exportés
│   │   ├── comparaison_3modeles.png
│   │   ├── feature_importance_et_prediction.png
│   │   ├── heatmap_mae_nb_features.png
│   │   └── top10_mae.png
│   └── models/                     # Modèles sauvegardés (joblib)
│
├── docs/                           # Documentation complémentaire
│
├── requirements.txt                # Dépendances Python
├── .gitignore
└── README.md
```

---

## Méthodologie — CRISP-DM

| Phase | Notebook | Description |
|---|---|---|
| Business Understanding | — | Optimisation CPU cloud, réduction des incidents |
| Data Understanding | `phase1` | Exploration, statistiques, visualisation des séries |
| Data Preparation | `phase2` | Filtre Hampel, log-transform, feature engineering, ADF |
| Modeling | `phase3` | RF / XGBoost / LightGBM, TimeSeriesSplit, tuning |
| Evaluation | `phase3` | MAE, RMSE, R², baseline naïve, robustesse |
| Deployment | `phase4` | Détection anomalies, score de fusion, visualisation |

---

## Modèles utilisés

### Prédiction (Phase 3)
| Modèle | Description |
|---|---|
| Random Forest | Ensemble d'arbres de décision |
| XGBoost | Gradient boosting optimisé |
| LightGBM | Gradient boosting rapide (Microsoft) |

Protocole d'évaluation : **TimeSeriesSplit (5 splits)** — aucun data leakage.  
Baseline de référence : modèle de persistance (`valeur(t) = valeur(t-1)`).

### Détection d'anomalies (Phase 4)
| Modèle | Type |
|---|---|
| Isolation Forest | Ensemble non supervisé |
| Local Outlier Factor (LOF) | Densité locale |
| One-Class SVM | Frontière de décision |
| Règle 3-sigma | Baseline statistique |

Les résultats des 3 modèles ML sont fusionnés avec l'erreur de prédiction via un **score pondéré** (seuil = 0.6).

---

## Features engineerées (Phase 2)

| Feature | Type | Description |
|---|---|---|
| `lag_1` … `lag_10` | Autocorrélation | Valeurs passées t-k |
| `moyenne_mobile_5/10` | Tendance | Lissage court/moyen terme |
| `rolling_std_5/10` | Volatilité | Instabilité locale |
| `diff_1`, `diff_2` | Stationnarité | Changements brusques |
| `hour_sin/cos` | Cyclique | Patterns journaliers |
| `weekday_sin/cos` | Cyclique | Patterns hebdomadaires |

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/Rihab160/ml-pfe.git
cd ml-pfe

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer Jupyter
jupyter notebook
```

---

## Exécution

Les notebooks doivent être exécutés **dans l'ordre** :

```
phase1 → phase2 → phase3 → phase4
```

Chaque phase produit les fichiers nécessaires à la suivante. Les notebooks détectent automatiquement leur dossier — aucun chemin absolu à modifier.

---

## Résultats clés

- **Meilleur modèle de prédiction** : LightGBM / XGBoost (MAE inférieur à la baseline naïve)
- **Détection d'anomalies** : score de fusion pondéré (consensus ML + erreur prédiction) avec analyse de sensibilité du seuil
- **Stationnarité** : confirmée par test ADF après transformation log + différenciation

---

## Données

- **Source** : [Numenta Anomaly Benchmark — AWS CloudWatch](https://www.kaggle.com/datasets/boltzmannbrain/nab)
- **Format** : séries temporelles CSV, fréquence 5 minutes
- **Serveurs** : 5 instances EC2 sur ~3 semaines

> Les fichiers CSV intermédiaires (`df_features_ready.csv`, `df_original_ready.csv`) sont générés automatiquement par `phase2_feature_engineering.ipynb` et ne sont pas versionnés.

---

## Auteur

**Rihab** — Étudiante en Génie Logiciel & Systèmes d'Information, 3ème année  
GitHub : [@Rihab160](https://github.com/Rihab160)
