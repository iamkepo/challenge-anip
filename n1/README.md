# Challenge ANIP - Analyse Démographique WPP 2024

## Description

Ce projet vise à collecter, nettoyer, enrichir et visualiser les données démographiques issues du **World Population Prospects 2024 (WPP 2024)**.  
L'objectif est de produire un **dataset prêt pour Power BI** et des visualisations interactives permettant d'analyser :

- L'évolution de la population par pays et région
- Les indicateurs démographiques clés : taux de croissance, ratio hommes/femmes, indice de développement
- Les disparités régionales et tendances temporelles

Le pipeline est organisé en trois tâches principales :

1. **Collecte & Préparation des données**
2. **Exploration & Analyse**
3. **Visualisation & Insights pour Power BI**

---

## Structure du projet

n1/
├─ data/
│  └─ WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_FULL.xlsx
├─ output/
│  ├─ WPP2024_DEMOGRAPHIC_CLEAN.csv
│  ├─ WPP2024_DEMOGRAPHIC_GLOSSARY.csv
│  ├─ WPP2024_DEMOGRAPHIC_ENRICHED.csv
│  ├─ WPP2024_DEMOGRAPHIC_ANOMALIES.csv
│  └─ graphs/
├─ output/powerbi_full/
│  ├─ WPP2024_POWERBI_READY.csv
│  ├─ graphs/
│  └─ maps/
├─ Tache1_ANIP_WPP.py
├─ Tache2_ANIP_WPP.py
├─ Tache3_ANIP_WPP.py
└─ requirements.txt

---

## Installation

### 1️⃣ Cloner le dépôt

```bash
git clone <repo-url>
cd n1

2️⃣ Créer un environnement virtuel et installer les dépendances

python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
pip install --upgrade pip
pip install -r requirements.txt

Note : Pour l’export PNG des cartes Plotly, le package Kaleido est inclus dans requirements.txt.

⸻

Exécution des scripts
	1.	Tâche 1 : Collecte & Nettoyage

python Tache1_ANIP_WPP.py

	2.	Tâche 2 : Enrichissement & Analyse

python Tache2_ANIP_WPP.py

	3.	Tâche 3 : Visualisation & Insights Power BI

python Tache3_ANIP_WPP.py

Vérifier les fichiers générés dans output/ et output/powerbi_full/ après chaque tâche.

⸻

Contenu des fichiers de sortie
	•	CSV principaux :
	•	WPP2024_DEMOGRAPHIC_CLEAN.csv → dataset nettoyé
	•	WPP2024_DEMOGRAPHIC_ENRICHED.csv → dataset enrichi avec KPI
	•	WPP2024_POWERBI_READY.csv → dataset prêt pour Power BI
	•	Graphiques et cartes :
	•	graphs/ → visualisations exploratoires et narratifs
	•	maps/ → cartes choroplèthes interactives et PNG
	•	Anomalies :
	•	WPP2024_DEMOGRAPHIC_ANOMALIES.csv → populations négatives ou décroissantes
	•	Glossaire :
	•	WPP2024_DEMOGRAPHIC_GLOSSARY.csv → description des variables, unités, sources

⸻

Notes importantes
	•	Les cartes choroplèthes nécessitent la colonne iso3.
	•	Les fichiers HTML des cartes peuvent être ouverts dans un navigateur et intégrés dans Power BI.
	•	Toutes les colonnes nécessaires pour Power BI (iso3, pop_growth_rate, sex_ratio, regional_dev_index, population_pct, pop_category) sont incluses dans le dataset final.

⸻

Auteurs
	•	Christ-Amour Kakpo – Développeur Full-Stack / Auteur des scripts Tâche 1 à 3

⸻

Sources
	•	UN World Population Prospects 2024