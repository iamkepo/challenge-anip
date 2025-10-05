# Tâche 2 : Exploration & Analyse (version enrichie pour Power BI)
# Auteur : Christ-Amour Kakpo
# Objectif : Produire un dataset consolidé avec tous les KPI nécessaires pour la Tâche 3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =======================
# 1️⃣ Chemins
# =======================
input_file = "outputs/tache1/WPP2024_DEMOGRAPHIC_CLEAN.csv"
output_file = "outputs/tache2/WPP2024_DEMOGRAPHIC_ENRICHED.csv"
anomalies_file = "outputs/tache2/WPP2024_DEMOGRAPHIC_ANOMALIES.csv"
graphs_dir = "outputs/tache2/graphs"
os.makedirs("outputs/tache2", exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

# =======================
# 2️⃣ Charger le dataset
# =======================
df = pd.read_csv(input_file)
print("✅ Dataset chargé :", df.shape)

# =======================
# 3️⃣ Analyse descriptive & nettoyage
# =======================
# Remplacer valeurs manquantes pour éviter erreurs
fill_cols = ["population_total","population_male","population_female",
             "pop_growth_rate","sex_ratio","regional_dev_index",
             "life_expectancy","fertility_rate","birth_rate","death_rate"]
for col in fill_cols:
    if col not in df.columns:
        df[col] = 0

# Supprimer lignes population <=0
df = df[df["population_total"] > 0]

# =======================
# 4️⃣ Détection anomalies
# =======================
# Valeurs manquantes
missing = df.isna().sum()
print("Valeurs manquantes :\n", missing[missing > 0])

# Populations décroissantes suspectes
df["pop_diff"] = df.groupby("region")["population_total"].diff()
anomalies_pop = df[df["pop_diff"] < 0]
print("⚠️ Populations décroissantes suspectes :", anomalies_pop.shape[0])

# Sauvegarder anomalies
anomalies = pd.concat([anomalies_pop]).drop_duplicates()
anomalies.to_csv(anomalies_file, index=False)
print(f"💾 Anomalies sauvegardées : {anomalies_file}")

# =======================
# 5️⃣ Création nouvelles variables
# =======================
# Taux de croissance annuel
df["pop_growth_rate"] = df.groupby("region")["population_total"].pct_change() * 100

# Ratio hommes/femmes
df["sex_ratio"] = df.apply(
    lambda row: row["population_male"]/row["population_female"]
    if row["population_female"]>0 else 1, axis=1
)

# Indice composite de développement régional
components = []
if "median_age" in df.columns:
    components.append(df["median_age"] / df["median_age"].max())
if "birth_rate" in df.columns:
    components.append(1 - df["birth_rate"] / df["birth_rate"].max())
if "life_expectancy" in df.columns:
    components.append(df["life_expectancy"] / df["life_expectancy"].max())

df["regional_dev_index"] = sum(components)/len(components) if components else 0

# Part de la population mondiale
df["world_population"] = df.groupby("year")["population_total"].transform("sum")
df["population_pct"] = df["population_total"] / df["world_population"] * 100

# =======================
# 6️⃣ Agrégations (par région / année)
# =======================
agg_cols = ["population_total","population_male","population_female",
            "pop_growth_rate","sex_ratio","regional_dev_index","population_pct"]
# Ajout de la colonne iso3 dans l'agrégation finale
df_agg = df.groupby(["region", "year"]).agg(
    {**{col: "mean" for col in agg_cols}, "iso3": "first"}
).reset_index()
print("✅ Agrégation effectuée :", df_agg.shape)

# =======================
# 7️⃣ Graphiques exploratoires
# =======================
# Population mondiale totale
plt.figure(figsize=(10,5))
df.groupby("year")["population_total"].sum().plot()
plt.title("Évolution de la population mondiale (1950-2023)")
plt.ylabel("Population (milliers)")
plt.savefig(f"{graphs_dir}/population_mondiale.png")
plt.close()

# Corrélations
plt.figure(figsize=(8,6))
sns.heatmap(df_agg[["population_total","pop_growth_rate","sex_ratio","regional_dev_index"]].corr(),
            annot=True, cmap="coolwarm")
plt.title("Corrélations principales")
plt.savefig(f"{graphs_dir}/correlation.png")
plt.close()

# =======================
# 8️⃣ Export dataset enrichi
# =======================
df_agg.to_csv(output_file, index=False)
print(f"💾 Dataset enrichi sauvegardé : {output_file}")