# Tâche 1 - Nettoyage & Préparation (Pandas)
# Auteur : Christ-Amour Kakpo
# Objectif : Nettoyer et préparer le dataset démographique WPP2024 pour analyse ultérieure

import pandas as pd
import os

# 1️⃣ Chemins
file_path = "data/WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_FULL.xlsx"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 2️⃣ Lire la feuille 'Estimates' en sautant les lignes d'en-tête inutiles
df = pd.read_excel(file_path, sheet_name="Estimates", header=16)  # La ligne 16 contient les bons noms de colonnes

print("Colonnes détectées :", df.columns.tolist())
print("Dimensions :", df.shape)

# 3️⃣ Sélectionner et renommer les colonnes essentielles
col_mapping = {
    "Location": "country",
    "ISO3 Alpha-code": "iso3",
    "Region, subregion, country or area *": "region",
    "Year": "year",
    "Total Population, as of 1 July (thousands)": "population_total",
    "Female Population, as of 1 July (thousands)": "population_female",
    "Male Population, as of 1 July (thousands)": "population_male",
    "Life Expectancy at Birth, both sexes (years)": "life_expectancy",
    "Total Fertility Rate (live births per woman)": "fertility_rate",
    "Crude Birth Rate (births per 1,000 population)": "birth_rate",
    "Crude Death Rate (deaths per 1,000 population)": "death_rate"
}

# Garder uniquement les colonnes existantes dans le fichier
existing_cols = [col for col in col_mapping.keys() if col in df.columns]
df_clean = df[existing_cols].copy()
df_clean.rename(columns={k: col_mapping[k] for k in existing_cols}, inplace=True)

# 4️⃣ Conversion des types
if "year" in df_clean.columns:
    df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce").astype("Int64")

for col in ["population_total", "population_female", "population_male",
            "life_expectancy", "fertility_rate", "birth_rate", "death_rate"]:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

# 5️⃣ Supprimer les lignes sans population
if "population_total" in df_clean.columns:
    df_clean = df_clean[df_clean["population_total"].notna()]

print("Dimensions après nettoyage :", df_clean.shape)

# 6️⃣ Sauvegarder le dataset final
output_csv = os.path.join(output_dir, "WPP2024_DEMOGRAPHIC_CLEAN.csv")
df_clean.to_csv(output_csv, index=False)
print(f"Dataset nettoyé sauvegardé : {output_csv}")

# 7️⃣ Générer un glossaire simple
glossary = pd.DataFrame({
    "Variable": df_clean.columns,
    "Description": [
        "Nom du pays",
        "Code ISO3 du pays",
        "Région / sous-région",
        "Année",
        "Population totale (en milliers)",
        "Population féminine (en milliers)",
        "Population masculine (en milliers)",
        "Espérance de vie à la naissance (années)",
        "Taux de fécondité total (enfants par femme)",
        "Taux brut de natalité (pour 1000 habitants)",
        "Taux brut de mortalité (pour 1000 habitants)"
    ][:len(df_clean.columns)],
    "Unité": [
        "-", "-", "-", "année", "milliers de personnes", "milliers de personnes", "milliers de personnes",
        "années", "enfants par femme", "pour 1000 habitants", "pour 1000 habitants"
    ][:len(df_clean.columns)],
    "Source": ["UN WPP 2024"]*len(df_clean.columns),
    "Période": ["1950-2023"]*len(df_clean.columns),
    "Géographie": ["Pays / région / sous-région"]*len(df_clean.columns)
})

glossary_csv = os.path.join(output_dir, "WPP2024_DEMOGRAPHIC_GLOSSARY.csv")
glossary.to_csv(glossary_csv, index=False)
print(f"Glossaire sauvegardé : {glossary_csv}")