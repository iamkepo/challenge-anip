# Tâche 3 - Visualisation & Insights (Power BI)
# Auteur : Christ-Amour Kakpo
# Objectif : Dataset enrichi + cartes choroplèthes et graphiques narratifs

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =======================
# 1️⃣ Chemins
# =======================
input_file = "output/WPP2024_DEMOGRAPHIC_ENRICHED.csv"
output_dir = "output/powerbi_full"
graphs_dir = os.path.join(output_dir, "graphs")
maps_dir = os.path.join(output_dir, "maps")
os.makedirs(graphs_dir, exist_ok=True)
os.makedirs(maps_dir, exist_ok=True)

# =======================
# 2️⃣ Charger dataset
# =======================
df = pd.read_csv(input_file)
print("✅ Dataset chargé :", df.shape)

# =======================
# 3️⃣ Nettoyage / KPI
# =======================
df.fillna({
    "pop_growth_rate": 0,
    "sex_ratio": 1,
    "regional_dev_index": 0
}, inplace=True)

# Population relative mondiale
df['world_population'] = df.groupby('year')['population_total'].transform('sum')
df['population_pct'] = df['population_total'] / df['world_population'] * 100

# Catégorisation population pour filtres
def categorize_population(pop):
    if pop < 1_000: return "<1M"
    elif pop < 10_000: return "1M-10M"
    elif pop < 50_000: return "10M-50M"
    elif pop < 100_000: return "50M-100M"
    else: return ">100M"
df['pop_category'] = df['population_total'].apply(categorize_population)

# =======================
# 4️⃣ Agrégation multi-dimensionnelle
# =======================
agg_cols = ['population_total', 'pop_growth_rate', 'sex_ratio', 'regional_dev_index', 'population_pct']
df_agg = df.groupby(['region', 'year'])[agg_cols].mean().reset_index()

# Export CSV Power BI
powerbi_csv = os.path.join(output_dir, "WPP2024_POWERBI_READY.csv")
df_agg.to_csv(powerbi_csv, index=False)
print(f"💾 Dataset Power BI sauvegardé : {powerbi_csv}")

# =======================
# 5️⃣ Graphiques narratifs
# =======================
# Population mondiale
plt.figure(figsize=(10,5))
df.groupby("year")["population_total"].sum().plot()
plt.title("Évolution population mondiale (1950-2023)")
plt.ylabel("Population (milliers)")
plt.savefig(f"{graphs_dir}/population_mondiale.png")
plt.close()

# Indice de développement régional moyen
plt.figure(figsize=(12,6))
sns.lineplot(data=df_agg, x='year', y='regional_dev_index', hue='region', legend=False)
plt.title("Indice de développement régional moyen (1950-2023)")
plt.ylabel("Indice composite")
plt.savefig(f"{graphs_dir}/regional_dev_index.png")
plt.close()

# Heatmap corrélations
corr_cols = ['population_total','pop_growth_rate','sex_ratio','regional_dev_index']
plt.figure(figsize=(8,6))
sns.heatmap(df_agg[corr_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Corrélations principales")
plt.savefig(f"{graphs_dir}/correlation_heatmap.png")
plt.close()

print(f"✅ Graphiques narratifs sauvegardés dans : {graphs_dir}")

# =======================
# 6️⃣ Cartes choroplèthes interactives
# =======================
# Vérifier colonne ISO3
if 'iso3' not in df.columns:
    raise ValueError("La colonne 'iso3' est requise pour les cartes.")

df_2023 = df[df['year'] == 2023]

# Carte population 2023
fig_pop = px.choropleth(
    df_2023,
    locations="iso3",
    color="population_total",
    hover_name="region",
    hover_data=["population_total", "pop_growth_rate", "regional_dev_index"],
    color_continuous_scale="Viridis",
    title="Population totale par pays (2023)"
)
fig_pop.write_html(os.path.join(maps_dir, "population_2023.html"))
fig_pop.write_image(os.path.join(maps_dir, "population_2023.png"))

# Carte indice de développement régional 2023
fig_index = px.choropleth(
    df_2023,
    locations="iso3",
    color="regional_dev_index",
    hover_name="region",
    hover_data=["population_total","pop_growth_rate","sex_ratio"],
    color_continuous_scale="Plasma",
    title="Indice de développement régional (2023)"
)
fig_index.write_html(os.path.join(maps_dir, "regional_dev_index_2023.html"))
fig_index.write_image(os.path.join(maps_dir, "regional_dev_index_2023.png"))

# Carte animée population 1950-2023
fig_anim = px.choropleth(
    df,
    locations="iso3",
    color="population_total",
    hover_name="region",
    animation_frame="year",
    color_continuous_scale="Viridis",
    title="Évolution population totale par pays (1950-2023)"
)
fig_anim.write_html(os.path.join(maps_dir, "population_1950_2023.html"))

print(f"✅ Cartes choroplèthes sauvegardées dans : {maps_dir}")