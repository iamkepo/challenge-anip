# Rapport synthétique — Challenges 1 & 2 (ANIP)

Date : 3 octobre 2025

Ce document synthétise le travail et les recommandations pour les deux volets du projet ANIP contenus dans ce dépôt :

- Challenge 1 (répertoire `n1/`) — traitement et enrichissement des données démographiques WPP2024.
- Challenge 2 (répertoire `n2/`) — trois tâches d'apprentissage machine sur images : reconnaissance faciale, estimation d'âge, OCR & détection de fraude.

Le rapport présente : objectifs, jeux de données, méthodes utilisées, résultats clés, recommandations opérationnelles et plan d'actions priorisé.

---

## 1. Résumé exécutif

Les deux challenges poursuivent des objectifs complémentaires : assurer la qualité et l'utilisabilité des données démographiques (Challenge 1) et construire des prototypes ML prêts à l'évaluation et à l'intégration (Challenge 2). Les artefacts livrés comprennent jeux de données nettoyés et enrichis, scripts d'analyse et d'entraînement, checkpoints et soumissions CSV. Pour passer du prototype à la production, il est nécessaire d'améliorer la gouvernance des données, l'instrumentation des expériences (métriques/artefacts), et la robustesse des modèles.

Priorités recommandées (ordre) :

1. Mettre en place l'instrumentation et la sauvegarde des métriques pour toutes les tâches (reproductibilité).
2. Formaliser la gouvernance des données et valider les anomalies identifiées dans n1 avec des experts métier.
3. Améliorer les modèles clés (ArcFace/metric learning pour matching, méthodes ordinales/ensembles pour estimation d'âge, pipeline multimodal pour OCR/fraude).
4. Dockeriser et exposer les modèles retenus via APIs d'inférence avec tests de non‑régression.

---

## 2. Détails par challenge

### Challenge 1 — Données démographiques (n1)

Objectif
- Nettoyer le fichier WPP2024, détecter et documenter les anomalies, enrichir avec métadonnées (glossaire, libellés régionaux) et produire des fichiers prêts pour PowerBI.

Données et livrables
- Source : `n1/data/WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_FULL.xlsx`.
- Livrables : `WPP2024_DEMOGRAPHIC_CLEAN.csv`, `WPP2024_DEMOGRAPHIC_ENRICHED.csv`, `WPP2024_DEMOGRAPHIC_ANOMALIES.csv`, `WPP2024_POWERBI_READY.csv` et visualisations dans `n1/output/`.

Méthodologie
- Pipeline ETL : lecture Excel → inspection → harmonisation des noms/pays → traitement des valeurs manquantes (imputation régionale médiane si NA < 20%) → détection d'anomalies (z-score, ruptures temporelles) → enrichissement (indices régionaux, classes d'âge) → export.

Résultats clés
- Jeu nettoyé prêt pour l'analyse et l'intégration PowerBI.
- Anomalies identifiées (espérance de vie > 120, hausses/pertes de population ponctuelles) listées pour validation métier.

Recommandations
- Valider les anomalies avec un comité métier avant toute correction automatique.
- Mettre en place versioning des datasets (hashs) et un glossaire vivant pour garantir traçabilité.

### Challenge 2 — Vision par ordinateur (n2)

Tâches
- Tâche 1 : Reconnaissance faciale — extraire embeddings, matching via Faiss ou NearestNeighbors.
- Tâche 2 : Estimation d'âge — double tête (classification bins + régression) sur backbone ResNet.
- Tâche 3 : OCR & détection de fraude — classification visuelle + extractions texte avec EasyOCR.

Données et livrables
- Dossiers `n2/data/dataset_*` organisés par tâche ; sorties dans `n2/outputs/` (checkpoints, submissions CSV).

Méthodologie et choix techniques
- Backbones standards: ResNet18/50, embeddings L2-normalisés, augmentation légère, sauvegarde checkpoints par epoch.
- Stratégies de prédiction : moyenne d'embeddings par PID (T1), combinaison classification+régression (T2), fusion visuel+OCR (T3 recommandé).

Résultats et état
- Checkpoints présents (T1 jusqu'à ep10, T2 jusqu'à ep6, T3 ep1) ; fichiers de soumission générés.
- Peu (ou pas) de métriques enregistrées dans le dépôt — ajouter `metrics.json` par run.

Recommandations
- Ajouter scripts d'évaluation standard (MAE, R@1/R@5, F1/AUC) et mécanisme d'archivage des métriques et checkpoints.
- Expérimenter ArcFace/metric-learning pour améliorer le matching (T1), essayer ordinal regression pour l'âge (T2), et construire un modèle multimodal texte+image pour la fraude (T3).

---

## 3. Recommandations transverses (opérationnelles)

Gouvernance & conformité
- Mettre en place politiques d'accès aux données (visages, documents) conformes RGPD : chiffrement, anonymisation quand possible, contrôle des accès.

Instrumentation et reproductibilité
- Pour chaque expérience : sauvegarder les hyperparamètres, seed, métriques par epoch (`metrics_epoch_{n}.json`), checkpoint, et env (requirements.txt). Utiliser un storage central (S3/GCS/artifacts repo).

Qualité des modèles
- Mesurer les performances par sous-groupes (âge, sexe, région) pour détecter biais.
- Introduire tests adversariaux simples (perturbations, recadrage, compression) pour valider robustesse.

Déploiement
- Docker + API (FastAPI) ; packager modèle avec TorchScript/ONNX pour latence et compatibilité.
- Metriques de production : latence, throughput, taux d'erreur, drift des données.

Communication aux décideurs
- Dashboard PowerBI (n1) et tableau de bord d’inférence (n2) avec KPI tiles : MAE/accuracy, anomlies critiques, part de population prioritaire, latence.

---

## 4. Plan d'actions proposé (court & moyen terme)

Court terme (0–3 mois)
- A1 : Ajouter scripts d'évaluation et sauvegarde métriques (evaluate_task{1,2,3}.py). (Urgent)
- A2 : Créer un petit comité métier pour valider anomalies n1 et définir règles de correction. (Urgent)
- A3 : Archiver artefacts existants (checkpoints, CSVs, visualisations) dans un stockage central.

Moyen terme (3–9 mois)
- B1 : Expérimentations : ArcFace pour T1, modèles plus lourds/ensembles pour T2, pipeline multimodal pour T3.
- B2 : Dockerisation et API d'inférence, tests automatiques et small-scale rollout.

Long terme (>9 mois)
- C1 : Déploiement en production avec surveillance continue et revue régulière des biais.

---

## 5. KPIs recommandés

- Données : % anomalies validées / trimestre, % lignes imputées.
- Modèles : MAE (âge), R@1/R@5 (matching), F1/AUC (fraude), latence (ms/image).
- Opérationnel : temps moyen pour traiter une anomalie (jours), couverture CI (tests passés).

---

## 6. Livrables proposés immédiatement

1. Scripts d'évaluation de base pour chaque tâche (je peux les ajouter).
2. CI minimal (GitHub Actions) pour exécuter un test d'inférence rapide.
3. Commit des rapports et PDFs (déjà créés) avec message proposé : "Add synthesis reports and PDFs for Challenges 1 & 2".

---

Si vous voulez, j'exécute l'une des actions suivantes maintenant :

- ajouter les scripts d'évaluation de base et lancer un run d'essai sur un petit sous-échantillon ;
- committer les rapports et PDFs (message ci‑dessus) ;
- générer des PDF haute-fidélité (je fournis les commandes d'installation nécessaires si vous préférez exécuter localement).

Indiquez votre préférence et je continue.
