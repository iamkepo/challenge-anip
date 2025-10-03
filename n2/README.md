# 🧑‍💻 Challenge ANIP — Reconnaissance faciale, Estimation d’âge et OCR fraude

## 📌 Objectifs
Ce challenge comporte **3 tâches principales** :

1. **Reconnaissance faciale robuste**  
   Identifier une personne sur des photos prises à différents moments de sa vie.  

2. **Estimation de l’âge à partir d’une photo**  
   Prédire l’âge d’une personne à partir d’une image.  

3. **OCR & Détection de fraude sur documents d’identité**  
   Extraire les zones de texte d’un document officiel et déterminer si celui-ci est **authentique** ou **falsifié**.  

---

## 📂 Organisation des données
Les datasets sont fournis sous `data/` :

data/
├─ dataset_tache_1/
│   ├─ train/   (images XXXX_0.jpg, XXXX_1.jpg)
│   └─ test/
├─ dataset_tache_2/
│   ├─ train/   (images XXXXXX_YZWW.jpg)
│   └─ test/
└─ dataset_tache_3/
├─ train/
│   ├─ esp/{normal,forgery_1..4,gt}
│   ├─ est/{…}
│   ├─ rus/{…}
│   └─ arizona_dl/{…}
└─ test/
├─ esp/
├─ est/
├─ rus/
└─ arizona_dl/

Les résultats et checkpoints sont stockés dans `outputs/`.

---

## ⚡ Installation

### 1. Créer un environnement virtuel
```bash
python -m venv .venv
source .venv/bin/activate

2. Installer les dépendances

pip install torch torchvision timm pandas scikit-learn faiss-cpu pillow tqdm easyocr

(Remplacer faiss-cpu par faiss-gpu si GPU disponible.)

⸻

🚀 Utilisation

🟢 Tâche 1 — Reconnaissance faciale

Entraînement + matching :

python Tache_1_Reconnaissance_Faciale.py \
  --mode train \
  --data_dir data/dataset_tache_1/dataset_tache_1 \
  --out_dir outputs/tache1 \
  --epochs 10

Évaluation (génération du CSV) :

python Tache_1_Reconnaissance_Faciale.py \
  --mode eval \
  --data_dir data/dataset_tache_1/dataset_tache_1 \
  --out_dir outputs/tache1 \
  --resume outputs/tache1/checkpoints/ckpt_ep10.pth

➡️ Produit : outputs/tache1/submissions/task1_matching_submission.csv

⸻

🟢 Tâche 2 — Estimation d’âge

Entraînement :

python Tache_2_Estimation_Age.py \
  --mode train \
  --data_dir data/dataset_tache_2/dataset_tache_2 \
  --out_dir outputs/tache2 \
  --epochs 10

Évaluation :

python Tache_2_Estimation_Age.py \
  --mode eval \
  --data_dir data/dataset_tache_2/dataset_tache_2 \
  --out_dir outputs/tache2 \
  --resume outputs/tache2/checkpoints/ckpt_ep10.pth

➡️ Produit : outputs/tache2/submissions/task2_age_submission.csv

⸻

🟢 Tâche 3 — OCR & Détection de fraude

Entraînement :

python Tache_3_OCR_Fraude.py \
  --mode train \
  --data_dir data/dataset_tache_3/dataset_tache_3/train \
  --out_dir outputs/tache3 \
  --epochs 10

Évaluation :

python Tache_3_OCR_Fraude.py \
  --mode eval \
  --data_dir data/dataset_tache_3/dataset_tache_3/test \
  --out_dir outputs/tache3 \
  --resume outputs/tache3/checkpoints/ckpt_ep10.pth

➡️ Produit : outputs/tache3/submissions/task3_ocr_fraud_submission.csv
avec les colonnes :

image,class,ocr_text


⸻

🧠 Notes techniques
	•	Tâche 1 : ResNet50 pour embeddings + Faiss/NN pour matching.
	•	Tâche 2 : ResNet50 avec double tête (classification + régression).
	•	Tâche 3 : ResNet18 pour classification, EasyOCR pour extraction texte.
	•	Tous les scripts utilisent PyTorch et peuvent tourner sur CPU ou GPU.
	•	Pour plus de robustesse, envisager ArcFace (task 1), meilleure discrétisation d’âges (task 2) et features OCR combinées (task 3).

⸻

📜 Auteurs
	•	Scripts développés par Christ-Amour Kakpo
	•	Challenge : ANIP