"""
Script d'entraînement — Bank Marketing Dataset
Objectif : prédire si un client va souscrire à un dépôt à terme (y = yes/no)
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

os.makedirs("artifacts", exist_ok=True)

# ── 1. Chargement ─────────────────────────────────────────
print("Chargement du dataset Bank Marketing...")
df = pd.read_csv("data/bank.csv")
print(f"  {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"  Distribution cible : {df['y'].value_counts().to_dict()}")

# ── 2. Prétraitement ──────────────────────────────────────
# Séparation features / cible
X = df.drop(columns=["y"])
y = (df["y"] == "yes").astype(int)   # 1 = souscription, 0 = pas de souscription

# Colonnes catégorielles vs numériques
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

print(f"\nFeatures numériques  : {num_cols}")
print(f"Features catégorielles : {cat_cols}")

# Encodage label des catégorielles
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Normalisation des numériques
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Split stratifié (déséquilibre ~26% yes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain : {len(X_train)} | Test : {len(X_test)}")

# ── 3. Entraînement ───────────────────────────────────────
print("\nEntraînement du modèle GradientBoostingClassifier...")
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    random_state=42,
)
model.fit(X_train, y_train)

# ── 4. Évaluation ─────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_proba)

print("\n=== Métriques ===")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=["no", "yes"]))

# Top features
feat_imp = pd.Series(model.feature_importances_, index=X.columns)
print("Top 5 features :")
print(feat_imp.sort_values(ascending=False).head(5).to_string())

# ── 5. Sauvegarde des artefacts ───────────────────────────
artifact = {
    "model":       model,
    "scaler":      scaler,
    "encoders":    encoders,
    "cat_cols":    cat_cols,
    "num_cols":    num_cols,
    "feature_names": list(X.columns),
}
with open("artifacts/model.pkl", "wb") as f:
    pickle.dump(artifact, f)

with open("artifacts/metrics.txt", "w") as f:
    f.write(f"accuracy={accuracy:.4f}\n")
    f.write(f"precision={precision:.4f}\n")
    f.write(f"recall={recall:.4f}\n")
    f.write(f"f1_score={f1:.4f}\n")
    f.write(f"roc_auc={roc_auc:.4f}\n")

print("\nArtefacts sauvegardés : artifacts/model.pkl, artifacts/metrics.txt")
