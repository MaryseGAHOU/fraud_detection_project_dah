'''
# -*- coding: utf-8 -*-
"""Développer un modèle pour détecter les transactions frauduleuses.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Zx57FZc900l-AjO3LWa3XXvOa4LztZyz

<style>
  body {
    font-family: 'Arial', sans-serif;
    background-color: #f4f7f6;
    color: #333;
    line-height: 1.6;
  }
  .container {
    margin: 0 auto;
    max-width: 900px;
    background: #fff;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
  }
  h1, h2, h3 {
    color: #3498db;
  }
  p {
    margin: 10px 0;
  }
  .section {
    margin-bottom: 40px;
  }
  .section-title {
    font-size: 1.8em;
    margin-bottom: 10px;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
  }
  .code-box {
    background: #f7f7f7;
    padding: 15px;
    border-radius: 5px;
    border-left: 4px solid #3498db;
    margin: 10px 0;
    font-family: 'Courier New', monospace;
    overflow-x: auto;
  }
  .image-box {
    text-align: center;
  }
  .image-box img {
    width: 80%;
    border-radius: 8px;
    margin-top: 20px;
  }
</style>

<div style="display: flex; align-items: center;">
    <img src="logo_DataAfriqueHub.jpg" alt="Logo de DAta Afrique Hub" style="width: 200px; height: auto;">
    <h1 style="margin-left: 10px;">Détection des Fraudes Bancaires par GAHOU Maryse</h1>
</div>

<h1>Contexte</h1>

La détection des fraudes est cruciale pour les institutions financières afin de protéger les clients et minimiser les pertes financières.
</br>

<h1>Problème</h1>

<p>Développer un modèle pour détecter les transactions frauduleuses et déployer une application qui permet de soumettre des transactions pour évaluation.</p>

# Importation des librairies
"""
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from fastapi import FastAPI

"""# Importation du dataset"""

data = pd.read_csv('creditcard.csv')

"""# Exploration du dataset

### Aperçu des données
"""

data.head()

#data.info()

data.describe()

"""### Vérification des valeurs manquantes"""

data.isnull().sum()

"""### Vérification des valeurs que peut prendre la variable "Class"
"""

data['Class'].unique()
'''
"""<h1>Méthodologie utilisée</h1>
Dans le cadre d'une détection de fraudes, on a la possibilité de faire :
<ul>
    <li>Un apprentissage supervisé</li>
    <li>Un apprentissage non supervisé</li>
    <li>Un apprentissage semi-supervisé</li>
</ul>

Ayant dans notre cas une variable nommée **Class** qui permet de savoir s'il y avait fraude ou non, alors tout au long du présent travail, je vais faire un **apprentissage supervisé**

# Prétraitement des données

### Normalisation des variables "Time" et "Amount"
"""
'''

scaler = StandardScaler()

data['Time'] = scaler.fit_transform(data[['Time']])

data['Amount'] = scaler.fit_transform(data[['Amount']])

"""### Gestion des classes déséquilibrées notamment la variable "Class"
"""

data['Class'].value_counts()

"""Le dataset est alors déséquilibré, avec beaucoup plus de transactions légitimes que de fraudes. Je vais alors utiliser des techniques comme le suréchantillonnage (SMOTE) ou le sous-échantillonnage pour gérer ce déséquilibre."""

# Séparation du dataset des features de la variable cible
X = data.drop('Class', axis=1)
y = data['Class']

# Application du suréchantillonnage SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

"""# Entraînement d'un modèle de détection de fraudes notamment le Random Forest

### Séparation des données en ensemble d'entraînement et de test
"""

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

"""### Entraînement du modèle Random Forest"""

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

"""### Prédiction avec le modèle Random Forest"""

y_pred = model.predict(X_test)

"""# Évaluer la performance du modèle"""

print(classification_report(y_test, y_pred))

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

"""# Création d'une application web pour permettre la détection des fraudes"""

app = FastAPI()

@app.post("/predict")
def predict(transaction: dict):
    # Prétraiter les données soumises
    features = np.array([transaction[key] for key in transaction.keys()]).reshape(1, -1)

    # Prédiction
    prediction = model.predict(features)

    # Résultat
    return {"fraudulent": bool(prediction[0])}
'''
"""<div style="margin-bottom: 20px;">
    <h2 style="text-align: center; font-family: 'Arial', sans-serif; color: blue; font-size: 24px;">
        Merci pour votre aimable attention
    </h2>
    <p style="text-align: center; font-family: 'Arial', sans-serif; color: orange; font-size: 18px;">
        Mais maintenant, place à la démo de prédiction
    </p>
</div>
"""
'''
