# Importation des librairies

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# API
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

# Importation du dataset
data = pd.read_csv('creditcard.csv')

# Exploration du dataset
# Aperçu des données

data.head()

# data.info()

data.describe()

# Vérification des valeurs manquantes

data.isnull().sum()

# Vérification des valeurs que peut prendre la variable "Class"


data['Class'].unique()
'''
Ayant dans notre cas une variable nommée **Class** qui permet de savoir s'il y avait fraude ou non, alors tout au long du présent travail, je vais faire un **apprentissage supervisé**

# Prétraitement des données

### Normalisation des variables "Time" et "Amount"
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

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42)

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

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
 

@app.post("/predict/")
async def predict(transaction: Transaction):
    try:
        # Exclure la variable 'Class' et convertir les valeurs en float
        features = np.array([value for key, value in transaction.dict().items()]).reshape(1, -1)
        
        # Prédiction
        prediction = model.predict(features)
        
        # Résultat
        return {"fraudulent": bool(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")
