import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os

# Cargar CSV
df = pd.read_csv("train/fer2013_publictest_onehot.csv")

# Detectar columnas de emociones (últimas 7 columnas)
emotion_cols = df.columns[-7:]

# Generar etiqueta numérica a partir de la posición del 1 en las columnas one-hot
df["emotion"] = df[emotion_cols].idxmax(axis=1)
df["emotion"] = df["emotion"].apply(lambda x: list(emotion_cols).index(x))

# Separar variables X y etiquetas y
X = df.drop(columns=emotion_cols.tolist() + ["emotion"]).values
y = df["emotion"].values

# Normalizar y escalar
X = X / 255.0
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entrenar
modelo = SVC(kernel='linear')
modelo.fit(X_train, y_train)

# Evaluar
print(classification_report(y_test, modelo.predict(X_test)))

# Guardar modelo y scaler
os.makedirs("app", exist_ok=True)
joblib.dump(modelo, "app/model_emociones.pkl")
joblib.dump(scaler, "app/scaler.pkl")
