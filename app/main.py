from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import cv2
import os

# Inicializar FastAPI
app = FastAPI()

# Habilitar CORS para el frontend (Angular)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia esto en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo y scaler
modelo = joblib.load("app/model_emociones.pkl")
scaler = joblib.load("app/scaler.pkl")

# Etiquetas del dataset FER2013 (en orden)
emociones = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Ruta de prueba
@app.get("/")
def raiz():
    return {"mensaje": "API de clasificación de emociones - SVM"}

# Endpoint para predecir emoción desde imagen
@app.post("/emocion")
async def detectar_emocion(file: UploadFile = File(...)):
    # Guardar imagen temporal
    contents = await file.read()
    with open("temp.jpg", "wb") as f:
        f.write(contents)

    # Leer imagen y convertir a escala de grises
    img = cv2.imread("temp.jpg", cv2.IMREAD_GRAYSCALE)

    # Detectar rostros
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    rostros = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

    if len(rostros) == 0:
        return {"mensaje": "No se detectaron rostros"}

    predicciones = []
    for (x, y, w, h) in rostros:
        rostro = img[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (48, 48)).flatten() / 255.0
        norm = scaler.transform([rostro])
        pred = modelo.predict(norm)[0]
        predicciones.append(emociones[pred])

    return {"emociones": predicciones}
