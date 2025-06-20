# Proyecto: Detector de Emociones

## Estructura del proyecto

```
bs/
├── app/
│   ├── main.py
│   ├── face_detect.py
│   ├── model_emociones.pkl
│   └── scaler.pkl
├── train/
│   ├── train_emotions_model.py
│   └── fer2013_publictest_onehot.csv
├── .env
└── requirements.txt
```

## Requisitos

- Python 3.10.11 (compatible con Python 3.10+)
- FastAPI para exponer una API REST
- OpenCV para detección de rostros
- SVM (Support Vector Machine) para clasificación de emociones

## Configuración del entorno

### 1. Crear y activar entorno virtual

```bash
python -m venv venv
source venv/Scripts/activate  # En Windows
# o
source venv/bin/activate      # En Linux/Mac
python -m pip install --upgrade pip
```

### 2. Instalar dependencias

```bash
pip install fastapi uvicorn scikit-learn pandas numpy opencv-python joblib python-dotenv
pip install -r requirements.txt
python train/train_emotions_model.py
```

> Puedes actualizar el archivo de requerimientos con:
> ```bash
> pip freeze > requirements.txt
> ```

### 3. Ejecutar la aplicación

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

Desarrollado con ❤️ para detectar emociones usando aprendizaje automático.
