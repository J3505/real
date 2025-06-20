import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detectar_rostros(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    rostros = face_cascade.detectMultiScale(img, 1.1, 5)
    recortes = []
    for (x, y, w, h) in rostros:
        rostro = cv2.resize(img[y:y+h, x:x+w], (48, 48)).flatten()
        recortes.append(rostro)
    return recortes
