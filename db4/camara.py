import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Cargar el modelo entrenado
model = load_model('modelo_emociones.h5')

# Inicializar la cámara
cap = cv2.VideoCapture(0)  # Puedes cambiar el número si tienes varias cámaras

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocesar el fotograma
    img = cv2.resize(frame, (100, 100))  # Ajusta el tamaño a la entrada del modelo
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Realizar la predicción
    predictions = model.predict(img)
    label = "Feliz" if np.argmax(predictions) == 0 else "Enojado"
    color = (0, 255, 0) if label == "Feliz" else (0, 0, 255)

    # Mostrar el resultado en la ventana de la cámara
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerrar la ventana de la cámara de manera segura
if 'cap' in locals():
    cap.release()
cv2.destroyAllWindows()
