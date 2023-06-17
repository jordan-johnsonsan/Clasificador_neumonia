# Dependencias
import io
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from fastapi import FastAPI, UploadFile

app = FastAPI()
# http://127.0.0.1:8000

# Cargar el modelo
model = load_model('pneumonia_trained.h5')


@app.post("/prediccion")
async def obtener_prediccion(file: UploadFile):
    # Leemos la imagen del cliente
    image = Image.open(io.BytesIO(await file.read())).convert("L")

    # Preparamos la imagen
    # Ajustar al tamaño esperado por el modelo
    image = image.resize((150, 150))
    image = img_to_array(image)  # Convertir la imagen a array
    image = image / 255.0  # Normalizar la imagen
    image = np.expand_dims(image, axis=0)  # Agregar una dimensión adicional

    # Obtenemos la predicción del modelo
    prediccion = model.predict(image)
    # Obtener la probabilidad de la clase predicha
    probabilidad = prediccion[0][0] * 100

    prediccion_redondeada = np.round(prediccion).item()
    diagnostico = 'normal' if prediccion_redondeada == 0 else 'neumonía'

    return {
        "Diagnostico": diagnostico,
        "Confianza": probabilidad
    }
