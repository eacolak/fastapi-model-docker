from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI()

# Modelin yüklenmesi
model = tf.keras.models.load_model("model/Trafic_signs_model.h5")

# Toplam class sayısı
NUM_CLASSES = 43

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Görüntünün okunması
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((30,30))
    
    # Ön hazırlık
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    # Tahmin yapılma kısmı
    predictions = model.predict(img_array)
    predicted_class = int(np.argmax(predictions, axis=1)[0])
    confidence = float(np.max(predictions))

    # Json olarak döndürme
    return JSONResponse({
        "predicted_class": predicted_class,
        "confidence": confidence,
    })    