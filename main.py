from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import io
import traceback
import os

app = FastAPI()

# Ensure static and templates directories exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Load model at startup and log success/failure
try:
    model_path = os.path.join(BASE_DIR, "face_shape_model.keras")
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print("Failed to load model:", e)
    model = None

class_names = ["oval", "rectangular", "round", "square"]


def preprocess_image(file_contents):
    img = image.load_img(io.BytesIO(file_contents), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "class_names": class_names}
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400, content={"error": "Only image files are allowed"}
        )

    if model is None:
        return JSONResponse(
            status_code=500, content={"error": "Model not loaded on server."}
        )

    try:
        contents = await file.read()
        processed_img = preprocess_image(contents)

        predictions = model.predict(processed_img)
        pred_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        return {
            "predicted_class": class_names[pred_class],
            "confidence": confidence,
            "all_predictions": {
                class_names[i]: float(predictions[0][i])
                for i in range(len(class_names))
            },
        }

    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500, content={"error": f"An error occurred: {str(e)}"}
        )
