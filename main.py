import re
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.text import Tokenizer
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from rich import print

model_path = r"Artifacts\BiGRU_Modle.keras"
tokenizer_path = r"Artifacts\tokenizer.pkl"
max_sequence_len = 50
emotion_labels = ["joy", "sadness", "anger", "fear", "love", "surprise"]

EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}

def preprocess_text(text: str)->str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class TextInput(BaseModel):
    text : str = Field(
        ..., 
        min_length=1, 
        max_length=2000, 
        description="The sentence to analyze",
        json_schema_extra={"example" : "I fell so happy and excited"}
        )

class PredictionResponse(BaseModel):
    text : str
    predicted_emotion : str
    confidence : float
    all_probabilities : dict[str, float]

class HealthResponse(BaseModel):
    status : str
    model_loaded : bool


dl_model = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Loading the model and tokenizer....')
    dl_model["BiGRU"] = load_model(model_path)
    with open(tokenizer_path, "rb") as file:
        dl_model["Tokenizer"] = pickle.load(file)
    print("Models are loaded successfully......")
    yield
    dl_model.clear()

app = FastAPI(
    title="Deep Learning Emotion Classifier",
    description="Emotion classification using a BiGRU model",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"]
)

app.mount(
    "/static", 
    StaticFiles(directory="static"),
    name="static"
    )

@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse('./static/index.html')

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="Server is running",
        model_loaded=bool(dl_model)
    )

@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):
    BIGRU_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("Tokenizer")

    if BIGRU_model is None or tokenizer_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Please try again later"
        )

    cleaned_text = preprocess_text(text_input.text)
    tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=max_sequence_len,
        padding="post",
        truncating="post"
    )

    probabilities = BIGRU_model.predict(padded_sequence)[0]
    top_emotion_index = int(np.argmax(probabilities))
    all_probabilities = {label : float(prob) for label,prob in zip(emotion_labels, probabilities)}
    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=emotion_labels[top_emotion_index],
        confidence=float(probabilities[top_emotion_index]),
        all_probabilities=all_probabilities
    )
