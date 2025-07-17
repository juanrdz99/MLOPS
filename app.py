# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import joblib
import numpy as np

# Inicializar API
app = FastAPI(title='Práctica MLOps + FastAPI')

# Cargar modelo y scaler entrenados desde archivos pickle
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('standard_scaler.pkl')

# Pipelines Hugging Face con modelos explícitos
sentiment_pipe = pipeline(
    'sentiment-analysis',
    model='distilbert/distilbert-base-uncased-finetuned-sst-2-english',
    tokenizer='distilbert/distilbert-base-uncased-finetuned-sst-2-english'
)
summarization_pipe = pipeline(
    'summarization',
    model='sshleifer/distilbart-cnn-12-6',
    tokenizer='sshleifer/distilbart-cnn-12-6'
)

# Schemas de request
class CancerRequest(BaseModel):
    features: list[float]  # Lista de 30 floats correspondientes a las características del dataset

class TextRequest(BaseModel):
    text: str

# Endpoints
@app.get('/')
async def health_check():
    '''Health check: confirma que la API está activa.'''
    return {'status': 'ok', 'message': 'API MLOps + FastAPI funcionando'}

@app.get('/items/{id}')
async def read_item(id: int, q: str = None):
    '''Devuelve un item con parámetro opcional.'''
    return {'id': id, 'q': q}

@app.get('/add')
async def add(a: float, b: float):
    '''Suma dos números pasados como query params.'''
    return {'a': a, 'b': b, 'result': a + b}

@app.post('/predict')
async def predict(req: CancerRequest):
    '''Recibe 30 características y devuelve predicción y probabilidad.'''
    if len(req.features) != 30:
        raise HTTPException(status_code=400, detail="Se requieren 30 valores en 'features'")
    x = np.array(req.features).reshape(1, -1)
    x_scaled = scaler.transform(x)
    pred = int(model.predict(x_scaled)[0])
    proba = float(model.predict_proba(x_scaled)[0][pred])
    return {'prediction': pred, 'probability': proba}

@app.post('/sentiment')
async def sentiment_analysis(req: TextRequest):
    '''Analiza sentimiento de un texto.'''
    res = sentiment_pipe(req.text)
    return {'operation': 'sentiment-analysis', 'input': req.text, 'output': res}

@app.post('/summarize')
async def summarize(req: TextRequest):
    '''Genera resumen de un texto.'''
    summaries = summarization_pipe(req.text, max_length=50, min_length=10, do_sample=False)
    return {'operation': 'summarization', 'input': req.text, 'output': summaries}