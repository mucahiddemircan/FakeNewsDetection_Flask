from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import re
import spacy
import nltk
import tensorflow as tf
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import pickle

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm") # İngilizce lemmatize

app = FastAPI()

# Tokenizer ve Model yükleme
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
model = load_model("rnn_model.h5")

class NewsItem(BaseModel):
    text: str

# Metin ön işleme
def word_operations(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\t', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = " ".join(text.split())
    text = " ".join([word for word in text.split() if len(word) > 2])
    return text

def remove_stop_words(text):
    return " ".join([word for word in text.split() if word not in stop_words])

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def preprocess(text):
    text = word_operations(text)
    text = remove_stop_words(text)
    text = lemmatize_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=300)
    return padded

@app.post("/predict")
async def predict(item: NewsItem):
    processed = preprocess(item.text)
    prediction = model.predict(processed)[0][0]
    result = "Fake" if prediction >= 0.5 else "Real"
    return {"prediction": result}
