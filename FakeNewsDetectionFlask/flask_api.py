from flask import Flask, request, jsonify
import numpy as np
import re
import spacy
import nltk
import tensorflow as tf
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import pickle

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

# Tokenizer ve Model yükleme
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
model = load_model("rnn_model.h5")

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

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    processed = preprocess(data["text"])
    prediction = model.predict(processed)[0][0]
    result = "Fake" if prediction >= 0.5 else "Real"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
    