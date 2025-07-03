import streamlit as st
import re
import string
import contractions
import spacy
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

st.set_page_config(page_title="Football Transfer Fake News Detector", page_icon="⚽")

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

def clean(text):
    text = contractions.fix(text)
    text = re.sub(r'<.*?>|\[.*?\]|\n', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.replace('$', ' dollar ').replace('€', ' euro ').replace('£', ' pound ')
    text = re.sub(r'[^a-zA-Z0-9\s\'-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()

    doc = nlp(text)
    tokens = [token.lemma_ if token.pos_ != "PROPN" else token.text for token in doc]
    return " ".join(tokens)

@st.cache_resource
def load_classification_model():
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    return tokenizer, model

def predict_bert(texts, tokenizer, model, max_len):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="tf")
    outputs = model(inputs)
    probs = tf.nn.softmax(outputs.logits, axis=1).numpy()
    return probs

def run_prediction_pipeline(headlines, tokenizer, model):
    cleaned = [clean(h) for h in headlines]
    max_len = max(len(tokenizer(h)['input_ids']) for h in cleaned)

    preds = predict_bert(cleaned, tokenizer, model, max_len)
    results = []
    for i, orig in enumerate(headlines):
        fake_prob = preds[i][1] * 100
        results.append({
            "original_headline": orig,
            "fake_probability": fake_prob
        })
    return results

tokenizer, model = load_classification_model()

st.title("⚽ Football Transfer Fake News Detector")

user_input = st.text_area(
    "✍️ Enter football transfer news (one per line):",
    height=150,
    value="Manchester United sign Jadon Sancho from Borussia Dortmund for £73 million"
)

if st.button("🔎 Predict"):

    if not user_input.strip():
        st.error("Please enter news headlines.")
        st.stop()

    headlines = [line.strip() for line in user_input.split("\n") if line.strip()]
    with st.spinner("Analyzing..."):
        results = run_prediction_pipeline(headlines, tokenizer, model)

    if results:
        for i, res in enumerate(results):
            st.subheader(f"📰 News {i+1}: {res['original_headline']}")
            st.write(f"**Fake Probability:** {res['fake_probability']:.2f}%")
            st.markdown("---")
    else:
        st.info("No valid news headlines to process.")
