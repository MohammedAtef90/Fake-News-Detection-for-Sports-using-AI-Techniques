import streamlit as st
import re
import string
import contractions
import numpy as np
import tensorflow as tf
import requests
import json
import time
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

nltk.download('stopwords')

# ------------------ API keys ------------------ #
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GOOGLE_CSE_API_KEY = st.secrets["GOOGLE_CSE_API_KEY"]
GOOGLE_CSE_CX = st.secrets["GOOGLE_CSE_CX"]
MAGE_PIPELINE_TRIGGER_URL_STREAMLIT = st.secrets["MAGE_PIPELINE_TRIGGER_URL_STREAMLIT"]

st.set_page_config(page_title="Football Transfer Fake News Detector", page_icon="⚽")

# ------------------ Clean text ------------------ #
stopword = set(stopwords.words('english')) - {"not", "won"}
stopword.update(string.punctuation, {'“','’','”','‘','...'})

def clean(text):
    text = contractions.fix(text)
    text = re.sub(r'<.*?>|\[.*?\]|\n', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.replace('$',' dollar ').replace('€',' euro ').replace('£',' pound ')
    text = re.sub(r'[^a-zA-Z0-9\s\'-]', '', text)
    text = re.sub(r'\s+',' ', text).strip().lower()

    words = text.split()
    words = [w for w in words if w not in stopword]
    return " ".join(words)

# ------------------ Load BERT once only ------------------ #
@st.cache_resource
def load_model():
    checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    return tokenizer, model

tokenizer, model = load_model()

# ------------------ BERT prediction ------------------ #
def predict_bert(texts):
    max_len = 64
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="tf")
    outputs = model(inputs)
    probs = tf.nn.softmax(outputs.logits, axis=1).numpy()
    return probs

# ------------------ Google Search ------------------ #
def perform_google_cse_search(query):
    domains = [
        "bbc.com","skysports.com","espn.com","theathletic.com","goal.com",
        "transfermarkt.com","marca.com","sport.es","bild.de","lequipe.fr",
        "gazzetta.it","reuters.com","apnews.com"
    ]
    site_filters = " OR ".join([f"site:{d}" for d in domains])
    full_query = f"{query} {site_filters}"

    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": GOOGLE_CSE_API_KEY, "cx": GOOGLE_CSE_CX, "q": full_query, "num": 5}

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        if 'items' not in data:
            return []

        out = []
        for i, item in enumerate(data['items']):
            out.append(f"{i+1}. {item.get('title')} - {item.get('link')}")
        return out
    except:
        return []

# ------------------ Gemini ------------------ #
def check_with_llm(text):
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{
            "parts": [{
                "text": f"Check if this football transfer news is real or fake.\nLabel: Real or Fake\nReason: short.\nNews: {cleaned}"
            }]
        }]
    }
    headers = {"Content-Type":"application/json"}

    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
        r.raise_for_status()
        txt = r.json()['candidates'][0]['content']['parts'][0]['text']

        lines = txt.split("\n")
        label = next((l for l in lines if "Label" in l), "Label: Unknown")
        reason = next((l for l in lines if "Reason" in l), "Reason: Unknown")
    except:
        label = "Label: Unknown"
        reason = "Reason: API error"

    sources = perform_google_cse_search(text)

    out = [label, reason]
    out.extend(sources if sources else ["No trusted sources found."])
    return out

# ------------------ Main prediction pipeline ------------------ #
def pipeline(lines):
    cleaned = [clean(t) for t in lines]

    start = time.time()
    preds = predict_bert(cleaned)
    infer_time = time.time() - start

    results = []
    for i, original in enumerate(lines):
        fake_prob = preds[i][1] * 100
        llm = check_with_llm(original)

        results.append({
            "headline": original,
            "fake_prob": fake_prob,
            "analysis": llm,
            "time": infer_time / len(lines)
        })

    return results

# ------------------ Streamlit UI ------------------ #
st.title("⚽ Football Transfer Fake News Detector")

user_input = st.text_area(
    "✍️ Enter football transfer news (one per line):",
    height=150,
    value="Manchester United sign Jadon Sancho from Borussia Dortmund for £73 million"
)

if st.button("🔎 Predict"):
    if not user_input.strip():
        st.error("Please enter news!")
        st.stop()

    lines = [x.strip() for x in user_input.split("\n") if x.strip()]

    with st.spinner("Analyzing..."):
        results = pipeline(lines)

        for i, r in enumerate(results):
            st.subheader(f"📰 News {i+1}: {r['headline']}")
            st.write(f"Fake Probability: {r['fake_prob']:.2f}%")
            st.write(f"Model Inference Time: {r['time']:.3f} seconds")

            for row in r['analysis']:
                st.markdown(row)

            st.markdown("---")
