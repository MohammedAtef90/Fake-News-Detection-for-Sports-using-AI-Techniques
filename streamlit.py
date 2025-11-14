import streamlit as st
import re
import string
import contractions
import spacy
import numpy as np
import tensorflow as tf
import requests
import json
import time
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Download NLTK stopwords
nltk.download('stopwords')

# API keys from secrets
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GOOGLE_CSE_API_KEY = st.secrets["GOOGLE_CSE_API_KEY"]
GOOGLE_CSE_CX = st.secrets["GOOGLE_CSE_CX"]
MAGE_PIPELINE_TRIGGER_URL_STREAMLIT = st.secrets["MAGE_PIPELINE_TRIGGER_URL_STREAMLIT"]

st.set_page_config(page_title="Football Transfer Fake News Detector", page_icon="⚽")

# ------------------ Load Models ------------------ #
# Load spaCy model (without @st.cache_resource to avoid errors)
nlp = spacy.load("en_core_web_sm")

stopword = set(stopwords.words('english')) - {"not", "won"}
stopword.update(string.punctuation, {'“', '’', '”', '‘', '...'})

preserved_entities = set([])

def clean(text):
    text = contractions.fix(text)
    text = re.sub(r'<.*?>|\[.*?\]|\n', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.replace('$', ' dollar ').replace('€', ' euro ').replace('£', ' pound ')
    text = re.sub(r'[^a-zA-Z0-9\s\'-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    doc = nlp(text)
    doc_entities = {ent.text.lower() for ent in doc.ents}
    current_preserved = preserved_entities.union(doc_entities)
    tokens = [
        token.lemma_ if token.text.lower() not in current_preserved and token.pos_ != "PROPN"
        else token.text
        for token in doc
    ]
    return " ".join(tokens)

# Load BERT model (smallest for testing; replace with your trained model)
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

# ------------------ Google Search ------------------ #
def perform_google_cse_search(query, trusted_domains, num_results=5):
    search_results = []
    site_filters = " OR ".join([f"site:{d}" for d in trusted_domains])
    full_query = f"{query} {site_filters}"
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_CSE_API_KEY,
        "cx": GOOGLE_CSE_CX,
        "q": full_query,
        "num": num_results
    }
    try:
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'items' in data:
            for i, item in enumerate(data['items']):
                title = item.get('title')
                link = item.get('link')
                if title and link:
                    search_results.append(f"{i+1}. {title} - {link}")
    except:
        pass
    return search_results

# ------------------ Gemini LLM ------------------ #
def clean_for_gemini(text):
    text = text.replace('$', ' dollar ').replace('€', ' euro ').replace('£', ' pound ')
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def check_with_llm(text):
    llm_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    cleaned_text = clean_for_gemini(text)
    prompt = f"""
You are a professional fact-checking assistant specialized in football transfer news.
Your job is to check whether the following football transfer news is real or fake,
and provide a concise reason. Reply in this exact format:
Label: Real or Fake
Reason: one short explanation (max 2 lines)
News: {cleaned_text}
"""
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    opinion, reason = "Gemini Opinion: Unknown", "Reason: Not provided"

    try:
        res = requests.post(llm_url, headers=headers, data=json.dumps(payload), timeout=30)
        res.raise_for_status()
        result = res.json()
        if 'candidates' in result and result['candidates']:
            lines = result['candidates'][0]['content']['parts'][0]['text'].strip().split("\n")
            for line in lines:
                if line.lower().startswith("label:"):
                    label = line.split(":", 1)[1].strip()
                    if label.lower() in ["real", "fake"]:
                        opinion = "Gemini Opinion: " + label
                elif line.lower().startswith("reason:"):
                    reason = line.strip()
    except:
        opinion, reason = "Gemini Opinion: Could not retrieve", "Reason: Please check API or input formatting"

    sources = perform_google_cse_search(
        text,
        ["bbc.com","skysports.com","espn.com","theathletic.com","goal.com",
         "transfermarkt.com","marca.com","sport.es","bild.de","lequipe.fr",
         "gazzetta.it","reuters.com","apnews.com"]
    )

    output = [opinion, reason]
    if sources:
        output.append("Please check these sources for more information:")
        output.extend(sources)
    else:
        output.append("No reliable sources found via search.")
    output.append("_**Tip:** Try searching the headline on Google + trusted sources._")
    return output

# ------------------ Pipeline ------------------ #
def run_prediction_pipeline(headlines, tokenizer, model):
    try:
        requests.post(MAGE_PIPELINE_TRIGGER_URL_STREAMLIT, timeout=10)
    except:
        pass
    results = []
    valid_headlines = [h.strip() for h in headlines if h.strip()]
    if not valid_headlines:
        return []
    cleaned = [clean(h) for h in valid_headlines]
    max_len = max(len(tokenizer(h)['input_ids']) for h in cleaned)
    start_time = time.time()
    preds = predict_bert(cleaned, tokenizer, model, max_len)
    total_inference_time = time.time() - start_time
    for i, orig in enumerate(valid_headlines):
        fake_prob = preds[i][1] * 100
        llm_output = check_with_llm(orig)
        results.append({
            "original_headline": orig,
            "fake_probability": fake_prob,
            "analysis_output": llm_output,
            "inference_time": total_inference_time / len(valid_headlines)
        })
    return results

# ------------------ Load model ------------------ #
tokenizer, model = load_classification_model()

# ------------------ Streamlit UI ------------------ #
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
                st.write(f"**Model Inference Time:** {res['inference_time']:.3f} seconds")
                for line in res['analysis_output']:
                    st.markdown(line)
                st.markdown("---")
        else:
            st.info("No valid news headlines to process.")
