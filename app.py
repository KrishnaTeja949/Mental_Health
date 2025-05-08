import pandas as pd
import numpy as np
import re
import string
import nltk
import joblib
import pytumblr
import plotly.express as px
import time
import streamlit as st
from nltk.corpus import stopwords
from scipy.sparse import hstack

# 📦 Download stopwords only
nltk.download('stopwords')

# 🌙 Dark theme CSS
st.markdown("""
    <style>
    .stTextInput > div > div > input {
        background-color: #262730;
        color: white;
        border: 1px solid #5c5c5c;
        padding: 10px;
        border-radius: 10px;
    }
    .big-font {
        font-size: 28px !important;
        font-weight: 700;
    }
    .emoji {
        font-size: 22px;
        padding-right: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# 🔑 Tumblr API credentials
CONSUMER_KEY = 'bOrfE7FX78FYds3Hwqw3ogX0lttzdMHhcsRR9spa3FuKo3BEzW'
CONSUMER_SECRET = 'fyuzPWnDsqVA0zZxXgMSXf4MCz5S43D2WpIkelPEVXjPbRUd9W'
OAUTH_TOKEN = 'Q1oBnBOkZtC2WMZkPm4z5wys5fnuLo8NZKLzTtN2Ea4DVdPBCY'
OAUTH_TOKEN_SECRET = 'ogEv56paFfs2kbo2bkKh6vC1ntzdV3uPxSkhG1fhtqTbIJZEgr'

# 🔌 Connect to Tumblr
client = pytumblr.TumblrRestClient(
    CONSUMER_KEY,
    CONSUMER_SECRET,
    OAUTH_TOKEN,
    OAUTH_TOKEN_SECRET
)

# 💾 Load model and vectorizer
model_dir = 'MentalHealthModels/'  # Update path accordingly
model = joblib.load(model_dir + 'xgboost_model.pkl')
vectorizer = joblib.load(model_dir + 'tfidf_vectorizer.pkl')
scaler = joblib.load(model_dir + 'scaler.pkl')
label_encoder = joblib.load(model_dir + 'label_encoder.pkl')

# 🧹 Text Preprocessing (No punkt dependency)
def preprocess(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = re.findall(r'\b\w+\b', text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# 🚩 Anorexia keywords
anorexia_keywords = {
    'fat', 'disgusting', 'skip', 'skipped', 'starving', 'guilt',
    'purge', 'thighs', 'ugly', 'calories', 'restrict', 'skinny',
    'fasting', 'binge', 'vomit', 'mirror', 'weight', 'control'
}

def keyword_flag(text):
    words = set(text.lower().split())
    return int(bool(words & anorexia_keywords))

# 📥 Load NRC Emotion Lexicon
emotion_df = pd.read_csv(model_dir + 'NRC-Emotion-Lexicon.csv')
emotion_map = emotion_df.set_index('word').T.to_dict()

# 🎯 Subemotion scoring
def emotion_vector(text):
    tokens = text.lower().split()
    emotion_scores = {emo: 0 for emo in list(emotion_df.columns)[1:]}
    for word in tokens:
        if word in emotion_map:
            for emo in emotion_scores:
                emotion_scores[emo] += emotion_map[word][emo]
    total = sum(emotion_scores.values())
    if total > 0:
        for emo in emotion_scores:
            emotion_scores[emo] = round((emotion_scores[emo] / total) * 100, 2)
    return emotion_scores

# 📊 Plotly bar chart
def plot_subemotions(subemotion_scores):
    filtered_scores = {emo: val for emo, val in subemotion_scores.items() if val > 10}
    if not filtered_scores:
        st.info("No significant subemotions to plot.")
        return

    df = pd.DataFrame({
        "Subemotion": list(filtered_scores.keys()),
        "Influence": list(filtered_scores.values())
    })

    fig = px.bar(df, x='Subemotion', y='Influence',
                 color='Subemotion',
                 title="Subemotion Influence on Prediction (Above 10%)",
                 text='Influence')
    fig.update_layout(
        yaxis_range=[0, 100],
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)

# 🌐 Fetch Tumblr posts
def fetch_tumblr_posts(blogname):
    posts = []
    try:
        response = client.posts(blogname, type='text', limit=50)
        for post in response['posts']:
            if 'body' in post:
                posts.append(post['body'])
            elif 'caption' in post:
                posts.append(post['caption'])
    except Exception as e:
        st.write(f"Error fetching Tumblr posts: {e}")
    return " ".join(posts)

# 🔍 Predict function
def predict(text):
    clean = preprocess(text)
    flag = keyword_flag(clean)
    vec = vectorizer.transform([clean])
    full_feat = hstack([vec, [[flag]]])
    scaled = scaler.transform(full_feat)
    pred = model.predict(scaled)[0]
    pred_label = label_encoder.inverse_transform([pred])[0]

    # Override for anorexia
    if flag == 1:
        pred_label = 'Anorexia'

    subemotion_scores = emotion_vector(clean)

    # Adjust prediction if overly positive
    positive_score_sum = sum([subemotion_scores.get(emo, 0) for emo in ['Positive', 'Joy', 'Trust']])
    if positive_score_sum > 45 and pred_label == 'Depression':
        pred_label = 'Control'

    return pred_label, subemotion_scores

# 🧠 Streamlit Interface
st.title("🧠 Mental Health Prediction from Tumblr Posts")
st.write("Analyze Tumblr blog posts and visualize emotional influences.")

blogname = st.text_input("🔍 Enter Tumblr blog name:")

if blogname:
    with st.spinner('🔍 Fetching Tumblr posts and analyzing...'):
        full_text = fetch_tumblr_posts(blogname)
        time.sleep(1)

    if not full_text.strip():
        st.error("No textual activity found for this user.")
    else:
        with st.spinner('🧠 Running prediction model...'):
            label, subemotions = predict(full_text)
            time.sleep(1)

        st.markdown(f"<div class='big-font'>🧠 Predicted Mental State: <span style='color:#00FFAA'>{label}</span></div>", unsafe_allow_html=True)

        if label == "Control":
            st.balloons()

        st.subheader("💡 Subemotion Influence (%):")
        for k, v in subemotions.items():
            if v > 10:
                emoji = {
                    "Joy": "😊", "Trust": "🤝", "Positive": "👍",
                    "Anger": "😠", "Fear": "😨", "Sadness": "😢",
                    "Disgust": "🤢", "Surprise": "😲", "Anticipation": "⏳"
                }.get(k, "💬")
                st.markdown(f"<span class='emoji'>{emoji}</span>**{k}**: {v:.2f}%", unsafe_allow_html=True)

        plot_subemotions(subemotions)
