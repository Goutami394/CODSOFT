import os
import re
import string
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ Load trained model and vectorizer
MODEL_PATH = "movie_genre_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
    raise FileNotFoundError("❌ Trained model or vectorizer not found! Ensure they are in the same directory.")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# ✅ Flask App
app = Flask(__name__)

# ✅ Text Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        plot = request.form["plot"]
        cleaned_plot = clean_text(plot)

        # ✅ Transform using TF-IDF
        plot_tfidf = vectorizer.transform([cleaned_plot])

        # ✅ Predict genre
        genre_encoded = model.predict(plot_tfidf)[0]
        genre_predicted = label_encoder.inverse_transform([genre_encoded])[0]

        return render_template("index.html", plot=plot, predicted_genre=genre_predicted)

    return render_template("index.html", plot="", predicted_genre="")

if __name__ == "__main__":
    app.run(debug=True)
