import pickle
import re
import string
import os
from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__, template_folder="templates", static_folder="static")

# ✅ Load pre-trained model and vectorizer
with open("spam_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# ✅ Text preprocessing function (same as used in training)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    return text

# ✅ Function to predict a single SMS message
def predict_sms(message):
    message = clean_text(message)
    message_tfidf = vectorizer.transform([message])  # Convert to TF-IDF
    prediction = model.predict(message_tfidf)[0]
    return "Spam" if prediction == 1 else "Ham"

# ✅ Handle favicon requests to prevent 404 errors
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.static_folder), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

# ✅ Flask Route for Main Page
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        message = request.form["message"]  # Get user input
        prediction = predict_sms(message)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
