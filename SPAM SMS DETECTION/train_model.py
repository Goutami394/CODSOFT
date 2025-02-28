import pandas as pd
import re
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ✅ Load dataset
df = pd.read_csv("D:/Codsoft/spam.csv", encoding="ISO-8859-1", usecols=[0, 1])
df.columns = ["label", "message"]  # Renaming for better readability

# ✅ Convert labels to binary (spam = 1, ham = 0)
df["label"] = df["label"].map({"spam": 1, "ham": 0})

# ✅ Handle missing values
df.dropna(inplace=True)

# ✅ Text preprocessing function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    return text

# ✅ Apply preprocessing
df["message"] = df["message"].apply(clean_text)

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label"], test_size=0.2, random_state=42)

# ✅ Convert text to TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ✅ Train Naïve Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ✅ Evaluate model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Save model and vectorizer for Flask app
with open("spam_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("✅ Model and vectorizer saved successfully!")
