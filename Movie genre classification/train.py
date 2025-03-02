import os
import re
import string
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import SGDClassifier

# ✅ Define dataset path
dataset_path = "D:/Movie genre classification/Genre Classification Dataset"
train_data_path = os.path.join(dataset_path, "train_data.txt")

# ✅ Initialize lists
plots = []
genres = []

# ✅ Load training data
try:
    with open(train_data_path, "r", encoding="ISO-8859-1") as file:
        for line in file:
            line = line.strip()
            parts = line.split(" ::: ")
            if len(parts) >= 4:
                genres.append(parts[2].strip())  # Genre
                plots.append(parts[3].strip())   # Plot summary
except FileNotFoundError:
    print(f"❌ Error: File not found at {train_data_path}")
    exit()

# ✅ Check if data is loaded properly
if len(plots) == 0 or len(genres) == 0:
    raise ValueError("❌ Error: No valid data found. Check 'train_data.txt' format.")

print(f"✅ Total samples loaded: {len(plots)}")

# ✅ Convert to DataFrame
df = pd.DataFrame({"plot": plots, "genre": genres})

# ✅ Improved Text Preprocessing Function
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df['clean_plot'] = df['plot'].apply(clean_text)

# ✅ Check if cleaned data is still empty
if df['clean_plot'].isnull().sum() > 0 or df['clean_plot'].apply(lambda x: len(x)).sum() == 0:
    raise ValueError("❌ Error: Cleaned plots are empty after preprocessing.")

# ✅ Remove genres with fewer than 2 samples to fix ValueError
genre_counts = df['genre'].value_counts()
df = df[df['genre'].isin(genre_counts[genre_counts > 1].index)]

# ✅ Re-encode genres after filtering
label_encoder = LabelEncoder()
df['genre_encoded'] = label_encoder.fit_transform(df['genre'])

# ✅ Ensure dataset is large enough after filtering
if df.shape[0] < 2:
    raise ValueError("❌ Error: Not enough data after filtering rare genres.")

# ✅ Dynamically adjust test_size to prevent stratification errors
num_classes = df['genre_encoded'].nunique()
min_test_size = max(num_classes, int(0.2 * len(df)))  # Ensure at least 1 sample per class
test_size = min_test_size / len(df) if min_test_size < len(df) else 0.2

print(f"✅ Adjusted test_size: {test_size:.2f} ({min_test_size} samples)")

# ✅ Perform the train-test split safely
X_train, X_test, y_train, y_test = train_test_split(df['clean_plot'], df['genre_encoded'],test_size=0.3, train_size=0.7, stratify=df['genre_encoded'], random_state=42)


# ✅ TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ✅ Train Different Models and Evaluate Performance
models = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "Naive Bayes": MultinomialNB(),
    "SVM": SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3)


}

best_model = None
best_accuracy = 0
best_model_name = ""

for model_name, model in models.items():
    print(f"\n🔹 Training {model_name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ {model_name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, labels=np.unique(y_test), target_names=[label_encoder.classes_[i] for i in np.unique(y_test)]))

    # ✅ Store best model details
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = model_name

# ✅ Ensure model was selected before printing
if best_model is not None:
    print(f"\n🎉 Best Model Selected: {best_model_name}")
    print(f"✅ Best Model Accuracy: {best_accuracy:.4f}")
else:
    print("❌ No model was selected. Check dataset or training process.")

# ✅ Save best model, vectorizer, and label encoder
joblib.dump(best_model, 'movie_genre_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print(f"✅ Best model ({best_model_name}) saved successfully!")
