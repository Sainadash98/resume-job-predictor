import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/resumes.csv")

X = df["Resume"]
y = df["Category"]

# TF-IDF vectorizer (FIT HERE)
tfidf = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf.fit_transform(X)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_tfidf, y_encoded)

# Save trained objects
with open("models/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Training complete")
print("TF-IDF fitted:", hasattr(tfidf, "idf_"))
