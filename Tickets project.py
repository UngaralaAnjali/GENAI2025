"""
test1.py
Robust pipeline running 7 embedding methods:
 - tfidf
 - bow (count)
 - hashing
 - char-ngrams
 - word-ngrams
 - doc2vec (gensim) [optional]
 - sentence-transformer (sentence-transformers) [optional]

Saves predictions_{method}.csv in the same folder.
"""
import os
import re
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (
    TfidfVectorizer, CountVectorizer, HashingVectorizer
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Optional imports (may not be installed)
try:
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    HAS_GENSIM = True
except Exception:
    HAS_GENSIM = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except Exception:
    HAS_ST = False


# ---------------- Preprocessor ----------------
class Preprocessor:
    """
    Extracts the right-hand side values from ticket text lines such as:
      Priority:HIGH  -> we extract "HIGH"
      Created:2024-01-28 11:45:00 -> we extract "2024-01-28 11:45:00"
    This makes the classification task meaningful (predict the label from the value).
    """
    def __init__(self):
        # patterns capture everything after the colon up to end-of-line
        self.patterns = {
            "Ticket ID": r"Ticket ID:\s*(.+)",
            "Priority": r"Priority:\s*(.+)",
            "Category": r"Category:\s*(.+)",
            "Status": r"Status:\s*(.+)",
            "Created": r"Created:\s*(.+)",
        }

    def preprocess_files(self, folder_path):
        data = []
        if not os.path.isdir(folder_path):
            print(f"⚠️ Folder not found: {folder_path}")
            return pd.DataFrame(columns=["X", "Y", "SourceFile"])

        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(".txt"):
                path = os.path.join(folder_path, filename)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except Exception as e:
                    print(f"Could not read {filename}: {e}")
                    continue

                for label, pattern in self.patterns.items():
                    for match in re.findall(pattern, text):
                        # strip whitespace
                        value = match.strip()
                        if value:
                            data.append([value, label, filename])

        return pd.DataFrame(data, columns=["X", "Y", "SourceFile"])


# ---------------- Embedding + training helpers ----------------
def get_embeddings_for_method(method, X_train, X_test):
    """
    Returns X_train_emb, X_test_emb, and an object describing the embedding (vectorizer/model).
    Will raise RuntimeError if an optional dependency is missing (handled by caller).
    """
    method = method.lower()

    if method == "tfidf":
        vect = TfidfVectorizer()
        X_train_emb = vect.fit_transform(X_train)
        X_test_emb = vect.transform(X_test)
        return X_train_emb, X_test_emb, vect

    if method == "bow":
        vect = CountVectorizer()
        X_train_emb = vect.fit_transform(X_train)
        X_test_emb = vect.transform(X_test)
        return X_train_emb, X_test_emb, vect

    if method == "hashing":
        hv = HashingVectorizer(n_features=2 ** 12, alternate_sign=False)
        # HashingVectorizer is stateless: transform directly
        X_train_emb = hv.transform(X_train)
        X_test_emb = hv.transform(X_test)
        return X_train_emb, X_test_emb, hv

    if method == "char-ngrams":
        vect = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
        X_train_emb = vect.fit_transform(X_train)
        X_test_emb = vect.transform(X_test)
        return X_train_emb, X_test_emb, vect

    if method == "word-ngrams":
        vect = TfidfVectorizer(ngram_range=(1, 2))
        X_train_emb = vect.fit_transform(X_train)
        X_test_emb = vect.transform(X_test)
        return X_train_emb, X_test_emb, vect

    if method == "doc2vec":
        if not HAS_GENSIM:
            raise RuntimeError("gensim not installed. Install with: pip install gensim")
        # Prepare TaggedDocument on X_train
        tagged = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(X_train)]
        model = Doc2Vec(vector_size=50, min_count=1, epochs=30)
        model.build_vocab(tagged)
        model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)
        # Get vectors for train (dedicated vectors) and infer for test
        X_train_emb = np.array([model.dv[str(i)] for i in range(len(X_train))])
        X_test_emb = np.array([model.infer_vector(text.split()) for text in X_test])
        return X_train_emb, X_test_emb, model

    if method == "sentence-transformer":
        if not HAS_ST:
            raise RuntimeError("sentence-transformers not installed. Install with: pip install sentence-transformers")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        X_train_emb = model.encode(list(X_train), convert_to_numpy=True, show_progress_bar=False)
        X_test_emb = model.encode(list(X_test), convert_to_numpy=True, show_progress_bar=False)
        return X_train_emb, X_test_emb, model

    raise ValueError(f"Unknown method: {method}")


def run_one_method(method, X_train, X_test, y_train, y_test, folder_path):
    print(f"\n----- RUNNING: {method} -----")
    try:
        X_train_emb, X_test_emb, emb_obj = get_embeddings_for_method(method, X_train, X_test)
    except Exception as e:
        print(f"Skipping {method} (error preparing embeddings): {e}")
        return False

    # Check labels in training: logistic regression needs at least 2 classes
    unique_labels = np.unique(y_train)
    if len(unique_labels) < 2:
        print(f"Skipping {method}: training set has <2 classes ({unique_labels})")
        return False

    # Train classifier
    clf = LogisticRegression(max_iter=1000)
    try:
        clf.fit(X_train_emb, y_train)
    except Exception as e:
        print(f"Failed to fit classifier for {method}: {e}")
        return False

    # Predict and evaluate
    try:
        y_pred = clf.predict(X_test_emb)
    except Exception as e:
        print(f"Failed to predict for {method}: {e}")
        return False

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save predictions to CSV
    out_df = pd.DataFrame({
        "Text": list(X_test),
        "ActualLabel": list(y_test),
        "PredictedLabel": list(y_pred)
    })
    fname = f"predictions_{method.replace(' ', '_')}.csv"
    save_path = os.path.join(folder_path, fname)
    try:
        out_df.to_csv(save_path, index=False, encoding="utf-8")
        print(f"Saved predictions -> {save_path}")
    except Exception as e:
        print(f"Could not save predictions for {method}: {e}")

    return True


# ---------------- Main ----------------
if __name__ == "__main__":
    print("Running file:", os.path.abspath(__file__))
    print("gensim available:", HAS_GENSIM)
    print("sentence-transformers available:", HAS_ST)
    print("-" * 60)

    folder_path = r"C:\Users\DELL\Desktop\Tickets Project"

    pre = Preprocessor()
    df = pre.preprocess_files(folder_path)

    print("Extracted rows:", len(df))
    print(df.head(), "\n")

    if df.empty:
        print("No data found. Place your ticket .txt files inside the folder above and re-run.")
        sys.exit(0)

    # split
    X = df["X"].astype(str)
    y = df["Y"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)

    methods = [
        "tfidf",
        "bow",
        "hashing",
        "char-ngrams",
        "word-ngrams",
        "doc2vec",
        "sentence-transformer"
    ]

    success = {}
    for m in methods:
        ok = run_one_method(m, X_train, X_test, y_train, y_test, folder_path)
        success[m] = ok

    print("\nSummary (method -> success):")
    for k, v in success.items():
        print(f"  {k}: {'OK' if v else 'SKIPPED/FAILED'}")

    print("\nDone.")
