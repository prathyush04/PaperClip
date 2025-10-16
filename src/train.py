import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train(reference_csv, out_dir="models"):
    df = pd.read_csv(reference_csv)
    df = df.dropna(subset=['text'])

    X = df['text'].values
    y_bin = df['label_pub'].values

    os.makedirs(out_dir, exist_ok=True)

    print("\n[INFO] Training Binary Classifier (Publishable vs Non-publishable)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )

    pipe_bin = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipe_bin.fit(X_train, y_train)
    preds = pipe_bin.predict(X_test)
    print("[RESULT] Binary Classifier Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    joblib.dump(pipe_bin, os.path.join(out_dir, "binary_pipeline.joblib"))
    print(f"[INFO] Binary model saved to {out_dir}/binary_pipeline.joblib")

    print("\n[INFO] Training Multiclass Conference Classifier (only publishable papers)...")
    df_pub = df[df['label_pub'] == 1]
    X_pub = df_pub['text'].values
    y_pub = df_pub['conf'].values

    Xtr, Xte, ytr, yte = train_test_split(
        X_pub, y_pub, test_size=0.2, random_state=42
    )

    pipe_multi = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000, multi_class='multinomial'))
    ])

    pipe_multi.fit(Xtr, ytr)
    preds = pipe_multi.predict(Xte)
    print("[RESULT] Multiclass Classifier Accuracy:", accuracy_score(yte, preds))
    print(classification_report(yte, preds))
    joblib.dump(pipe_multi, os.path.join(out_dir, "multiclass_pipeline.joblib"))
    print(f"[INFO] Multiclass model saved to {out_dir}/multiclass_pipeline.joblib")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_csv", default="data/reference_dataset.csv")
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()
    train(args.reference_csv, args.out_dir)
