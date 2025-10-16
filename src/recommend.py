import os
from pathlib import Path
import joblib
from preprocess import preprocess

def recommend(paper_path, model_dir="models"):
    paper_path = Path(paper_path)
    if not paper_path.exists():
        print(f"[ERROR] File not found: {paper_path}")
        return

    txt_file = Path("data") / "text" / "Papers" / paper_path.with_suffix(".txt").name
    if not txt_file.exists():
        print(f"[WARN] Missing text file. Reading PDF directly (if PDF text embedded)...")
        print("Please run extract_text.py to generate .txt files first.")
        return

    text = open(txt_file, "r", encoding="utf-8").read()
    text_proc = preprocess(text)

    bin_model = joblib.load(os.path.join(model_dir, "binary_pipeline.joblib"))
    multi_model = joblib.load(os.path.join(model_dir, "multiclass_pipeline.joblib"))

    pred_bin = bin_model.predict([text_proc])[0]
    pred_conf = multi_model.predict([text_proc])[0] if pred_bin == 1 else "none"

    print(f"\nPaper: {paper_path.name}")
    print(f"Predicted publishable: {'YES' if pred_bin==1 else 'NO'}")
    if pred_bin == 1:
        print(f"Recommended conference: {pred_conf}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", required=True, help="Path to the PDF paper")
    parser.add_argument("--model_dir", default="models")
    args = parser.parse_args()
    recommend(args.paper, args.model_dir)
