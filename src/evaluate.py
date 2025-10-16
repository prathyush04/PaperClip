import os
from pathlib import Path
import joblib
import pandas as pd
from preprocess import preprocess

def evaluate(papers_folder="dataset/Papers", model_dir="models", out_csv="data/papers_results.csv"):
    papers_folder = Path(papers_folder)
    bin_model = joblib.load(os.path.join(model_dir, "binary_pipeline.joblib"))
    multi_model = joblib.load(os.path.join(model_dir, "multiclass_pipeline.joblib"))

    rows = []
    all_pdfs = list(papers_folder.glob("*.pdf"))
    print(f"[INFO] Found {len(all_pdfs)} papers to evaluate")

    for pdf in all_pdfs:
        txt_file = Path("data") / "text" / "Papers" / pdf.with_suffix(".txt").name
        if not txt_file.exists():
            print(f"[WARN] Missing text file for {pdf}")
            continue
        text = open(txt_file, "r", encoding="utf-8").read()
        text_proc = preprocess(text)

        pred_bin = bin_model.predict([text_proc])[0]
        pred_conf = multi_model.predict([text_proc])[0] if pred_bin == 1 else "none"

        rows.append({
            "filename": pdf.name,
            "predicted_publishable": int(pred_bin),
            "predicted_conf": pred_conf
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Predictions saved to {out_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers_folder", default="dataset/Papers")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--out_csv", default="data/papers_results.csv")
    args = parser.parse_args()
    evaluate(args.papers_folder, args.model_dir, args.out_csv)
