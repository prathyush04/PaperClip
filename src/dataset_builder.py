import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from extract_text import extract_text_from_pdf
from preprocess import preprocess

def build_reference_dataset(base_folder, out_csv):
    rows = []
    publishable_folder = Path("dataset") / "Reference" / "Publishable"
    nonpub_folder = Path("dataset") / "Reference" / "Non-Publishable"

    for conf in os.listdir(publishable_folder):
        conf_dir = publishable_folder / conf
        if not conf_dir.is_dir():
            continue
        for pdf_file in conf_dir.glob("*.pdf"):
            txt_file = Path("data") / "text" / "Reference" / "Publishable" / conf / pdf_file.with_suffix(".txt").name
            if not txt_file.exists():
                print(f"[WARN] Missing text file for {pdf_file}")
                continue
            text = open(txt_file, 'r', encoding='utf-8').read()
            text = preprocess(text)
            rows.append({
                "text": text,
                "label_pub": 1,
                "conf": conf,
                "filename": pdf_file.name
            })

    for pdf_file in nonpub_folder.glob("*.pdf"):
        txt_file = Path("data") / "text" / "Reference" / "Non-Publishable" / pdf_file.with_suffix(".txt").name
        if not txt_file.exists():
            print(f"[WARN] Missing text file for {pdf_file}")
            continue
        text = open(txt_file, 'r', encoding='utf-8').read()
        text = preprocess(text)
        rows.append({
            "text": text,
            "label_pub": 0,
            "conf": "none",
            "filename": pdf_file.name
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved CSV to {out_csv}, total rows: {len(df)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_folder", default="PaperClip")
    parser.add_argument("--out_csv", default="data/reference_dataset.csv")
    args = parser.parse_args()
    build_reference_dataset(args.base_folder, args.out_csv)
