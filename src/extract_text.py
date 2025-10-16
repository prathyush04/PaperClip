import os
import sys
from pathlib import Path
from PyPDF2 import PdfReader
from tqdm import tqdm

def extract_text_from_pdf(pdf_path: Path) -> str:

    try:
        reader = PdfReader(str(pdf_path))
        parts = []
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                parts.append(txt)
        return "\n".join(parts).strip()
    except Exception as e:
        
        print(f"[WARN] Could not extract {pdf_path}: {e}")
        return ""

def write_txt(out_path: Path, text: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fw:
        fw.write(text if text else "")

def should_process(pdf_path: Path, out_root: Path, base_root: Path) -> bool:
    rel = pdf_path.relative_to(base_root)
    out_path = out_root.joinpath(rel).with_suffix(".txt")
    if out_path.exists():
        try:
            if out_path.stat().st_size > 0:
                return False
        except:
            return True
    return True

def extract_folder_tree(base_root: Path, out_root: Path, patterns=(".pdf",)):
    base_root = base_root.resolve()
    out_root = out_root.resolve()
    all_pdfs = [p for p in base_root.rglob("*") if p.suffix.lower() in patterns and p.is_file()]
    print(f"Found {len(all_pdfs)} PDF files under {base_root}")
    for pdf in tqdm(all_pdfs, desc="PDFs"):
        rel = pdf.relative_to(base_root)
        out_txt = out_root.joinpath(rel).with_suffix(".txt")
        if not should_process(pdf, out_root, base_root):
            continue
        text = extract_text_from_pdf(pdf)
        write_txt(out_txt, text)

def main():
    project_root = Path(__file__).parent.parent
    default_base = project_root / "dataset"
    if not default_base.exists():
        print(f"[ERROR] Expected dataset folder at {default_base}.")
        sys.exit(1)

    out_root = project_root / "data" / "text"
    extract_folder_tree(default_base, out_root)
    print("Done. Texts saved under:", out_root)

if __name__ == "__main__":
    main()