import pathway as pw
import io
from PyPDF2 import PdfReader
import os
from typing import Dict

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
            return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

class PathwayPDFProcessor:
    def __init__(self, reference_path: str, papers_path: str):
        self.reference_path = reference_path
        self.papers_path = papers_path

    def get_pdf_files(self, directory: str) -> list:
        """Get all PDF files from a directory and its subdirectories."""
        pdf_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        return pdf_files

    def process_pdfs(self):
        """Set up Pathway pipeline for processing PDFs."""
        # Create tables for reference and unclassified papers
        reference_files = self.get_pdf_files(self.reference_path)
        papers_files = self.get_pdf_files(self.papers_path)

        # Create Pathway tables
        reference_table = pw.Table.from_python(
            [{"path": path, "is_reference": True} for path in reference_files]
        )
        
        papers_table = pw.Table.from_python(
            [{"path": path, "is_reference": False} for path in papers_files]
        )

        # Process PDFs in parallel
        processed_reference = reference_table.select(
            text=pw.apply(extract_text_from_pdf, reference_table.path),
            path=reference_table.path,
            is_reference=reference_table.is_reference
        )

        processed_papers = papers_table.select(
            text=pw.apply(extract_text_from_pdf, papers_table.path),
            path=papers_table.path,
            is_reference=papers_table.is_reference
        )

        # Combine tables
        all_processed = pw.concat([processed_reference, processed_papers])

        return all_processed

def main():
    # Initialize processor with your paths
    processor = PathwayPDFProcessor(
        reference_path="C:/Users/Vinish/Reference",
        papers_path="C:/Users/Vinish/Papers"
    )

    # Process PDFs using Pathway
    processed_table = processor.process_pdfs()

    # Run the pipeline
    pw.run()

if __name__ == "__main__":
    main()