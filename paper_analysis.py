import os
import sys
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List, Dict
from pathway_processor import PathwayPDFProcessor
import re

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Define paths - these will match the Docker volume mounts
reference_path = "/app/Reference"
unclassified_path = "/app/papers"
output_path = "/app/output"

def check_directories():
    """Verify all required directories exist and are accessible"""
    try:
        for path in [reference_path, unclassified_path, output_path]:
            if not os.path.exists(path):
                print(f"Creating directory: {path}")
                os.makedirs(path)
            if not os.access(path, os.R_OK | os.W_OK):
                raise PermissionError(f"Cannot access directory: {path}")
        
        publishable_path = os.path.join(reference_path, "Publishable")
        non_publishable_path = os.path.join(reference_path, "Non-Publishable")
        
        # Create conference subdirectories if they don't exist
        conference_dirs = ['CVPR', 'EMNLP', 'KDD', 'NeurIPS', 'TMLR']
        for conf in conference_dirs:
            conf_path = os.path.join(publishable_path, conf)
            if not os.path.exists(conf_path):
                os.makedirs(conf_path)
            
        print("Directory structure verified successfully")
        return True
    except Exception as e:
        print(f"Error checking directories: {str(e)}")
        return False

def read_pdf(file_path: str) -> str:
    """Read and extract text from a PDF file with better error handling"""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = []
            for page in reader.pages:
                try:
                    text.append(page.extract_text())
                except Exception as e:
                    print(f"Warning: Could not extract text from page in {file_path}: {e}")
            return " ".join(text).strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def preprocess_text(text: str) -> str:
    """Preprocess text by removing stopwords and normalizing"""
    try:
        if not text:
            return ""
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
        return " ".join(tokens)
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return text

def extract_sections(text: str) -> Dict[str, str]:
    """Extract key sections from the paper text with improved section detection"""
    sections = {
        'abstract': '',
        'introduction': '',
        'methodology': '',
        'results': '',
        'conclusion': ''
    }
    
    # Common section header patterns
    section_patterns = {
        'abstract': ['abstract', 'summary'],
        'introduction': ['introduction', '1.', 'i.', 'overview'],
        'methodology': ['method', 'approach', '3.', 'iii.', 'implementation'],
        'results': ['result', 'evaluation', '4.', 'iv.', 'experiment'],
        'conclusion': ['conclusion', 'discussion', '5.', 'v.', 'future work']
    }
    
    current_section = None
    lines = text.split('\n')
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Identify sections using patterns
        for section, patterns in section_patterns.items():
            if any(pattern in line_lower[:30] for pattern in patterns):
                current_section = section
                break
                
        # Add content to current section
        if current_section and line.strip():
            sections[current_section] += line + '\n'
    
    return sections

class RationaleGenerator:
    def __init__(self):
        # Define evaluation criteria and weights
        self.criteria = {
            'similarity_thresholds': {
                'high': 0.85,
                'medium': 0.70,
                'low': 0.50
            },
            'keywords': {
                'methodology': {
                    'experiment': 3,
                    'analysis': 2,
                    'evaluation': 2,
                    'framework': 2,
                    'algorithm': 2,
                    'method': 1,
                    'approach': 1
                },
                'novelty': {
                    'novel': 3,
                    'innovative': 3,
                    'new': 2,
                    'proposed': 2,
                    'improvement': 2,
                    'enhanced': 1,
                    'advanced': 1
                },
                'results': {
                    'significant': 3,
                    'outperform': 3,
                    'improvement': 2,
                    'accuracy': 2,
                    'performance': 2,
                    'effective': 1,
                    'efficient': 1
                }
            }
        }

    def _analyze_section(self, text: str, keyword_dict: Dict[str, int]) -> float:
        """Analyze a section using weighted keywords"""
        if not text:
            return 0.0
            
        text = text.lower()
        total_score = 0
        max_possible = sum(weight * 5 for weight in keyword_dict.values())
        
        for keyword, weight in keyword_dict.items():
            count = text.count(keyword)
            total_score += count * weight
            
        return min(total_score / max_possible, 1.0)

    def _extract_metrics(self, text: str) -> Dict[str, List[float]]:
        """Extract numerical metrics from text"""
        metrics = {}
        
        # Find percentages
        percentages = re.findall(r'(\d+\.?\d*)%', text)
        if percentages:
            metrics['percentages'] = [float(p) for p in percentages]
            
        # Find accuracy/performance values
        performance = re.findall(r'(?:accuracy|precision|recall|f1)[\s:]+(\d+\.?\d*)', text.lower())
        if performance:
            metrics['performance'] = [float(v) for v in performance]
            
        return metrics

    def generate_rationale(self, paper_text: str, matched_conference: str, 
                         similarity_score: float, is_publishable: bool) -> str:
        """Generate a comprehensive rationale based on paper analysis"""
        sections = extract_sections(paper_text)
        
        # Analyze each aspect
        methodology_score = self._analyze_section(
            sections['methodology'], 
            self.criteria['keywords']['methodology']
        )
        
        novelty_score = self._analyze_section(
            sections['abstract'] + sections['introduction'], 
            self.criteria['keywords']['novelty']
        )
        
        results_score = self._analyze_section(
            sections['results'], 
            self.criteria['keywords']['results']
        )
        
        metrics = self._extract_metrics(sections['results'])
        
        # Build rationale
        reasons = []
        
        # Evaluate similarity
        if similarity_score >= self.criteria['similarity_thresholds']['high']:
            if is_publishable:
                reasons.append(f"Strong alignment with {matched_conference} themes")
            else:
                reasons.append("High similarity with existing work raises originality concerns")
        elif similarity_score >= self.criteria['similarity_thresholds']['medium']:
            reasons.append(f"Good thematic alignment with {matched_conference}" if is_publishable 
                         else "Moderate thematic alignment")
        
        # Evaluate methodology
        if methodology_score > 0.7:
            reasons.append("Strong methodological foundation")
        elif methodology_score < 0.3:
            reasons.append("Methodology needs more detailed exposition")
        
        # Evaluate novelty
        if novelty_score > 0.7:
            reasons.append("Demonstrates significant innovation")
        elif novelty_score < 0.3:
            reasons.append("Limited novelty in approach")
        
        # Evaluate results
        if results_score > 0.7:
            reasons.append("Well-supported results with strong validation")
        elif results_score < 0.3:
            reasons.append("Results require stronger empirical support")
        
        # Add metrics-based evaluation
        if metrics.get('performance'):
            max_perf = max(metrics['performance'])
            if max_perf > 90:
                reasons.append(f"Excellent performance metrics ({max_perf:.1f}%)")
            elif max_perf > 80:
                reasons.append(f"Strong performance metrics ({max_perf:.1f}%)")
        
        # Combine reasons
        if is_publishable:
            rationale = f"Recommended for {matched_conference} based on: "
        else:
            rationale = "Not recommended for publication due to: "
            
        return rationale + "; ".join(reasons) + "."

class PaperProcessor:
    def __init__(self):
        print("Initializing PaperProcessor...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        try:
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=self.device)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        self.rationale_generator = RationaleGenerator()
        self.embeddings = {}

    def process_pdf_files(self, directory: str, is_reference: bool = False) -> List[Dict]:
        """Process all PDF files in a directory with improved error handling"""
        documents = []
        print(f"Processing directory: {directory}")
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    print(f"Processing {file_path}")
                    
                    try:
                        content = read_pdf(file_path)
                        if not content:
                            print(f"Warning: No content extracted from {file_path}")
                            continue
                        
                        is_publishable = "Publishable" in root if is_reference else None
                        conference = os.path.basename(root) if is_reference else None
                        
                        doc = {
                            "id": file,
                            "text": content,
                            "processed_text": preprocess_text(content),
                            "is_publishable": is_publishable,
                            "conference": conference,
                            "is_reference": is_reference,
                            "filename": file,
                            "path": file_path
                        }
                        
                        embedding = self.model.encode(doc["processed_text"], convert_to_tensor=True)
                        self.embeddings[file] = embedding
                        documents.append(doc)
                        print(f"Successfully processed {file}")
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        continue
                        
        return documents

    def compute_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            return float(torch.nn.functional.cosine_similarity(embedding1.unsqueeze(0), 
                                                             embedding2.unsqueeze(0)))
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0

    def compute_similarities_with_rationale(self, reference_docs: List[Dict], 
                                         unclassified_docs: List[Dict]) -> List[Dict]:
        """Compute similarities and generate rationales with improved error handling"""
        results = []
        
        for unclass_doc in unclassified_docs:
            try:
                unclass_embedding = self.embeddings[unclass_doc["filename"]]
                
                max_similarity = -1
                best_match = None
                
                for ref_doc in reference_docs:
                    ref_embedding = self.embeddings[ref_doc["filename"]]
                    similarity = self.compute_similarity(unclass_embedding, ref_embedding)
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match = ref_doc
                
                if best_match:
                    rationale = self.rationale_generator.generate_rationale(
                        unclass_doc["text"],
                        best_match["conference"],
                        max_similarity,
                        best_match["is_publishable"]
                    )
                    
                    results.append({
                        "filename": unclass_doc["filename"],
                        "matched_with": best_match["filename"],
                        "similarity_score": max_similarity,
                        "is_publishable": best_match["is_publishable"],
                        "conference": best_match["conference"],
                        "plagiarism_flag": max_similarity > 0.8,
                        "rationale": rationale
                    })
            except Exception as e:
                print(f"Error processing {unclass_doc['filename']}: {e}")
                continue
                
        return results

def main():
    try:
        print("Starting paper analysis...")
        
        if not check_directories():
            print("Directory verification failed. Exiting...")
            sys.exit(1)

        # Initialize Pathway processor
        pathway_processor = PathwayPDFProcessor(reference_path, unclassified_path)
        
        # Process PDFs using Pathway
        print("\nProcessing papers using Pathway...")
        processed_docs = pathway_processor.process_pdfs()
        
        # Run Pathway pipeline
        pw.run()
            
        # Initialize paper processor for similarity analysis
        processor = PaperProcessor()
        
        print("\nProcessing reference papers...")
        reference_docs = [
            doc for doc in processed_docs 
            if doc.is_reference and doc.text.strip()
        ]
        
        print("\nProcessing unclassified papers...")
        unclassified_docs = [
            doc for doc in processed_docs 
            if not doc.is_reference and doc.text.strip()
        ]
        
        if not reference_docs or not unclassified_docs:
            print("No documents to process. Exiting...")
            sys.exit(1)
        
        print("\nComputing similarities and generating rationales...")
        results = processor.compute_similarities_with_rationale(reference_docs, unclassified_docs)
        
        if results:
            # Create output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            
            # Save results to CSV
            results_df = pd.DataFrame(results)
            output_file = os.path.join(output_path, "analysis_results.csv")
            results_df.to_csv(output_file, index=False)
            
            # Generate detailed report
            report_file = os.path.join(output_path, "detailed_report.txt")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("Paper Analysis Report\n")
                f.write("===================\n\n")
                for result in results:
                    f.write(f"Paper: {result['filename']}\n")
                    f.write(f"Status: {'Publishable' if result['is_publishable'] else 'Non-Publishable'}\n")
                    if result['is_publishable']:
                        f.write(f"Recommended Conference: {result['conference']}\n")
                    f.write(f"Similarity Score: {result['similarity_score']:.2f}\n")
                    f.write("Rationale:\n")
                    f.write(f"{result['rationale']}\n")
                    f.write("\n" + "="*50 + "\n\n")
            
            print(f"\nResults saved to: {output_file}")
            print(f"Detailed report saved to: {report_file}")
            
            print("\nAnalysis Summary:")
            print(f"Total reference documents processed: {len(reference_docs)}")
            print(f"Total unclassified documents processed: {len(unclassified_docs)}")
            print(f"Total matches found: {len(results)}")
            print(f"Potential plagiarism cases: {sum(1 for r in results if r['plagiarism_flag'])}")
        else:
            print("\nNo similarity results were generated")
            
        print("\nAnalysis complete.")
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()