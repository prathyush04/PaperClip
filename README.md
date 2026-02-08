# 📎 PaperClip

AI-powered research paper publishability prediction and conference recommendation system.

## 🚀 Features

- **Binary Classification**: Predicts if a research paper is publishable or not
- **Conference Recommendation**: Suggests the best conference for publishable papers
- **Web Interface**: User-friendly React frontend for easy paper upload
- **Batch Processing**: Evaluate multiple papers at once
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## 🏗️ Project Structure

```
PaperClip/
├── dataset/                    # Training data
│   ├── Papers/                # Research papers (P001-P135)
│   └── Reference/             # Labeled reference papers
├── data/                      # Processed data
│   ├── text/                 # Extracted text files
│   └── *.csv                 # Dataset files
├── models/                    # Trained ML models
├── src/                       # Python scripts
├── frontend/                  # React web interface
├── app.py                     # Flask API server
└── requirements.txt           # Python dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm or yarn

### Backend Setup
```bash
pip install -r requirements.txt
pip install flask-cors
python -m spacy download en_core_web_sm
```

### Frontend Setup
```bash
cd frontend
npm install
```

## 📊 Usage

### 1. Extract Text from PDFs
```bash
python src/extract_text.py
```

### 2. Build Training Dataset
```bash
python src/dataset_builder.py --out_csv data/reference_dataset.csv
```

### 3. Train Models
```bash
python src/train.py --reference_csv data/reference_dataset.csv --out_dir models
```

### 4. Start Web Application

**Backend (Terminal 1):**
```bash
python app.py
```

**Frontend (Terminal 2):**
```bash
cd frontend
npm start
```

### 5. Command Line Usage

**Single Paper Recommendation:**
```bash
python src/recommend.py --paper dataset/Papers/P001.pdf --model_dir models
```

**Batch Evaluation:**
```bash
python src/evaluate.py --papers_folder dataset/Papers --model_dir models --out_csv data/results.csv
```

## 🎯 Supported Conferences

- **CVPR** - Computer Vision and Pattern Recognition
- **EMNLP** - Empirical Methods in Natural Language Processing
- **KDD** - Knowledge Discovery and Data Mining
- **NeurIPS** - Neural Information Processing Systems
- **TMLR** - Transactions on Machine Learning Research

## 🔧 API Endpoints

### POST /predict
Upload a PDF file and get publishability prediction and conference recommendation.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: PDF file

**Response:**
```json
{
  "filename": "paper.pdf",
  "publishable": true,
  "conference": "CVPR"
}
```

## 📱 Web Interface

The responsive web interface allows users to:
- Upload PDF research papers
- Get real-time predictions
- View conference recommendations
- Works on all device sizes

## 🤖 Machine Learning Pipeline

1. **Text Extraction**: PyPDF2 extracts text from PDF files
2. **Preprocessing**: spaCy and NLTK clean and normalize text
3. **Feature Engineering**: TF-IDF vectorization with n-grams
4. **Binary Classification**: Logistic Regression for publishability
5. **Multiclass Classification**: Logistic Regression for conference recommendation

## 📈 Model Performance

- **Binary Classifier**: Distinguishes publishable vs non-publishable papers
- **Multiclass Classifier**: Recommends appropriate conference for publishable papers
- **Training Data**: 15 reference papers (10 publishable, 5 non-publishable)

## 🔄 Workflow

```
PDF Upload → Text Extraction → Preprocessing → Binary Prediction → Conference Recommendation
```

## 📋 Requirements

### Python Dependencies
- flask
- flask-cors
- pandas
- scikit-learn
- PyPDF2
- tqdm
- spacy
- nltk
- joblib

### Frontend Dependencies
- react
- axios

## 🚀 Deployment

1. Ensure all dependencies are installed
2. Train models with your dataset
3. Start Flask backend server
4. Build and serve React frontend
5. Configure reverse proxy if needed

## 📄 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For issues and questions, please open a GitHub issue.
