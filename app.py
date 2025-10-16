from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from pathlib import Path
import joblib
from src.preprocess import preprocess
from src.extract_text import extract_text_from_pdf

app = Flask(__name__)
CORS(app)

MODEL_DIR = "models"

bin_model = joblib.load(os.path.join(MODEL_DIR, "binary_pipeline.joblib"))
multi_model = joblib.load(os.path.join(MODEL_DIR, "multiclass_pipeline.joblib"))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files['file']
        filename = file.filename
        save_path = os.path.join("uploads", filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(save_path)

        text = extract_text_from_pdf(Path(save_path))
        if not text:
            return jsonify({"error": "Text extraction failed"}), 500
        text_proc = preprocess(text)

        pred_bin = bin_model.predict([text_proc])[0]
        pred_conf = multi_model.predict([text_proc])[0] if pred_bin == 1 else "none"

        return jsonify({
            "filename": filename,
            "publishable": bool(pred_bin),
            "conference": pred_conf
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
