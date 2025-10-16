import spacy
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.strip()

def preprocess(text, remove_stopwords=True, lemmatize=True):
    text = clean_text(text)
    tokens = text.split()
    if lemmatize:
        doc = nlp(" ".join(tokens))
        tokens = [tok.lemma_ for tok in doc if not tok.is_space]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

if __name__ == "__main__":
    import sys
    txt_path = sys.argv[1]
    text = open(txt_path, 'r', encoding='utf-8').read()
    processed = preprocess(text)
    print(processed[:500])
