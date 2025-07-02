from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import urllib.parse
import tldextract
import xgboost as xgb
import re

app = Flask(__name__)
CORS(app)

# === Load Models ===
try:
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('models/phishing_model.json')

    sql_model = load_model('models/sql_injection_model.h5')

    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

except Exception as e:
    print(f"[Model Load Error] {e}")
    raise

# === Feature Extraction for Phishing Detection ===
def extract_url_features(url):
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc
        path = parsed.path
        query = parsed.query
        ext = tldextract.extract(url)

        features = [
            len(url), int(url.startswith('https://')), domain.count('.'),
            int('@' in url), int('//' in url[8:]), int('-' in domain),
            len(ext.subdomain.split('.')), int(any(part.isdigit() for part in domain.split('.'))),
            len(domain), int(domain.lower() != ext.domain.lower()),
            len(path.split('/')), int('%' in url), int('&' in url), int('=' in url),
            int('?' in url), int('#' in url), int('~' in url), int(',' in url), int('+' in url),
            int('_' in url), int(';' in url), int('$' in url), int('!' in url), int('*' in url),
            int('(' in url), int(')' in url), int('|' in url), int('^' in url), int('{' in url),
            int('}' in url)
        ]

        # Keywords
        keywords = ['login', 'signin', 'bank', 'account', 'verify', 'secure', 'update', 'webscr', 'password', 'confirm']
        features.extend([int(keyword in url.lower()) for keyword in keywords])

        # TLDs
        tlds = ['.com', '.net', '.org', '.info', '.biz', '.ru', '.uk', '.in']
        features.extend([int(url.endswith(tld)) for tld in tlds])

        # Remove these placeholders:
        # features.extend([0]*30)

        features.append(int(len(query) > 0))
        features.append(int('php' in path.lower()))
        features.append(int('asp' in path.lower()))
        features.append(int('js' in path.lower()))

        # Pad zeros only if feature length < expected by model
        while len(features) < xgb_model.n_features_in_:
            features.append(0)

        return np.array(features)
    except Exception as e:
        print(f"[Feature Extraction Error] {e}")
        return np.zeros(xgb_model.n_features_in_)

# === Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/phishing', methods=['POST'])
def predict_phishing():
    data = request.get_json()
    url = data.get('url', '')

    if not url.startswith('http'):
        return jsonify({'error': 'Invalid URL'}), 400

    features = extract_url_features(url)
    if len(features) != xgb_model.n_features_in_:
        return jsonify({'error': 'Feature mismatch'}), 500
        
    pred_probs = xgb_model.predict_proba([features])[0]

    phishing_prob = pred_probs[0]  # class 0 = phishing
    safe_prob = pred_probs[1]      # class 1 = safe

    threshold = 0.3  # set based on what works best empirically

    if phishing_prob > threshold:
        pred_class = 0  # phishing
        confidence = float(phishing_prob)
    else:
        pred_class = 1  # safe
        confidence = float(safe_prob)


    print("[DEBUG] URL:", url)
    print("[DEBUG] Feature Vector Length:", len(features))
    print("[DEBUG] Feature Vector:", features)
    print("[DEBUG] Prediction Probabilities:", pred_probs)
    print("[DEBUG] Prediction Class:", pred_class)
    print("[DEBUG] Confidence Score:", confidence)
    print(f"[DEBUG] Final prediction: {pred_class} ({'Phishing' if pred_class == 0 else 'Safe'})")

    label_map = {0: "Phishing", 1: "Safe"}

    return jsonify({
        'prediction': pred_class,
        'confidence': confidence,
       # 'features': [f"feature_{i}: {val}" for i, val in enumerate(features)],
        })

# === Improved Rule-based Signature Detection ===
def is_malicious_pattern(query: str) -> list:
    q = query.lower()
    patterns = []

    if '1=1' in q and re.search(r"\bor\b|\band\b", q):
        patterns.append("Always true condition (1=1)")
    if re.search(r"--|#", q) and not re.search(r"like\\s+'%--%'", q):
        patterns.append("SQL comment detected")
    if re.search(r"\\bunion\\b", q) and "select" in q:
        patterns.append("UNION keyword detected")
    if re.search(r";\\s*(drop|delete)", q):
        patterns.append("Stacked query or destructive command")
    if 'sleep(' in q or 'waitfor' in q:
        patterns.append("Time delay function")
    if 'exec(' in q or 'xp_' in q:
        patterns.append("Command execution")

    return patterns

# === SQL Injection Detection Endpoint ===
@app.route('/predict/sql', methods=['POST'])
def predict_sql():
    data = request.get_json()
    query = data.get('query', '')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        cleaned = ' '.join(query.strip().lower().split())
        patterns = is_malicious_pattern(cleaned)

        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=100, padding='post')
        pred = float(sql_model.predict(padded)[0][0])

        # Add these debug prints:
        print(f"Cleaned Query: {cleaned}")
        print(f"Model Prediction (raw): {pred}")

        # === Confidence Boost for Legitimate Patterns ===
        # Strong confidence boost for known-safe queries
        safe_patterns = [
            r"^select\s+\*\s+from\s+\w+\s+where\s+\w+\s*=\s*['\"].+['\"]$",
            r"^insert\s+into\s+\w+\s*\(.+\)\s+values\s*\(.+\)$",
            r"^update\s+\w+\s+set\s+.+\s+where\s+.+$",
            r"^delete\s+from\s+\w+\s+where\s+.+$"
        ]

        for pattern in safe_patterns:
            if re.fullmatch(pattern, cleaned, flags=re.IGNORECASE):
                pred = 0.02  # Treat it as very safe
                print("[Safe Pattern Detected] Forcing low confidence.")
                break


        # === Adjust prediction based on patterns ===
        if patterns:
            if pred < 0.9:
                pred = 0.95  # Boost confidence if any malicious pattern is found
        else:
            if 0.5 < pred < 0.9:
                pred = 1 - pred  # Reduce confidence if no patterns but model seems unsure

        # Final label decision
        if pred >= 0.9:
            label = "Injection Found"
        elif pred <= 0.1:
            label = "Legitimate Query"
        else:
            label = "Suspicious/Uncertain"


        return jsonify({
            'prediction': 1 if pred >= 0.9 else 0,  # 1 for malicious, 0 for safe
            'confidence': pred,
            'patterns': patterns,
            'preprocessed': cleaned
        })

    except Exception as e:
        print(f"[SQL Prediction Error] {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)