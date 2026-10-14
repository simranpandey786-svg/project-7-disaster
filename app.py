"""Part 4 - Flask web app. Run: python app.py  (or: gunicorn -w 1 --threads 2 app:app)"""
from flask import Flask, jsonify, render_template, request

from predictors import available_predictors

app = Flask(__name__)
PREDICTORS = available_predictors()
MAX_CHARS = 1000
EXAMPLES = [
    "Forest fire near La Ronge Sask. Canada",
    "13,000 people receive #wildfires evacuation orders in California",
    "On plus side LOOK AT THE SKY LAST NIGHT IT WAS ABLAZE",
    "Crying out for more! Set me ablaze",
]


@app.get("/")
def index():
    return render_template("index.html", models=list(PREDICTORS), examples=EXAMPLES, max_chars=MAX_CHARS)


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text") or "").strip()
    if not text:
        return jsonify(error="Paste a tweet to classify."), 400
    if len(text) > MAX_CHARS:
        return jsonify(error=f"Keep the text under {MAX_CHARS} characters."), 400
    if not PREDICTORS:
        return jsonify(error="No trained model found. Run the training scripts, then restart the app."), 503

    results = []
    for name, model in PREDICTORS.items():
        prob = float(model.predict_proba([text])[0])
        results.append({"model": name, "probability": prob, "label": "disaster" if prob >= 0.5 else "not_disaster"})
    return jsonify(text=text, results=results)


@app.get("/healthz")
def health():
    return jsonify(status="ok", models=list(PREDICTORS))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
