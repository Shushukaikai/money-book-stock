from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import os

app = Flask(__name__, static_folder='static')

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

def dummy_predict(data):
    return float(np.clip(np.random.normal(0.5, 0.2), 0, 1))

@app.route("/predict", methods=["POST"])
def predict():
    symbol = request.json.get("symbol")
    try:
        df = yf.download(symbol, period="6mo")
        if len(df) < 30:
            return jsonify({"error": "股票数据不足"})

        df["return"] = df["Close"].pct_change()
        df["ma5"] = df["Close"].rolling(5).mean()
        df.dropna(inplace=True)

        features = ["Close", "return", "ma5"]
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df[features])
        window = 10
        seq = data[-window:]

        prob = dummy_predict(seq)
        return jsonify({"symbol": symbol, "score": round(prob, 3)})
    except Exception as e:
        return jsonify({"error": str(e)})

app.run(host="0.0.0.0", port=3000)
