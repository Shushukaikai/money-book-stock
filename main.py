from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# 定义模型结构
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# 初始化模型并加载参数
model = LSTMModel()
model.load_state_dict(torch.load("lstm_model_vt.pt", map_location=torch.device("cpu")))
model.eval()

@app.route("/")
def home():
    return "✅ 股票涨势评分系统已启动"

@app.route("/predict", methods=["GET"])
def predict():
    code = request.args.get("code", default="VT").upper()
    try:
        df = yf.download(code, period="6mo")
        close_prices = df["Close"].values.reshape(-1, 1)

        if len(close_prices) < 30:
            return jsonify({"error": "数据不足"})

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close_prices)

        latest_seq = scaled[-30:]
        input_tensor = torch.tensor(latest_seq.reshape(1, 30, 1), dtype=torch.float32)

        with torch.no_grad():
            prediction = model(input_tensor).item()

        last_real = scaled[-1][0]
        score = max(0.0, min(1.0, 0.5 + (prediction - last_real)))
        return jsonify({"code": code, "score": round(score, 3)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
