from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")


@app.route("/")
def home():
    return "NBA Player Performance Prediction API is Running!"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    input_df = pd.DataFrame([data])

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    return jsonify({
        "Predicted_GmSc": float(prediction[0])
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)