from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import os

# Load the model
model = tf.keras.models.load_model("sentiment_rnn_model.h5")

# Load tokenizer
with open("tokenizer.json") as f:
    json_string = f.read()
    tokenizer = tokenizer_from_json(json_string)


max_len = 200

# FastAPI App
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def form_get():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sentiment Analyzer</title>
        <style>
            body { font-family: Arial; background: #f5f5f5; display: flex; justify-content: center; padding-top: 50px; }
            .container { background: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); width: 500px; }
            textarea { width: 100%; height: 150px; padding: 10px; margin-bottom: 20px; border-radius: 5px; border: 1px solid #ccc; }
            button { width: 100%; padding: 10px; font-size: 18px; background: #007BFF; color: #fff; border: none; border-radius: 5px; }
            button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>IMDB Review Sentiment Analysis</h2>
            <form action="/predict" method="post">
                <textarea name="text" placeholder="Enter your movie review here..." required></textarea>
                <button type="submit">Predict Sentiment</button>
            </form>
        </div>
    </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict_sentiment(request: Request, text: str = Form(...)):
    # Tokenize and pad the input text
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)[0][0]
    sentiment = "Positive" if pred > 0.5 else "Negative"

    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Result</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }}
        .card {{
            background-color: #fff;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.1);
            width: 500px;
        }}
        .card h2 {{
            color: #333;
        }}
        .card p {{
            background: #f1f1f1;
            padding: 15px;
            border-radius: 5px;
            font-style: italic;
        }}
        .result {{
            margin-top: 20px;
            font-size: 18px;
        }}
        .result b {{
            color: {'green' if sentiment == 'Positive' else 'red'};
        }}
        .back {{
            display: inline-block;
            margin-top: 25px;
            padding: 10px 20px;
            background: #007BFF;
            color: #fff;
            text-decoration: none;
            border-radius: 5px;
        }}
        .back:hover {{
            background-color: #0056b3;
        }}
    </style>
</head>
<body>
    <div class="card">
        <h2>Review Submitted:</h2>
        <p>{text}</p>
        <div class="result">
            <h3>Predicted Sentiment: <b>{sentiment}</b> <small>({pred:.2f})</small></h3>
        </div>
        <a href="/" class="back">← Go Back</a>
    </div>
</body>
</html>
"""

