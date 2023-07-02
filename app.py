from flask import Flask, render_template, request, jsonify
import re
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

app = Flask(__name__)

# Load the model, tokenizer, and label encoder
model = tf.keras.models.load_model("model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def predict_url(url):
    max_length = 150
    sequence = tokenizer.texts_to_sequences([url])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    prediction_proba = model.predict(padded_sequence)[0][0]
    label = "phishing" if prediction_proba > 0.5 else "legitimate"
    prediction_percentage = prediction_proba * 100 if label == "phishing" else (1 - prediction_proba) * 100
    return label, prediction_percentage

def is_valid_url(url):
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_url = request.form['url']
        if is_valid_url(input_url):
            result, prediction_percentage = predict_url(input_url)
            return jsonify({
                'status': 'success',
                'message': f"The URL '{input_url}' is classified as {result} with a {prediction_percentage:.2f}% confidence."
            })
        else:
            return jsonify({'status': 'error', 'message': 'Invalid URL'})

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
