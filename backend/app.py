from flask import Flask, request, jsonify, render_template
import pandas as pd
import sqlite3
from flask_cors import CORS
from datetime import datetime
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model", "ids_model.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "model", "feature_names.pkl"))

# -----------------------------
# create flask app
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# database helpers
# -----------------------------

def get_db():
    return sqlite3.connect("ids_logs.db")

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

# initialize db on startup
init_db()

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/logs-page')
def logs_page():
    return render_template('logs.html')

@app.route('/about')
def about():
    return render_template('about.html')

# -----------------------------
# prediction API
# -----------------------------

@app.route('/predict', methods=['POST'])
def predict():

    # request must be json
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.json

    # convert input to dataframe
    df_input = pd.DataFrame([data])

    # align input with training features (fill missing with 0)
    df_input = df_input.reindex(columns=feature_names, fill_value=0)

    # model prediction
    prediction = model.predict(df_input)[0]
    result = "ATTACK" if prediction == 1 else "NORMAL"


    # store result in database
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO logs (prediction, timestamp) VALUES (?, ?)",
        (result, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

    return jsonify({"prediction": result})

# -----------------------------
# logs API
# -----------------------------

@app.route('/logs')
def logs():
    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        "SELECT prediction, timestamp FROM logs ORDER BY id DESC LIMIT 20"
    )
    rows = cur.fetchall()
    conn.close()

    data = []
    for r in rows:
        data.append({
            "prediction": r[0],
            "timestamp": r[1]
        })

    return jsonify(data)

# -----------------------------
# run server
# -----------------------------

if __name__ == '__main__':
    app.run(debug=True)
