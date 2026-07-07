from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('sonar_model.pkl', 'rb'))
scaler = pickle.load(open('sonar_scaler.pkl', 'rb'))

# Label mapping — M=0, R=1
LABEL_MAP = {0: 'MINE', 1: 'ROCK'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        df = pd.read_csv(file, header=None)

        # Handle label column if present (61 columns)
        if df.shape[1] == 61:
            actual_labels = df.iloc[:, 60].tolist()
            df = df.iloc[:, :60]
        elif df.shape[1] == 60:
            actual_labels = ['-'] * len(df)
        else:
            return render_template('index.html',
                error=f"Invalid CSV! Expected 60 or 61 columns, got {df.shape[1]}.")

        # Assign feature names
        columns = [f'feature_{i}' for i in range(1, 61)]
        data_df = pd.DataFrame(df.values.astype(float), columns=columns)

        # Scale features
        data_scaled = scaler.transform(data_df)

        # Predict
        predictions = model.predict(data_scaled)
        probabilities = model.predict_proba(data_scaled)

        # Build results
        results = []
        for i in range(len(predictions)):
            pred = int(predictions[i])
            prob = probabilities[i]

            # M=0 MINE, R=1 ROCK
            if pred == 0:
                label = 'MINE'
                confidence = round(prob[0] * 100, 2)
                alert = 'danger'
            else:
                label = 'ROCK'
                confidence = round(prob[1] * 100, 2)
                alert = 'safe'

            # Check if prediction matches actual
            actual = str(actual_labels[i]).strip()
            if actual == '-':
                correct = None
            elif actual == 'M' and label == 'MINE':
                correct = True
            elif actual == 'R' and label == 'ROCK':
                correct = True
            else:
                correct = False

            results.append({
                'row': i + 1,
                'prediction': label,
                'confidence': confidence,
                'alert': alert,
                'actual': actual,
                'correct': correct
            })

        # Summary
        total = len(results)
        mines = sum(1 for r in results if r['prediction'] == 'MINE')
        rocks = total - mines
        
        # Accuracy if actual labels present
        labeled = [r for r in results if r['correct'] is not None]
        if labeled:
            correct_count = sum(1 for r in labeled if r['correct'])
            accuracy = round(correct_count / len(labeled) * 100, 2)
        else:
            accuracy = None

        return render_template('index.html',
                               results=results,
                               total=total,
                               mines=mines,
                               rocks=rocks,
                               accuracy=accuracy,
                               batch=True)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)