from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os
from src.pipeline.predict_pipeline import PredictPipeline
application = Flask(__name__)
app = application
# Metrics container (computed at startup if model.pkl exists)
METRICS = None

def compute_metrics():
    """Load model.pkl and dataset, evaluate on a test split and populate METRICS."""
    global METRICS
    model_path = 'artifacts/model.pkl'
    preprocessorpath = 'artifacts/preprocessor.pkl'
    data_path = os.path.join('notebook', 'data', 'stud.csv')
    if not os.path.exists(model_path):
        METRICS = None
        return
    try:
        df = pd.read_csv(data_path)
        features = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course", "reading_score", "writing_score"]
        X = df[features]
        y = df['math_score'].astype(float)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(preprocessorpath, 'rb') as f:  
            prepro = pickle.load(f)
        scaled = prepro.transform(X_test)      
        preds = model.predict(scaled)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        METRICS = {
            'mae': round(float(mae), 2),
            'rmse': round(float(rmse), 2),
            'r2': round(float(r2), 3),
            'n_test': int(len(y_test))
        }
    except Exception as e:
        METRICS = {'error': str(e)}

# Compute metrics once at startup
compute_metrics()


@app.route('/')
def home():
    return render_template('index.html', metrics=METRICS)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            form = request.form
            X = pd.DataFrame([{
                "gender": form["gender"],
                "race_ethnicity": form["race_ethnicity"],
                "parental_level_of_education": form["parental_level_of_education"],
                "lunch": form["lunch"],
                "test_preparation_course": form["test_preparation_course"],
                "reading_score": float(form["reading_score"]),
                "writing_score": float(form["writing_score"])
            }])
            predpipe = PredictPipeline()
            
            prediction = predpipe.predict(X)[0]
            output = round(float(prediction), 2)

            return render_template('home.html', prediction=output, metrics=METRICS)

        except Exception as e:
            return render_template('home.html', prediction=f'Error: {str(e)}', metrics=METRICS)

    # GET -> show the form
    return render_template('home.html', metrics=METRICS)   

if __name__ == "__main__":
    app.run(host="0.0.0.0" , debug = True)