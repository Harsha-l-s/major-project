from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Base directory for relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load CSV data (if needed)
df = pd.read_csv(os.path.join(BASE_DIR, 'hypothyroid.csv'))

# Load trained model and encoders
model = joblib.load(os.path.join(BASE_DIR, 'thyroid_model.pkl'))
label_encoders = joblib.load(os.path.join(BASE_DIR, 'label_encoders.pkl'))
feature_columns = joblib.load(os.path.join(BASE_DIR, 'model_columns.pkl'))

# Map prediction to advice/suggestions
suggestions = {
    'negative': "Your thyroid levels appear normal. Maintain a healthy lifestyle and regular check-ups.",
    'positive': "Potential thyroid disorder detected. Please consult an endocrinologist. General advice: Avoid high-iodine supplements, maintain a healthy diet, and monitor symptoms."
}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/new_predict', methods=['GET', 'POST'])
def new_predict():
    if request.method == 'POST':
        try:
            form_data = request.form.to_dict()
            input_data = pd.DataFrame([form_data])

            # Convert numeric columns to numeric
            numeric_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
            for col in numeric_cols:
                if col in input_data.columns:
                    input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0)

            # Encode categorical features
            for column in input_data.columns:
                if column in label_encoders and column not in numeric_cols:
                    try:
                        input_data[column] = label_encoders[column].transform(input_data[column])
                    except ValueError:
                        input_data[column] = label_encoders[column].transform(
                            [label_encoders[column].classes_[0]]
                        )

            # Ensure all required features are present
            for col in feature_columns:
                if col not in input_data.columns:
                    input_data[col] = 0  # Fill missing with default

            input_data = input_data.reindex(columns=feature_columns, fill_value=0)

            # Predict
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)

            # Get class labels
            class_names = label_encoders['binaryClass'].classes_
            prediction_label = class_names[prediction[0]]
            probabilities = {
                class_names[0]: float(probability[0][0]),
                class_names[1]: float(probability[0][1])
            }

            # Get suggestion based on prediction
            advice = suggestions.get(prediction_label.lower(), "Consult a doctor for more information.")

            return render_template('new_index.html', prediction=prediction_label, probabilities=probabilities, advice=advice)

        except Exception as e:
            return render_template('new_index.html', error=str(e))

    else:
        return render_template('new_index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form.to_dict()
        input_data = pd.DataFrame([form_data])

        numeric_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
        for col in numeric_cols:
            if col in input_data.columns:
                input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0)

        for column in input_data.columns:
            if column in label_encoders and column not in numeric_cols:
                try:
                    input_data[column] = label_encoders[column].transform(input_data[column])
                except ValueError:
                    input_data[column] = label_encoders[column].transform(
                        [label_encoders[column].classes_[0]]
                    )

        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data.reindex(columns=feature_columns, fill_value=0)

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        class_names = label_encoders['binaryClass'].classes_
        prediction_label = class_names[prediction[0]]

        result = {
            'prediction': prediction_label,
            'probabilities': {
                class_names[0]: float(probability[0][0]),
                class_names[1]: float(probability[0][1])
            },
            'advice': suggestions.get(prediction_label.lower(), "Consult a doctor for more information.")
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
