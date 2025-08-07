from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load your trained models
models = {}
models['knn'] = pickle.load(open('KNN_model.pkl', 'rb'))
models['logistic_regression'] = pickle.load(open('Logistic.pkl', 'rb'))
models['decision_tree'] = pickle.load(open('DecisionTree_model.pkl', 'rb'))
models['random_forest'] = pickle.load(open('RandomForest_model.pkl', 'rb'))
models['naive_bayes'] = pickle.load(open('NaiveBayes_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')  # Ensure index.html is in templates folder

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        input_data = np.array([[
            float(data['DebtRatio']),
            int(data['OpenCreditLines']),
            int(data['RealEstateLoans']),
            float(data['MonthlyIncome']),
            int(data['Dependents']),
            int(data['Education']),
            int(data['RegionCentral']),
            int(data['RegionEast']),
            int(data['RegionNorth']),
            int(data['RegionWest'])
        ]])

        algorithm = data['algorithm']

        if algorithm not in models:
            return jsonify({'error': 'Invalid algorithm selected.'})

        model = models[algorithm]
        prediction = model.predict(input_data)[0]
        result = "✅ Good Customer" if prediction == 1 else "❌ Bad Customer"

        return jsonify({'result': result, 'algorithm': algorithm.title()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

