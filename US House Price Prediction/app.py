# app.py

from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from math import exp

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('linear_regression_model.pkl', 'rb'))

# Assuming X is the set of features used during training
# Update this according to your training data
X = pd.read_csv("E:\\Data Science Practice\\US House Price Prediction\\FinalDataSet.csv")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input features from the form
        features = [int(request.form['MSSubClass']),
                    float(request.form['LotFrontage']),
                    int(request.form['LotArea']),
                    int(request.form['OverallQual']),
                    int(request.form['OverallCond']),
                    int(request.form['YearBuilt']),
                    int(request.form['YearRemodAdd']),
                    float(request.form['MasVnrArea']),
                    int(request.form['BsmtFinSF1']),
                    int(request.form['BsmtFinSF2']),
                    int(request.form['BsmtUnfSF']),
                    int(request.form['TotalBsmtSF']),
                    int(request.form['1stFlrSF']),
                    int(request.form['2ndFlrSF']),
                    int(request.form['LowQualFinSF']),
                    int(request.form['GrLivArea']),
                    int(request.form['BsmtFullBath']),
                    int(request.form['BsmtHalfBath']),
                    int(request.form['FullBath']),
                    int(request.form['HalfBath']),
                    int(request.form['BedroomAbvGr']),
                    int(request.form['KitchenAbvGr']),
                    int(request.form['TotRmsAbvGrd']),
                    int(request.form['Fireplaces']),
                    float(request.form['GarageYrBlt']),
                    int(request.form['GarageCars']),
                    int(request.form['GarageArea']),
                    int(request.form['WoodDeckSF']),
                    int(request.form['OpenPorchSF']),
                    int(request.form['EnclosedPorch']),
                    int(request.form['3SsnPorch']),
                    int(request.form['ScreenPorch']),
                    int(request.form['PoolArea']),
                    int(request.form['MiscVal']),
                    int(request.form['MoSold']),
                    int(request.form['YrSold'])]

        # Create a DataFrame from the input features
        input_data = pd.DataFrame([features], columns=X.columns)

        # Make a prediction using the model
        prediction = model.predict(input_data)

        # Display the prediction on a new page or return as JSON
        return render_template('result.html', prediction=prediction[0])

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")