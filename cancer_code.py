# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 12:20:19 2024

@author: user
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from flask import Flask, request, render_template
import re
import math

app = Flask("__name__")

q = ""

@app.route("/")
def loadPage():
    return render_template('home.html', query="")


@app.route("/", methods=['POST'])
def cancerPrediction():
    dataset_url = "https://raw.githubusercontent.com/apogiatzis/breast-cancer-azure-ml-notebook/master/breast-cancer-data.csv"
    df = pd.read_csv(dataset_url)

    df.info()

    inputQuery1 = float(request.form['query1'])
    inputQuery2 = float(request.form['query2'])
    inputQuery3 = float(request.form['query3'])
    inputQuery4 = float(request.form['query4'])
    inputQuery5 = float(request.form['query5'])

    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    train, test = train_test_split(df, test_size=0.2)

    features = ['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean']

    train_X = train[features]
    train_y = train.diagnosis

    test_X = test[features]
    test_y = test.diagnosis

    model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    model.fit(train_X, train_y)

    prediction = model.predict(test_X)
    accuracy = metrics.accuracy_score(test_y, prediction)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5]]

    new_df = pd.DataFrame(data, columns=features)
    single = model.predict(new_df)
    probability = model.predict_proba(new_df)[:, 1]
    print(probability)

    if single == 1:
        output = "The patient is diagnosed with Breast Cancer"
        output1 = "Confidence: {:.2f}%".format(probability[0] * 100)
    else:
        output = "The patient is not diagnosed with Breast Cancer"
        output1 = "Confidence: {:.2f}%".format((1 - probability[0]) * 100)

    return render_template('home.html', output1=output, output2=output1, query1=request.form['query1'], query2=request.form['query2'], query3=request.form['query3'], query4=request.form['query4'], query5=request.form['query5'])


app.run(debug=True)

