from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import pickle


app = Flask(__name__, template_folder='templetes')
model = pickle.load(open("Price.pkl","rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('result.html', result='Predicted price range is $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
