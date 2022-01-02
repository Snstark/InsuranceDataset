from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('frontend.html')


@app.route('/predict', methods=['POST', 'GET'])
def result():
    age = float(request.form['age'])
    sex = float(request.form['sex'])
    bmi = float(request.form['bmi'])
    children = float(request.form['children'])
    smoker = float(request.form['smoker'])

    X = np.array([[age, sex, bmi, children, smoker]])
    model = pickle.load(open('insurance.pkl', 'rb'))
    y_prediction = model.predict(X)
    return jsonify({'Prediction': float(y_prediction)})


if __name__ == "__main__":
    app.run(debug=True, port=1230)
