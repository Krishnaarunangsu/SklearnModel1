# https://www.geeksforgeeks.org/deploy-machine-learning-model-using-flask/?ref=rp
# https://www.geeksforgeeks.org/learning-model-building-scikit-learn-python-machine-learning-library/
# https://towardsdatascience.com/how-to-deploy-machine-learning-models-as-a-microservice-using-fastapi-b3a6002768af
# importing libraries
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

# creating instance of the class
app = Flask(__name__)


# to tell flask what url should trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')
    # return "Hello World"


# prediction function
def predict_income(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 12)
    # loaded_model = pickle.load(open("model.pkl", "rb"))
    # pickle.load(open("adults_dtree.pkl", "rb"))

    # read data from a file
    with open('myfile.pickle', "rb") as fin:
        loaded_model = pickle.load(fin)

    result_prediction = loaded_model.predict(to_predict)
    return result_prediction[0]


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result_pred = predict_income(to_predict_list)

        if int(result_pred) == 1:
            prediction = 'Income more than 50K'
        else:
            prediction = 'Income less that 50K'

        return render_template("result.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
