from flask import Flask, request, jsonify, render_template
import numpy as np
import json
import pickle

# Load the saved Logistic Regression model
with open('Diabetes_Model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

# The route() function of the Flask class is a decorator
# which tells the application which URL should call
# the associated function
@app.route('/')
# '/' URL is bound with home() function
def home():
  return render_template('index.html')


@app.route('/predict', methods = ['POST'])
def predict():

  # Map "Yes" or "No" answers to numerical values
  Age = float(request.form['Age'])
  Gender = float(request.form['Gender'])
  Polyuria = float(request.form['Polyuria'])
  Polydipsia = float(request.form['Polydipsia'])
  Sudden_Weight_Loss = float(request.form['Sudden_Weight_Loss'])
  Weakness = float(request.form['Weakness'])
  Polyphagia = float(request.form['Polyphagia'])
  Genital_Thrush = float(request.form['Genital_Thrush'])
  Visual_Blurring = float(request.form['Visual_Blurring'])
  Itching = float(request.form['Itching'])
  Irritability = float(request.form['Irritability'])
  Delayed_Healing = float(request.form['Delayed_Healing'])
  Partial_Paresis = float(request.form['Partial_Paresis'])
  Muscle_Stiffness = float(request.form['Muscle_Stiffness'])
  Alopecia = float(request.form['Alopecia'])
  Obesity = float(request.form['Obesity'])

  # Create a list of features based on the user input
  prediction  = model.predict(np.array([[
      Age,
      Gender,
      Polyuria,
      Polydipsia,
      Sudden_Weight_Loss,
      Weakness,
      Polyphagia,
      Genital_Thrush,
      Visual_Blurring,
      Itching,
      Irritability,
      Delayed_Healing,
      Partial_Paresis,
      Muscle_Stiffness,
      Alopecia
  ]]))

  # Render the same template but pass the input value to be displayed
  return render_template('index_html', response = prediction)

  # # Return a result based on the prediction
  # if prediction[0] == 1:
  #   result = "You may have diabetes. Please consult a doctor"
  # else:
  #   result = "You are unlikely to have diabetes.  However, if you have concerns, consult a doctor."

  # return jsonify(result = result)

# main driver function
if __name__ == "__main__":
  # run() method of FLask class runs the application
  # on the local development server
  app.run(debug = True)