print('flask is running')
from flask import Flask, render_template,jsonify,request,app
from flask import Response
import pickle 
import numpy as np
import pandas as pd 
application = Flask(__name__)
app = application 

scaler = pickle.load(open(r'D:\Impact Batch\ML-Project-Diabetes-Prediction-LogisticRegression\model\StandardScaler.pkl','rb'))
model = pickle.load(open(r'D:\Impact Batch\ML-Project-Diabetes-Prediction-LogisticRegression\model\modelForPrediction','rb'))

## Route for Homepage
@app.route('/')
def index():
    return "Hii.. This is our Prediction Model for Diabetes Prediction"

## Route for Single data prediction 
@app.route('/predictdata',methods = ['GET','POST'])
def predict_datapoint():
    result ="" 
    
    if request.method == 'POST':
        
        Pregnancies = float(request.form.get("Pregnancies"))
        Glucose = float(request.form.get("Glucose"))
        BloodPressure = float(request.form.get("BloodPressure"))
        SkinThickness = float(request.form.get("SkinThickness"))
        Insulin = float(request.form.get("Insulin"))
        BMI = float(request.form.get("BMI"))
        DiabetesPedigreeFunction = float(request.form.get("DiabetesPedigreeFunction"))
        Age = float(request.form.get("Age"))

        new_data_scaled = scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict  = model.predict(new_data_scaled)
        if predict[0] == 1 :
            result = 'Diabetic'
        else : result = "Non-Diabetic"
        return render_template("single_prediction.html",result = result)
    else : return render_template('Home.html')

if __name__ == '__main__':
    app.run(host = "127.0.0.1",port= 5000)