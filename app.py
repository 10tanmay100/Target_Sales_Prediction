from flask import Flask
import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from matplotlib.pyplot import pcolor
import pandas as pd

app=Flask(__name__,template_folder="templates")
cluster0=pickle.load(open("model_cluster0.pkl","rb"))
cluster1=pickle.load(open("model_cluster1.pkl","rb"))
scaler=pickle.load(open("preprocessed.pkl","rb"))

@app.route('/',methods=['POST', 'GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    '''
    For rendering results on HTML GUI
    '''
    a=request.form.get("type")
    b=request.form.get("Temperature")
    c=request.form.get("Fuel_Price")
    d=request.form.get("MarkDown1")
    e=request.form.get("MarkDown2")
    f=request.form.get("MarkDown3")
    g=request.form.get("MarkDown4")
    h=request.form.get("MarkDown5")
    i=request.form.get("cpi")
    j=request.form.get("Unemployment rate")
    k=request.form.get("Holiday")
    l=request.form.get("size")


    df=pd.DataFrame({"Type":[a],"Temperature":[b],"Fuel_Price":[c],"MarkDown1":[d],"MarkDown2":[e],"MarkDown3":[f],"MarkDown4":[g],"MarkDown5":[h],
    "CPI":[i],"Unemployment_Rate":[j],"Holiday":[k],"Size":[l]})
    # ans=cluster0.predict(df)
    return render_template('index.html', prediction_text='Sales Price is {}'.format(cluster0.predict(scaler.transform(df))))

if __name__=="__main__":
    app.run(debug=True)