import numpy as np
from flask import Flask, request, jsonify, render_template
import re
import pickle

flask_app = Flask(__name__)
model = pickle.load(open("spam-sms-mnb-model.pkl","rb"))
trans = pickle.load(open('cv-transform.pkl','rb'))

@flask_app.route("/")
def Home():
    return render_template("login.html")

@flask_app.route("/mails", methods= ["POST"])
def mails():
    return render_template("index.html")

@flask_app.route("/messages", methods= ["POST"])
def messages():
   return render_template("index.html")

@flask_app.route("/predict", methods= ["POST"])
def predict():
    if request.method == 'POST':
     mail = request.form['mail']
     data = [mail]
     float_features = trans.transform(data).toarray()
     my_prediction = model.predict(float_features)
     return render_template("index.html", prediction = my_prediction)
    
if __name__ == "__main__":
    flask_app.run(debug=True)
