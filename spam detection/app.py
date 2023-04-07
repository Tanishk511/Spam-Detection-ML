import numpy as np
from flask import Flask, request, jsonify, render_template
import re
import pickle

flask_app = Flask(__name__)
model = pickle.load(open("spam-sms-mnb-model.pkl","rb"))
cv = pickle.load(open('cv-transform.pkl','rb'))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods= ["POST"])
def predict():
    if request.method == 'POST':
     message = request.form['mail']
     data = [message]
     float_features = cv.transform(data).toarray()
     my_prediction = model.predict(float_features)
     return render_template("result.html", prediction = my_prediction)

if __name__ == "__main__":
    flask_app.run(debug=True)