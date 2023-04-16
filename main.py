from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn import metrics
from feature import FeatureExtraction

warnings.filterwarnings('ignore')

with open("pickle/model.pkl", "rb") as f:
    xgboost = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        url = request.form["url"]
        
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)
        
        # _y_pred = xgboost.predict(x)[0]
        # _y_pro_phishing = xgboost.predict_proba(x)[0, 0]
        y_pro_non_phishing = xgboost.predict_proba(x)[0, 1]
        
        return render_template('index.html', xx=round(y_pro_non_phishing, 2), url=url)
    
    return render_template("index.html", xx=-1)

if __name__ == "__main__":
    app.run(debug=True,port=8000)
