from flask import Flask, render_template, request, redirect, url_for
import subprocess
from reviewAnalysis.pipeline.prediction import PredictionPipeline
import nltk
nltk.download('stopwords')

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train")
def training():
    try:
        #subprocess.run(["python", "main.py"])
        subprocess.run(["dvc", "repro"])
        return "Training successful !!"
    except Exception as e:
        return f"Error Occurred! {e}"

@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        text = request.form["userid"]
        obj = PredictionPipeline()
        prediction = obj.predict(text)
        return render_template("index.html", prediction=prediction)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=8080)
    app.run(debug=True)
