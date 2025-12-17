from flask import Flask, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("RF.pkl", "rb"))

@app.route("/")
def home():
    with open("index.html") as f:
        return f.read()

@app.route("/predict", methods=["POST"])
def predict():

    data = [
        float(request.form["f1"]),
        float(request.form["f2"]),
        float(request.form["f3"]),
        float(request.form["f4"]),
        float(request.form["f5"]),
        float(request.form["f6"]),
        float(request.form["f7"]),
        float(request.form["f8"]),
        float(request.form["f9"]),
        float(request.form["f10"])
    ]

    arr = np.array([data])
    pred = model.predict(arr)[0]

    result = "STARFORMING" if pred == 1 else "STARBURST"

    with open("inner_page.html") as f:
        html = f.read()

    return html.replace("{{result}}", result)

if __name__ == "__main__":
    app.run(debug=True)