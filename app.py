from flask import Flask, jsonify, request
from classifier import  get_prediction
import numpy as np
import pandas as pd

app = Flask(__name__)

cv2 = pd.read_csv("labels.csv")["labels"]

@app.route("/predict-alphabet", methods=["POST"])
def predict_data():
 
  image = cv2.imdecode(np.fromstring(request.files.get("alphabet").read(), np.uint8), cv2.IMREAD_UNCHANGED)
  image = request.files.get("alphabet")
  prediction = get_prediction(image)
  return jsonify({
    "prediction": prediction
  }), 200

if __name__ == "__main__":
  app.run(debug=True)
