
from flask import Flask, request,jsonify
import os
from core_software.service_layer.prediction_service import ToneClassifier
app = Flask(__name__)



@app.route("/predict",methods=["Post"])
def predict():

    # get audio file and save it
    audio_file = request.files["file"]
    file_name = "saved_file"
    audio_file.save(file_name)

    # invoke ting hao le service
    predictor=ToneClassifier()

    # make a prediction
    predicted_tone=predictor.predict(file_name)

    # remove audio file
    os.remove(file_name)

    # send back the predicted tone/etc as JSON
    data = {"tone":predicted_tone}
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=False)
