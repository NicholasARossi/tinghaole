
from flask import Flask, request,jsonify
import os
from core_software.service_layer.prediction_service import PredictMA
app = Flask(__name__)



@app.route("/predict",methods=["Post"])
def predict():
    pass

    # get audio file and save it
    audio_file = request.files["file"]
    file_name = "saved_file"
    audio_file.save(file_name)

    # invoke ting hao le service
    predictor=PredictMA()
    predicted_phoneme=predictor.predict()

    # make a prediction

    # remove audio file
    os.remove(file_name)

    # send back the predicted tone/etc as JSON
    data = {"keyword":predicted_phoneme}
    return jsonify(data)



if __name__ == '__main__':
    app.run(debug=False)