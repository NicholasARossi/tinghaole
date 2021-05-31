
import os
from flask_cors import CORS
from flask_restful import Api
from flask import Flask, make_response, request, current_app,jsonify,render_template
from datetime import timedelta
from functools import update_wrapper

from core_software.service_layer.prediction_service import ToneClassifier
app = Flask(__name__)
# CORS(app,resources={r"/predict": {"origins": "*"}})
# app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"/predict": {"origins": "http://0.0.0.0:800"}})
cors = CORS(app, resources={r"/": {"origins": "http://0.0.0.0:800"}})

#api = Api(app)

@app.route("/predict",methods=["POST","GET"])
def predict():

    # get audio file and save it
    audio_file = request.files["audio_data"]
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
    # print(jsonify(data))
    return jsonify(data)

@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        f = open('./file.wav', 'wb')
        f.write(request.get_data("audio_data"))
        f.close()
        if os.path.isfile('./file.wav'):
            print("./file.wav exists")

        return render_template('index.html', request="POST")
    else:
        return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=False)
