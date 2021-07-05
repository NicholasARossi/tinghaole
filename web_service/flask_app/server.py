import os
from flask_cors import CORS
from flask import Flask, make_response, request, current_app,jsonify,render_template

from codebase.prediction_service import ToneClassifier

#from core_software.service_layer.prediction_service import ToneClassifier
app = Flask(__name__)
# CORS(app,resources={r"/predict": {"origins": "*"}})
# app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"/predict": {"origins": "*"}})

#api = Api(app)

@app.route("/predict",methods=["POST","GET"])
def predict():
    if request.method == 'POST':
        # try:

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
        # except:
        #     data = {"tone": 'unrecognized tone'}

        return jsonify(data)
    else:
        data = {"tone":''}
        return jsonify(data)



if __name__ == '__main__':
    app.run(debug=True)
