from flask.json.tag import JSONTag
from collaborative_filtering import get_recomm
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    try:
        username = request.form.get("username")
        prediction = get_recomm(username)

        print(prediction)
        # output = round(prediction[0], 2)
    except:
       prediction = ['Not found']

    return render_template('index.html', prediction_text=prediction)

@app.route('/results',methods=['POST'])
def results():
    try:
        data = request.get_json(force=True)
        prediction = model.predict([np.array(list(data.values()))])

        output = prediction[0]
    except:
       output = 'User Not found'
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)