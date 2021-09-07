import os
from werkzeug.utils import secure_filename

from flask import Flask
from flask import url_for, redirect, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

model = load_model("model_VGG16.h5")




def predict(img_path, model):

    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    ## Scaling
    x=x/255.
    x = np.expand_dims(x, axis=0)  
    pred = model.predict(x)
    pred = np.argmax(pred, axis=1)

    if pred == 0:
        pred = "Healthy!"
    elif pred == 1:
        pred = "Doubtfull!"
    elif pred == 2:
        pred = "Minimal!"
    elif pred == 3:
        pred = "Moderate!"
    elif pred == 4:
        pred = "Severe!"

    return pred   

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads/', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = predict(file_path, model)
        result=preds
        return result
    return 'Not Working'    

if __name__ == '__main__':

    app.run(port=5001,debug=False)



