from __future__ import division, print_function
import os
import numpy as np

# Keras

from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename # use to store the file name
# Flask utils
from flask import Flask, redirect, url_for, request, render_template


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/inc_v31.h5'

# Load your trained model
model = load_model(MODEL_PATH)

# print('Model loaded. Start serving...')

#print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(128, 128))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = x * 1./255
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x]) #Stack arrays in sequence vertically (row wise).
    res = np.argmax(model.predict(images)) #Returns the indices of the maximum values .
    return res

#@app.route(’/’) , where @app is the name of the object containing our Flask app
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__) # current working directory 
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model) # call prediction function

        disease_list=['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot ','Corn_(maize)___Common_rust','Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy']    

        result = disease_list[int(preds)]
        return result
    return None


if __name__ == '__main__':
    app.run(threaded=False,host='10.7.3.15', port=5000)
   
