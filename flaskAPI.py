from flask import app, Flask, redirect, url_for, request, render_template, flash, config
import os
from werkzeug.utils import secure_filename
import jsonify
# import model.py
import cv2
import tensorflow as tf
import keras

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}  # Add other video formats as needed
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
listOfNames = ["stranger","andy","james","josh","junnur"]


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def default():
    return "Welcome to flask server!"

@app.route('/infoUpload', methods=['POST'])
def my_form_post():
    name = request.form.get('name', '')
    relationship = request.form.get('relationship', '')
    processedText = name.lower()+relationship.lower()
    listOfNames.append(processedText)
    print(listOfNames)
    return " "


def liveVideoInput():
    kmodel = keras.saving.load_model("model.keras")
    frame = cv2.VideoCapture(0)
    kmodel.predict(frame)
    list[kmodel.predict(frame)-1]

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'message': 'File successfully uploaded'}), 200
    
    return jsonify({'error': 'File type not allowed'}), 400

# @app.route('/train')
# def train():
#     model.keras.train()



if __name__ == '__main__':
    app.run(debug=True)
