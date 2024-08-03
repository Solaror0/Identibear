from flask import app, Flask, redirect, url_for, request, render_template, flash, config
import os
from werkzeug.utils import secure_filename
import jsonify
# import model.py
import cv2
import tensorflow as tf
import keras
from gtts import gTTS
from playsound import playsound
import openai


app = Flask(__name__)
# openai.api_key = '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'
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}  # Add other video formats as needed
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# dictOfNames = {
# 0:{
#     "name":"stranger",
#     "relationship":"stranger",
#     "dateOfMeeting":" ",
#     "sigMemory": " "
# },
# 1:{
#  "name":"andy",
#     "relationship":"friend",
#     "dateOfMeeting":"2024/08/03",
#     "sigMemory": " "
# },
# 2:{
#     "name":"josh",
#     "relationship":"friend",
#     "dateOfMeeting":"2024/08/03",
#     "sigMemory": " "
# },
# 3:{
#     "name":"james",
#     "relationship":"friend",
#     "dateOfMeeting":"2024/08/03",
#     "sigMemory": " "
# },
# 4:{
#     "name":"junnur",
#     "relationship":"friend",
#     "dateOfMeeting":"2024/08/03",
#     "sigMemory": " "
# }
# }


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
    dateOfMeeting = request.form.get('dateOfMeeting',' ')
    sigMemory = request.form.get('sigMemory','')

    name = name.lower()
    relationship = relationship.lower()
    print(name, relationship, dateOfMeeting, sigMemory)

    # dictOfNames[dictOfNames.__len__] = {"name":name,"relationship":relationship,"dateOfMeeting":dateOfMeeting,"sigMemory":sigMemory}
    # print(dictOfNames)

    # text = gTTS(text=name+" is your " + relationship,lang='en', slow=False)
    # text.save("audio.mp3")
    # playsound("audio.mp3",True)
    # os.remove('audio.mp3') # TTS TEST, save later

    # train()
    return " "


# def liveVideoInput():
#     kmodel = keras.saving.load_model("model.keras")
#     frame = cv2.VideoCapture(0)
#     kmodel.predict(frame)
#     list[kmodel.predict(frame)-1]

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # if file and allowed_file(file.filename):
        # filename = secure_filename(file.filename)
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # return jsonify({'message': 'File successfully uploaded'}), 200
    
    return " "
# def train():
#     kmodel = keras.saving.load_model("model.keras")
#     #kmodel.train()


if __name__ == '__main__':
    app.run(debug=True)
