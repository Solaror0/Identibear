from flask import app, Flask, redirect, url_for, request, render_template, flash, config
import os
from werkzeug.utils import secure_filename
import jsonify
import model
import dataFormatter
import threading
import cv2
import tensorflow as tf
import keras
from gtts import gTTS
from playsound import playsound
import openai
import time


app = Flask(__name__)
openai.api_key = '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'
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}  # Add other video formats as needed
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

dictOfNames = {
0:{
    "name":"stranger",
    "relationship":"stranger",
    "dateOfMeeting":" ",
    "sigMemory": " "
},
1:{
    "name":"josh",
    "relationship":"friend",
    "dateOfMeeting":"2024/08/03",
    "sigMemory": "When he walked around drunk hitting people"
   
},
2:{
    "name":"andy",
    "relationship":"friend",
    "dateOfMeeting":"2024/08/03",
    "sigMemory": "When he stayed up all hackathon and couldn't speak coherently"
},
3:{
    "name":"junnur",
    "relationship":"friend",
    "dateOfMeeting":"2024/08/03",
    "sigMemory": "When he joined the tiktok rizz partay"
},
4:{
    "name":"james",
    "relationship":"friend",
    "dateOfMeeting":"2024/08/03",
    "sigMemory": "Spent the whole night learning how to get rizzy"
}
}

listOfNames = ["stranger","josh","andy","junnur","james"]
listOfPredicts = []


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def default():
    return "Welcome to flask server!"

def delayed_function():
    time.sleep(5)  # Delay for 5 seconds
    print("Function executed after delay")



def TTSandAPI(strings):
    toSpeechQuery = f"This is {strings[0]} your {strings[1]} and you met on {strings[2]}. A memory you share is {strings[3]}"
    text = gTTS(text=toSpeechQuery)
    text.save("audio.mp3")
    playsound("audio.mp3",True)
    os.remove('audio.mp3')

@app.route('/infoUpload', methods=['POST'])
def my_form_post():
    name = request.form.get('name', '')
    relationship = request.form.get('relationship', '')
    dateOfMeeting = request.form.get('dateOfMeeting',' ')
    sigMemory = request.form.get('sigMemory','')

    name = name.lower()
    relationship = relationship.lower()


    dictOfNames[dictOfNames.__len__] = {"name":name,"relationship":relationship,"dateOfMeeting":dateOfMeeting,"sigMemory":sigMemory}
    listOfNames.append(name)
    
    print(dictOfNames)
    print(listOfNames)

    train()
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

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return " " #jsonify({'message': 'File successfully uploaded'}), 200
    
    return jsonify({'error': 'File type not allowed'}), 400

def train():
    # Get the list of all files and directories
    videos = os.listdir("./uploads")
    for i in range(0,videos.__len__()):
        print(i)
        videos[i] = f"uploads/{videos[i]}"
        
    print(videos)
    dataFormatter.create_data(videos)
    print("BOOM BOOM " + os.path.join(os.getcwd(), 'train', 'images'))
    kmodel = model.train_model(os.path.join(os.getcwd(), 'train', 'images'), os.path.join(os.getcwd(), 'train', 'labels'))
    kmodel.save("model.keras")

    print(dictOfNames)
    
@app.route('/predict', methods = ['POST'])

def webcamPredict():
    # output = request.json
    # print(output["class"])

    output = request.form.get('class', '')
    print(output)
    
    listOfNames.append(output)
    listOfPredicts.append(output)
 
    print(listOfPredicts)
    if len(listOfPredicts)>1:
        print("if1")
        if (listOfPredicts[0] != listOfPredicts[len(listOfPredicts)-2]):
            print("if2")
            listOfPredicts.clear()
           

    if len(listOfPredicts) > 10 and not("0" in listOfPredicts):  
        personInfoDict = (dictOfNames[int(listOfPredicts[8])])
        name = personInfoDict["name"]
        relationship = personInfoDict["relationship"]
        dateOfMeeting = personInfoDict["dateOfMeeting"]
        sigMemory = personInfoDict["sigMemory"]
        print("woah")
        strings = [name,relationship,dateOfMeeting,sigMemory]
        TTSandAPI(strings)
        listOfPredicts.clear()

    return " "   

if __name__ == '__main__':
    app.run(debug=True)
