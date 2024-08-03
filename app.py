import streamlit as st
import requests
import base64

# open css file
with open('./style/styles.css') as f:
  st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("Identibear")

uploaded_file = st.file_uploader("Upload Video of Target Individual", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file:
    st.video(uploaded_file)

    file = {'file': uploaded_file}
    response = requests.post('http://127.0.0.1:5000/upload', files=file)

    if response.status_code == 200:
        st.success('File successfully uploaded to the Flask server.')
    else:
        st.error('Failed to upload file to the Flask server.')


col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Name of Individual")
    date = st.date_input("Date you met [OPTIONAL]:")

with col2:
    relationship = st.text_input("Relationship with Individual")
    sigMemory = st.text_input("A Significant Memory [OPTIONAL]:")

uploadButton = st.button("Upload")
if uploadButton:
    response = requests.post('http://127.0.0.1:5000/infoUpload',data={'name': name, 'relationship': relationship, 'dateOfMeeting':date,'sigMemory':sigMemory})

    if response.status_code == 200:
        st.success('File successfully uploaded to the Flask server.')
    else:
        st.error('Failed to upload file to the Flask server.')


def sidebar_bg(main_bg):
   main_bg_ext = 'png'
   st.markdown(
    f"""
     <style>
     .stApp, st-emotion-cache-h4xjwg ezrtsby2 {{
         background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
         background-size: cover;
         background-repeat: no-repeat; 
         background-position: center;
     }}
     </style>
     """,
    unsafe_allow_html=True,
)
bg = "./assets/dementia-25d.png"
sidebar_bg(bg)