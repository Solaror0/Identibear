import streamlit as st
import requests

st.title("Title")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

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
    title = st.text_input("Person One:")
    st.write("The current person is", title)

with col2:
    title = st.text_input("Relationship One:")
    st.write("The current relationship is", title)