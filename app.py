import streamlit as st
import requests
import base64
import os

# open css file
with open('./style/styles.css') as f:
  st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# the background
def set_background(main_bg):
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
set_background(bg)

st.title("IdentiBear")

# uploaded_file = st.file_uploader("Upload Video of Target Individual", type=["mp4", "mov", "avi", "mkv"])
# if uploaded_file:
#     st.video(uploaded_file)

if 'collection_count' not in st.session_state:
    st.session_state.collection_count = 0
    st.session_state.collection_dict = {}

def render_collections(count):
    for i in range(count):
        col1, col2 = st.columns(2)
        uploaded_file = st.file_uploader("Upload Video of Target Individual (REQUIRED)", type=["mp4", "mov", "avi", "mkv"], key=f'file_{i}')
        st.session_state.collection_dict[f'file_{i}'] = uploaded_file

        with col1:
            name = st.text_input(f"Name of Individual (REQUIRED):", key=f'name_{i}')
            date = st.date_input(f"Date you met (OPTIONAL):", key=f'date_{i}')
        with col2:
            relationship = st.text_input(f"Relationship with Individual (REQUIRED):", key=f'relationship_{i}')
            sigMemory = st.text_input(f"A Significant Memory (OPTIONAL):", key=f'sigMemory_{i}')
        
        st.session_state.collection_dict[f'name_{i}'] = name
        st.session_state.collection_dict[f'date_{i}'] = date
        st.session_state.collection_dict[f'relationship_{i}'] = relationship
        st.session_state.collection_dict[f'sigMemory_{i}'] = sigMemory
        st.write("---")

button_col1, button_col2 = st.columns([1, 0.5])

with button_col1:
    if st.button("Add new collection +"):
        st.session_state.collection_count += 1

with button_col2:
    if st.button("Remove collection -"):
        if st.session_state.collection_count > 1:
            st.session_state.collection_count -= 1

render_collections(st.session_state.collection_count)

uploadButton = st.button("Upload All")
cameraButton = st.button("Activate Bear")

if cameraButton:
    os.system("python webcam.py")
    st.write("Script has been run.")

if uploadButton:
     for i in range(st.session_state.collection_count):
        name = st.session_state.collection_dict.get(f'name_{i}')
        relationship = st.session_state.collection_dict.get(f'relationship_{i}')
        date = st.session_state.collection_dict.get(f'date_{i}')
        sigMemory = st.session_state.collection_dict.get(f'sigMemory_{i}')

        # req fields
        if not name or not relationship:
            st.error(f"Missing required fields for Collection {i+1}. Please fill all required fields.")
            continue
        
        response = requests.post('http://127.0.0.1:5000/infoUpload', data={'name': name, 'relationship': relationship, 'dateOfMeeting': date, 'sigMemory': sigMemory})
        if response.status_code == 200:
            st.success(f'Information for Collection {i+1} successfully uploaded to the Flask server.')
        else:
            st.error(f'Failed to upload information for Collection {i+1} to the Flask server.')

        uploaded_file = st.session_state.collection_dict.get(f'file_{i}')
        if uploaded_file:
            files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post('http://127.0.0.1:5000/upload', files=files)

            if response.status_code == 200:
                st.success('Video file successfully uploaded to the Flask server.')
            else:
                st.error('Failed to upload video file to the Flask server.')
        else:
            st.error("Please upload file")
