import streamlit as st
import numpy as np
from ultralytics import YOLO
import tempfile
from sort import *
from webcam import web_cam_pred
from car_counter import car_counter

st.set_page_config(layout="centered")

st.markdown(
    """
    <h1 style='text-align: center;'>Real-Time Object Detection</h1>
    """,
    unsafe_allow_html=True
)

def load_model():
    return YOLO('src/components/yolo-weights/yolov8n.pt')
    
# to read the upload video
def get_temporary_file_path(uploaded_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.getvalue())
    temp_file_path = temp_file.name
    temp_file.close()
    return temp_file_path

def main():
    st.write('Option - 1:')
    use_webcam = st.button('Use Webcam')
    st.write('\n\n')
    st.write('\n\n')
    
    if use_webcam:
        web_cam_pred(model, 0)
    
    uploaded_file = st.file_uploader("Option - 2: Upload a video file", type=["mp4"])
    st.write('\n\n')
    st.write('\n\n')


    if uploaded_file is not None:
        temp_file_path = get_temporary_file_path(uploaded_file)
        if st.button("Start Counting"):
            car_count = car_counter(model, temp_file_path)
            st.write("Total Cars: ", car_count)
    
    st.write("Option - 3: Dont have a video to test? No worries")
    default = st.button("Use deafult video for testing")
    st.write('\n\n')
    st.write('\n\n')
    
    if default:
        video_file = 'src/components/car-counter/cars.mp4'
        car_count = car_counter(model, video_file)
        st.write("Total Cars: ", car_count)
            
if __name__ == "__main__":
    model = load_model()
    main()