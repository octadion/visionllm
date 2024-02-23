import streamlit as st
import tempfile
import cv2
import torch
from super_gradients.training import models
import numpy as np
import math
from PIL import Image
from gtts import gTTS
from app import *
def main():
    st.title("VisionLLM")
    st.sidebar.title("Settings")
    st.sidebar.subheader("Parameters")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 300px;
            margin-left: -300px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    app_mode = st.sidebar.selectbox("Pages", ["About", "VisionLLM"])

    if app_mode == "About":
        st.markdown("This project is about **Computer Vision with LLM and TTS**")
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 300px;
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 300px;
                margin-left: -300px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    elif app_mode == "VisionLLM":

        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 300px;
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 300px;
                margin-left: -300px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.sidebar.markdown('---')
        use_webcam = st.sidebar.checkbox('Use Webcam')
        st.sidebar.markdown('---')
        image_file_buffer = st.sidebar.file_uploader("Upload an Image", type = ["jpg", "png", "jpeg"])
        if image_file_buffer:
            image = Image.open(image_file_buffer)
            st.sidebar.image(image, caption='Uploaded Image', use_column_width=True)
            stframe = st.empty()
            st.markdown("<hr/>", unsafe_allow_html = True)
            st.title("Responses")
            response = load_yolonas_process_frame(image, stframe)
            output_filename = "runs/test/output_audio.wav"
            if response and isinstance(response, str):
                with st.spinner("Generating voice output... Please wait"):
                    speech = gTTS(text=response, lang="en", slow=False)
                    speech.save(output_filename)
                st.audio(output_filename, format='audio/wav', start_time=0)
            else:
                st.write("No text to speak.")
        else:
            st.write("No image uploaded")

        video_file_buffer = st.sidebar.file_uploader("Upload a Video", type = ["mp4", "avi", "mov", "asf"])
        tffile = tempfile.NamedTemporaryFile(suffix=".mp4", delete = False)
        if video_file_buffer:
            DEMO_VIDEO = "data/videos/vidd2.mp4"
            tffile = tempfile.NamedTemporaryFile(suffix=".mp4", delete = False)
            if use_webcam:
                tffile.name = 0
            else:
                tffile.write(video_file_buffer.read())
                demo_video = open(tffile.name, 'rb')
                demo_bytes = demo_video.read()
                st.sidebar.text("Input Video")
                st.sidebar.video(demo_bytes)
            stframe = st.empty()
            st.markdown("<hr/>", unsafe_allow_html = True)
            st.title("Responses")
            response = load_yolonas_process_each_frame(tffile.name, stframe)
            output_filename = "runs/test/output_audio.wav"
            if response and isinstance(response, str):
                with st.spinner("Generating voice output... Please wait"):
                    speech = gTTS(text=response, lang="en", slow=False)
                    speech.save(output_filename)
                st.audio(output_filename, format='audio/wav', start_time=0)
            else:
                st.write("No text to speak.")
        else:
            st.write("No video uploaded")
        
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
