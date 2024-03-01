import streamlit as st
import requests
from PIL import Image
import io

def main():
    st.title("VisionLLM")
    st.sidebar.title("Settings")
    st.sidebar.subheader("Parameters")

    app_mode = st.sidebar.selectbox("Pages", ["About", "VisionLLM"])

    if app_mode == "About":
        st.markdown("This project is about **Computer Vision with LLM and TTS**")
    elif app_mode == "VisionLLM":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()

            files = {'file': img_bytes}
            
            response_image = requests.post('http://167.71.211.12:8001/image', files=files)
            response_objects = requests.get('http://167.71.211.12:8001/get_objects')
            response_llm = requests.get('http://167.71.211.12:8001/llm')
            response_get_response = requests.get('http://167.71.211.12:8001/get_response')
            response_gtts = requests.get('http://167.71.211.12:8001/gtts')
            response_save_prediction = requests.get('http://167.71.211.12:8001/save_prediction')

            st.image(response_image.content, caption='Response Image.')
            st.write('Objects:', response_objects.json())
            st.write('LLM:', response_llm.json())
            st.write('Get Response:', response_get_response.json())
            st.audio(response_gtts.content)
            st.write('Save Prediction:', response_save_prediction.json())

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
