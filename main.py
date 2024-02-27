from flask import Flask,  Response, jsonify, request, send_file
from model import video_detection, image_detection
import cv2
from werkzeug.utils import secure_filename
import os
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from huggingface_hub import hf_hub_download
from gtts import gTTS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'visionllm'
app.config['VIDEO_UPLOAD_FOLDER'] = 'data/videos'
app.config['IMAGE_UPLOAD_FOLDER'] = 'data/images'

objects = None
response = None
def generate_frames(path_x = ''):
    yolo_output = video_detection(path_x)
    for detection_, objects_detected in yolo_output:
        ret, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        global objects
        objects = str(objects_detected)
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

def generate_frames_image(path_x=''):
    yolo_output = image_detection(path_x)
    for detection_, objects_detected in yolo_output:
        ret, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        global objects
        objects = str(objects_detected)
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')


@app.route('/')
def home():
    return "Welcome"

@app.route('/video', methods = ['POST'])
def video():
    vid = request.files['file']
    filename = secure_filename(vid.filename)
    vid.save(os.path.join(app.config['VIDEO_UPLOAD_FOLDER'], filename))
    return Response(generate_frames(path_x=os.path.join(app.config['VIDEO_UPLOAD_FOLDER'], filename)), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/image', methods = ['POST'])
def image():
    img = request.files['file']
    filename = secure_filename(img.filename)
    img.save(os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], filename))
    return Response(generate_frames_image(path_x=os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], filename)), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_objects', methods = ['GET'])
def get_objects():
    global objects
    return jsonify(result=objects)

@app.route('/llm')
def llm():
    global response
    global objects
    n_gpu_layers = -1
    llm = LlamaCpp(
        streaming = True,
        model_path='./app/models/llm/zephyr-7b-beta.Q4_K_S.gguf',
        temperature=0.1,
        top_p=1,
        n_gpu_layers=n_gpu_layers, 
        verbose=False,
        # n_ctx=4096
        )
    template = """You are a language model designed to provide instructions, 
    directions, and descriptions based on the given input to assist blind people. 
    Your goal is to generate responses that contain instructions, directions, 
    and descriptions of the surrounding environment based on the given input objects. 
    Always respond in English, use clear words, be capable of guiding, be capable of 
    describing surroundings, and make blind people understand. Use 'you' as a main focus of describing.
    Do not create responses out of context.
    {question}
    Responses:
    """
    prompt = PromptTemplate.from_template(template)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run(objects)
    return jsonify(result=response)

@app.route('/get_response', methods = ['GET'])
def get_response():
    global response
    return jsonify(result=response)

@app.route('/gtts')
def gtts():
    global response
    speech = gTTS(text=response, lang='en', slow=False)
    speech.save('./runs/test/output_audio.wav')
    return send_file('./runs/test/output_audio.wav', mimetype='audio/wav')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5000')
