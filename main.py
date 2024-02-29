from flask import Flask,  Response, jsonify, request, send_file
from model import video_detection, image_detection
import cv2
from werkzeug.utils import secure_filename
import os
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from huggingface_hub import hf_hub_download
from gtts import gTTS
import psycopg2
import boto3
from datetime import datetime
from dotenv import load_dotenv
import schedule
import time
import random
import threading

load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = 'visionllm'
app.config['VIDEO_UPLOAD_FOLDER'] = 'data/videos'
app.config['IMAGE_UPLOAD_FOLDER'] = 'data/images'

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS")
)
s3 = boto3.client('s3',
                  region_name=os.getenv("S3_REGION"),
                  endpoint_url=os.getenv("S3_ENDPOINT"),
                  aws_access_key_id=os.getenv("S3_KEY"),
                  aws_secret_access_key=os.getenv("S3_SECRET"))

objects = None
response = None
image_id = None
audio_id = None
video_id = None
text_id = None

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
    global video_id
    vid = request.files['file']
    filename = secure_filename(vid.filename)
    vid.save(os.path.join(app.config['VIDEO_UPLOAD_FOLDER'], filename))
    with open(os.path.join(app.config['VIDEO_UPLOAD_FOLDER'], filename), "rb") as data:
        s3.upload_fileobj(data, "results", "videos/" + filename)
    cur = conn.cursor()
    video_url = "https://visionllm.sgp1.digitaloceanspaces.com/results/videos/" + filename
    cur.execute("INSERT INTO videos (data) VALUES (%s) RETURNING id", (video_url,))
    video_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    return Response(generate_frames(path_x=os.path.join(app.config['VIDEO_UPLOAD_FOLDER'], filename)), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/image', methods = ['POST'])
def image():
    global image_id
    img = request.files['file']
    filename = secure_filename(img.filename)
    img.save(os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], filename))
    result_generator = generate_frames_image(path_x=os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], filename))
    for idx, result in enumerate(result_generator):
        result_filename = f"result_{idx}_" + filename
        with open(os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], result_filename), 'wb') as f:
            f.write(result.split(b'\r\n\r\n')[1])
        with open(os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], result_filename), "rb") as data:
            s3.upload_fileobj(data, "results", "images/" + result_filename)
        cur = conn.cursor()
        image_url = "https://visionllm.sgp1.digitaloceanspaces.com/results/images/" + result_filename
        cur.execute("INSERT INTO images (data, create_time) VALUES (%s, NOW()) RETURNING id", (image_url,))
        image_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        ext = os.path.splitext(filename)[1]
        if ext.lower() == '.png':
            mimetype = 'image/png'
        elif ext.lower() in ['.jpg', '.jpeg']:
            mimetype = 'image/jpeg'
        else:
            mimetype = 'application/octet-stream'
        return send_file(os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], result_filename), mimetype=mimetype)


@app.route('/get_objects', methods = ['GET'])
def get_objects():
    global objects, text_id
    cur = conn.cursor()
    cur.execute("INSERT INTO texts (object, create_time) VALUES (%s, NOW()) RETURNING id", (objects,))
    text_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    return jsonify(result=objects)

@app.route('/llm')
def llm():
    global response, objects
    n_gpu_layers = -1
    llm = LlamaCpp(
        streaming = True,
        model_path='models/llm/zephyr-7b-beta.Q4_K_S.gguf',
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
    global response, text_id
    cur = conn.cursor()
    cur.execute("UPDATE texts SET response = %s WHERE id = %s", (response, text_id))
    conn.commit()
    cur.close()
    return jsonify(result=response)

@app.route('/gtts')
def gtts():
    global response, audio_id
    speech = gTTS(text=response, lang='en', slow=False)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = "output_audio_" + timestamp + ".wav"
    speech.save('./runs/test/' + filename)
    with open('./runs/test/' + filename, "rb") as data:
        s3.upload_fileobj(data, "results", "audios/" + filename)
    audio_url = "https://visionllm.sgp1.digitaloceanspaces.com/results/audios/" + filename
    cur = conn.cursor()
    cur.execute("INSERT INTO audios (data, create_time) VALUES (%s, NOW()) RETURNING id", (audio_url,))
    audio_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    return send_file('./runs/test/' + filename, mimetype='audio/wav')

@app.route('/save_prediction')
def save_prediction():
    global image_id, audio_id, text_id
    cur = conn.cursor()
    if image_id is not None and text_id is not None and audio_id is not None:
        cur.execute("INSERT INTO predictions (image_id, text_id, audio_id, create_time) VALUES (%s, %s, %s, NOW())", (image_id, text_id, audio_id))
        conn.commit()
        cur.close()
        return jsonify(result="Prediction saved successfully")
    else:
        return jsonify(result="Error: image_id, text_id, or audio_id is None")
    
def read_random_file(bucket_name, folder_name):
    response = s3.list_objects(Bucket=bucket_name, Prefix=folder_name)
    files = [file['Key'] for file in response['Contents']]
    file_to_read = random.choice(files)
    return file_to_read

def job():
    global image_id, objects, text_id, response, audio_id
    # detection
    bucket_name = 'visionllm'
    folder_name = 'raw/images/'
    file_name = read_random_file(bucket_name, folder_name)
    obj = s3.get_object(Bucket=bucket_name, Key=file_name)
    file_content = obj['Body'].read()
    filename = secure_filename(file_name)
    with open(os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], filename), 'wb') as f:
        f.write(file_content)
    result_generator = generate_frames_image(path_x=os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], filename))
    for idx, result in enumerate(result_generator):
        result_filename = f"result_{idx}_" + filename
        with open(os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], result_filename), 'wb') as f:
            f.write(result.split(b'\r\n\r\n')[1])
        with open(os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], result_filename), "rb") as data:
            s3.upload_fileobj(data, "results", "images/" + result_filename)
        cur = conn.cursor()
        image_url = "https://visionllm.sgp1.digitaloceanspaces.com/results/images/" + result_filename
        cur.execute("INSERT INTO images (data, create_time) VALUES (%s, NOW()) RETURNING id", (image_url,))
        image_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
    # detected objects
    cur = conn.cursor()
    cur.execute("INSERT INTO texts (object, create_time) VALUES (%s, NOW()) RETURNING id", (objects,))
    text_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    # llm
    n_gpu_layers = -1
    llm = LlamaCpp(
        streaming = True,
        model_path='models/llm/zephyr-7b-beta.Q4_K_S.gguf',
        temperature=0.1,
        top_p=1,
        n_gpu_layers=n_gpu_layers, 
        verbose=False,
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
    # response
    cur = conn.cursor()
    cur.execute("UPDATE texts SET response = %s WHERE id = %s", (response, text_id))
    conn.commit()
    cur.close()
    # tts
    speech = gTTS(text=response, lang='en', slow=False)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = "output_audio_" + timestamp + ".wav"
    speech.save('./runs/test/' + filename)
    with open('./runs/test/' + filename, "rb") as data:
        s3.upload_fileobj(data, "results", "audios/" + filename)
    audio_url = "https://visionllm.sgp1.digitaloceanspaces.com/results/audios/" + filename
    cur = conn.cursor()
    cur.execute("INSERT INTO audios (data, create_time) VALUES (%s, NOW()) RETURNING id", (audio_url,))
    audio_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    cur = conn.cursor()
    if image_id is not None and text_id is not None and audio_id is not None:
        cur.execute("INSERT INTO predictions (image_id, text_id, audio_id, create_time) VALUES (%s, %s, %s, NOW())", (image_id, text_id, audio_id))
        conn.commit()
        cur.close()

def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(1)
    
if __name__ == "__main__":

    schedule.every().day.at("15:00").do(job)
    schedule_thread = threading.Thread(target=run_schedule)
    schedule_thread.start()
    app.run(host='0.0.0.0', port='8001')
