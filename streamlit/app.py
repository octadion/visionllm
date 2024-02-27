import cv2
import torch
from super_gradients.training import models
import numpy as np
import math
import streamlit as st
import yaml
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from huggingface_hub import hf_hub_download
from PIL import Image
# model_name_or_path = "TheBloke/zephyr-7B-beta-GGUF"
# model_basename = "zephyr-7b-beta.Q4_K_S.gguf"
# model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

n_gpu_layers = -1
def generate_frames(sign):
    llm = LlamaCpp(
        streaming = True,
        model_path='./models/llm/zephyr-7b-beta.Q4_K_S.gguf',
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
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
    return llm_chain.run(sign)

def load_yolonas_process_each_frame(video_name, stframe):
    cap = cv2.VideoCapture(video_name)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = models.get('yolo_nas_s', num_classes= 77, checkpoint_path='models/yolo-nas/ckpt_best2.pth')
    count = 0
    with open('data/config/data2.yaml', 'r') as file:
        data = yaml.safe_load(file)
    classNames = data.pop('names')
    out = cv2.VideoWriter('runs/test/output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    class_final_names = []
    while True:
        ret, frame = cap.read()
        count += 1
        if ret:
            result = model.predict(frame, conf=0.30)
            bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
            confidences = result.prediction.confidence
            labels = result.prediction.labels.tolist()
            for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
                bbox = np.array(bbox_xyxy)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                classname = int(cls)
                class_name = classNames[classname]
                class_final_names.append(class_name)
                conf = math.ceil((confidence*100))/100
                label = f'{class_name}{conf}'
                print("Frame N", count, "", x1, y1,x2, y2)
                t_size = cv2.getTextSize(label, 0, fontScale = 1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] -3
                cv2.rectangle(frame, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
                cv2.putText(frame, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType = cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                #resize_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            stframe.image(frame, channels='BGR', use_column_width=True)
            out.write(frame)
            #cv2.imshow("Frame", resize_frame)
            #if cv2.waitKey(1) & 0xFF == ord('1'):
            #    break
        else:
            break
    lists = np.array(class_final_names)
    unique_list = np.unique(lists)
    objects_detected = ','.join(unique_list)
    print(objects_detected)
    response = generate_frames(objects_detected)
    print(response)
    st.write(response)
    out.release()
    cap.release()
    return response
    #cv2.destroyAllWindows()


def load_yolonas_process_frame(input, stframe):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = models.get('yolo_nas_s', num_classes= 77, checkpoint_path='models/yolo-nas/ckpt_best2.pth')
    with open('data/config/data2.yaml', 'r') as file:
        data = yaml.safe_load(file)
    classNames = data.pop('names')
    class_final_names = []

    if isinstance(input, Image.Image):
        frame = np.array(input)
        result = model.predict(frame, conf=0.30)
        bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
        confidences = result.prediction.confidence
        labels = result.prediction.labels.tolist()
        for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
            bbox = np.array(bbox_xyxy)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            classname = int(cls)
            class_name = classNames[classname]
            class_final_names.append(class_name)
            conf = math.ceil((confidence*100))/100
            label = f'{class_name}{conf}'
            t_size = cv2.getTextSize(label, 0, fontScale = 1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] -3
            cv2.rectangle(frame, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            cv2.putText(frame, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType = cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        stframe.image(frame, channels='BGR', use_column_width=True)

    lists = np.array(class_final_names)
    unique_list = np.unique(lists)
    objects_detected = ','.join(unique_list)
    print(objects_detected)
    response = generate_frames(objects_detected)
    print(response)
    st.write(response)
    return response
