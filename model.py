import cv2
import torch
from super_gradients.training import models
import numpy as np
import math
import yaml
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from huggingface_hub import hf_hub_download

def video_detection(path_x):
    cap = cv2.VideoCapture(path_x)

    frame_width = int(cap.get(3))

    frame_height = int(cap.get(4))

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = models.get('yolo_nas_s', num_classes= 77, checkpoint_path='models/weights/ckpt_best2.pth').to(device)

    count = 0

    with open('data/data2.yaml', 'r') as file:
        data = yaml.safe_load(file)

    classNames = data.pop('names')

    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        count+=1
        class_final_names=[]
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
                class_name= classNames[classname]
                class_final_names.append(class_name)
                conf = math.ceil((confidence*100))/100
                label = f'{class_name}{conf}'
                print("Frame N", count, "", x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                t_size = cv2.getTextSize(label, 0, fontScale = 1, thickness=2)[0]
                c2 = x1+t_size[0], y1-t_size[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, [255, 144, 30], -1, cv2.LINE_AA)
                cv2.putText(frame, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType = cv2.LINE_AA)
                resize_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            lists = np.array(class_final_names)
            unique_list = np.unique(lists)
            objects_detected = ','.join(unique_list)
            print(objects_detected)
            yield frame, objects_detected
            #out.write(frame)
            #cv2.imshow("Frame", resize_frame)
            #if cv2.waitKey(1) & 0xFF==ord('1'):
            #    break
        else:
            break

#out.release()
#cap.release()
#cv2.destroyAllWindows()

def image_detection(path_x):
    image = cv2.imread(path_x)


    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = models.get('yolo_nas_s', num_classes= 77, checkpoint_path='models/weights/ckpt_best2.pth').to(device)
    with open('data/data2.yaml', 'r') as file:
        data = yaml.safe_load(file)

    classNames = data.pop('names')

    result = model.predict(image, conf=0.30)
    bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
    confidences = result.prediction.confidence
    labels = result.prediction.labels.tolist()
    class_final_names = []
    for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
        bbox = np.array(bbox_xyxy)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        classname = int(cls)
        class_name= classNames[classname]
        class_final_names.append(class_name)
        conf = math.ceil((confidence*100))/100
        label = f'{class_name}{conf}'
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)
        t_size = cv2.getTextSize(label, 0, fontScale = 1, thickness=2)[0]
        c2 = x1+t_size[0], y1-t_size[1] - 3
        cv2.rectangle(image, (x1, y1), c2, [255, 144, 30], -1, cv2.LINE_AA)
        cv2.putText(image, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType = cv2.LINE_AA)
    lists = np.array(class_final_names)
    unique_list = np.unique(lists)
    objects_detected = ','.join(unique_list)
    print(objects_detected)
    yield image, objects_detected