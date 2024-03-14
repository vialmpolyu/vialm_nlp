from sentence_transformers import SentenceTransformer, util
import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import requests
import json
from utils import get_obj_list_from_cv
from utils import get_task_id
from utils import ocr_detect
from cnocr import CnOcr

# Initialize models
text_model = SentenceTransformer("../infer_api_client_dev/all-MiniLM-L6-v2")
print('text model loaded')
vision_model = YOLO('../infer_api_client_dev/yolov8n.pt')
print('vision model loaded')
ocr_model= CnOcr(det_model_name='en_PP-OCRv3_det', rec_model_name='en_PP-OCRv3')
print('ocr model loaded')

# Predefined queries
user_query_1 = "Can you describe how is the environment around me?"
user_query_2 = "What is the total cost on the receipt?"
query_embeddings = text_model.encode([user_query_1, user_query_2])



def req_desc():
    # 打开 RealSense 摄像头
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 10)
    pipeline.start(config)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            # print(frames.frame_number)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            if frames.frame_number % 6 == 0:
                frame_data = np.asanyarray(color_frame.get_data())
                results = vision_model(frame_data)
                obj_list = get_obj_list_from_cv(results)
                # print(obj_list)
                annotated_frame = results[0].plot()
                cv2.imshow("YOLOv8 Inference", annotated_frame)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    cv2.destroyWindow("YOLOv8 Inference")  # 销毁 YOLOv8 Inference 窗口
                    break
                elif key == ord("r"):
                    try:
                        response = requests.post("http://10.21.4.51:5000/", json=obj_list)
                        if response.content:
                            cont = json.loads(response.content)
                            print("The surrounding is: ", cont['prediction'])
                    except:
                        print("Would you mind to run again?")
                    cv2.destroyWindow("YOLOv8 Inference")
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()

def req_ocr():
    # 打开 RealSense 摄像头
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 10)
    pipeline.start(config)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            # print(frames.frame_number)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            if frames.frame_number % 6 == 0:
                frame_data = np.asanyarray(color_frame.get_data())
                results = vision_model(frame_data)
                orig_img = results[0].orig_img
                ocr_out= ocr_model.ocr(orig_img)
                text = ocr_detect(ocr_out)

                cv2.imshow("Img", orig_img)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    cv2.destroyWindow("Img")  # 销毁 YOLOv8 Inference 窗口
                    break
                if key == ord("c"):
                    try:
                        response = requests.post("http://10.21.4.51:5050/", json={"in_text": text})
                        if response.content:
                            cont = json.loads(response.content)
                            print("The total cost is: ", cont['prediction'])
                    except:
                        print("Would you mind to run again?")
                    cv2.imshow("Img", orig_img)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()

if __name__=="__main__":
    while True:
        input_text = input("Please describe your task, or 'exit' to quit: ")
        if input_text.lower() == 'exit':
            print("Exiting program.")
            break
        task_id = get_task_id(input_text, query_embeddings, text_model)
        if task_id==0:
            req_desc()
        elif task_id==1:
            req_ocr()