
import cv2
import pyrealsense2 as rs
import numpy as np
import pdb
from ultralytics import YOLO
import requests
import json
from cnocr import CnOcr
from PIL import ImageFont, ImageDraw, Image


model = YOLO('yolov8n.pt')

#打开 RealSense 摄像头
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 10)
pipeline.start(config)

def ocr_detect(img):
    ocr = CnOcr(det_model_name='en_PP-OCRv3_det', rec_model_name='en_PP-OCRv3')
    out = ocr.ocr(img)

    text = ''
    for i in range(len(out)):
        if out[i]['score'] > 0.5:
            coords = out[i]['position']
            minX = 100000
            maxX = -1
            minY = 100000
            maxY = -1
            for coord in coords:
                x = int(coord[0])
                y = int(coord[1])
                minX = minX if minX < x else x
                minY = minY if minY < y else y
                maxX = maxX if maxX > x else x
                maxY = maxY if maxY > y else y
            if len(out[i]['text']) <= 0:
                print("skip")
                continue
            text = text + out[i]['text'] + '\n'
    #print(text)
    return text

try:
    while True:
        frames = pipeline.wait_for_frames()
        # print(frames.frame_number)
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        if frames.frame_number % 6 == 0:
            frame_data = np.asanyarray(color_frame.get_data())
            results = model(frame_data)
            orig_img=results[0].orig_img

            text=ocr_detect(orig_img)

            cv2.imshow("Img", orig_img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            if key==ord("c"):
                try:
                    response = requests.post("http://10.21.4.51:5050/", json={"in_text": text})
                    if response.content:
                        cont = json.loads(response.content)
                        print("The total cost is: ", cont['prediction'])
                except:
                    print("Would you mind to run again?")

except KeyboardInterrupt:
    pass
finally:
    pipeline.stop()