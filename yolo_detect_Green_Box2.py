import os
import sys
import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs


# ---------------- ARGUMENTS ---------------- #

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--source', required=True)
parser.add_argument('--thresh', default=0.7)
parser.add_argument('--resolution', default=None)

args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution


# ---------------- MODEL LOAD ---------------- #

if not os.path.exists(model_path):
    print("ERROR: Model not found.")
    sys.exit(0)

model = YOLO(model_path)
labels = model.names


# ---------------- SOURCE TYPE ---------------- #

if 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])

elif 'realsense' in img_source:
    source_type = 'realsense'

elif os.path.isfile(img_source):
    source_type = 'video'

else:
    print("Invalid source.")
    sys.exit(0)


# ---------------- RESOLUTION ---------------- #

resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))


# ---------------- SOURCE INIT ---------------- #

if source_type == 'usb':

    cap = cv2.VideoCapture(usb_idx, cv2.CAP_DSHOW)
    time.sleep(2)

    if not cap.isOpened():
        print("ERROR: Camera not opened.")
        sys.exit(0)

elif source_type == 'video':

    cap = cv2.VideoCapture(img_source)

elif source_type == 'realsense':

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color,
        640, 480,
        rs.format.bgr8,
        30
    )
    pipeline.start(config)


# ---------------- COLORS ---------------- #

colors = [
    (164,120,87),(68,148,228),(93,97,209),
    (178,182,133),(88,159,106),
    (96,202,231),(159,124,168),
    (169,162,241),(98,118,150),
    (172,176,184)
]


# ---------------- LOOP ---------------- #

fps_buffer = []
fps_len = 50

while True:

    t1 = time.perf_counter()

    # -------- FRAME LOAD -------- #

    if source_type in ['usb','video']:

        ret, frame = cap.read()

        if not ret or frame is None:
            print("Frame read failed — retrying.")
            continue

    elif source_type == 'realsense':

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())


    # -------- RESIZE -------- #

    if resize:
        frame = cv2.resize(frame,(resW,resH))


    # -------- YOLO INFERENCE -------- #

    results = model(frame, verbose=False)

    count = 0

    if results and results[0].boxes is not None:

        for det in results[0].boxes:

            conf = float(det.conf.item())
            if conf < min_thresh:
                continue

            xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
            xmin,ymin,xmax,ymax = xyxy

            cls = int(det.cls.item())
            label = f"{labels[cls]} {int(conf*100)}%"

            color = colors[cls % 10]

            cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),color,2)

            cv2.putText(
                frame,label,(xmin,ymin-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,(0,0,0),1
            )

            count += 1


    # -------- FPS -------- #

    fps = 1/(time.perf_counter()-t1)

    fps_buffer.append(fps)
    if len(fps_buffer) > fps_len:
        fps_buffer.pop(0)

    avg_fps = np.mean(fps_buffer)


    cv2.putText(
        frame,
        f"FPS: {avg_fps:.2f}",
        (10,20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,(0,255,255),2
    )

    cv2.putText(
        frame,
        f"Objects: {count}",
        (10,45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,(0,255,255),2
    )


    # -------- DISPLAY -------- #

    cv2.imshow("YOLO Detection", frame)

    key = cv2.waitKey(30)
    if key == ord('q'):
        break


# ---------------- CLEANUP ---------------- #

if source_type in ['usb','video']:
    cap.release()

elif source_type == 'realsense':
    pipeline.stop()

cv2.destroyAllWindows()
