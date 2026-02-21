#import cv2
#import numpy as np
#import time
#import pyrealsense2 as rs
#from ultralytics import YOLO


# import logging


# import rtde_control
# import rtde_receive
# import time

# ROBOT_IP = "192.168.52.170"

# # Connect
# rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
# rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

# print("RTDE connected")

# # Read current TCP pose
# current_pose = rtde_r.getActualTCPPose()
# print("Current TCP pose:", current_pose)

####
# import pyrealsense2 as rs
# import numpy as np

# class Camera:
#     def __init__(self):
#         self.pipeline = rs.pipeline()
#         self.config = rs.config()
        
#         self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#         self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
#         self.align = rs.align(rs.stream.color)
        
#         self.intrinsics = None
#         self.depth_scale = None

#     def start(self):
#         profile = self.pipeline.start(self.config)

#         color_stream = profile.get_stream(rs.stream.color)
#         self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

#         depth_sensor = profile.get_device().first_depth_sensor()
#         self.depth_scale = depth_sensor.get_depth_scale()

#     def get_frame(self):
#         frames = self.pipeline.wait_for_frames()
#         aligned = self.align.process(frames)

#         color_frame = aligned.get_color_frame()
#         depth_frame = aligned.get_depth_frame()

#         if not color_frame or not depth_frame:
#             return None, None

#         color = np.asanyarray(color_frame.get_data())
#         depth = np.asanyarray(depth_frame.get_data())

#         return color, depth

#     def stop(self):
#         self.pipeline.stop()
      
####

from ultralytics import YOLO

class Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image, conf=0.5):
        results = self.model(image, conf=conf, verbose=False)

        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                detections.append({
                    "center": (cx, cy),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": float(box.conf[0])
                })

        return detections  

