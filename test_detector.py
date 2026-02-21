import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisionSystem:

    def __init__(self, model_path):
        self.pipeline = None
        self.align = None
        self.model = None

        self.intrinsics = None
        self.depth_scale = None

        self.model_path = model_path
        logger.info(f'VisionSystem initialized with model path: {model_path}')
        
    def setup_camera(self):
        import pyrealsense2 as rs

        logger.info('Setting up RealSense camera...')
        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        logger.debug('Configured color and depth streams')

        profile = self.pipeline.start(config)
        logger.info('Pipeline started successfully')

        self.align = rs.align(rs.stream.color)
        logger.debug('Alignment filter initialized')

        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        logger.info(f'Camera intrinsics retrieved: fx={self.intrinsics.fx}, fy={self.intrinsics.fy}')

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        logger.info(f'Depth scale: {self.depth_scale}')
        
    def setup_model(self):
        from ultralytics import YOLO
        logger.info(f'Loading YOLO model from: {self.model_path}')
        self.model = YOLO(self.model_path)
        logger.info('YOLO model loaded successfully')
    
    def get_frame(self):
        import numpy as np

        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            logger.warning('Missing color or depth frame')
            return None, None

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        logger.debug(f'Frame captured - Color shape: {color.shape}, Depth shape: {depth.shape}')

        return color, depth

