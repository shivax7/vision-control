import cv2
import numpy as np
import time
import pyrealsense2 as rs
from ultralytics import YOLO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOTester:
    def __init__(self):
        # Initialize RealSense camera
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable streams
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Alignment object
        self.align = rs.align(rs.stream.color)
        
        # Load YOLO model
        try:
            self.model = YOLO(r"C:\Projects\Robotic vision\Green_Box\train\weights\best.pt")
            logger.info("YOLO model loaded successfully")
            logger.info(f"Available classes: {list(self.model.names.values())}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
        
        # Camera intrinsics (will be set when camera starts)
        self.color_intrinsics = None
        self.depth_scale = None
    
    def start_camera(self):
        """Start RealSense camera"""
        try:
            # Start pipeline
            profile = self.pipeline.start(self.config)
            
            # Get camera intrinsics
            color_stream = profile.get_stream(rs.stream.color)
            self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            # Get depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            logger.info("RealSense camera started successfully")
            logger.info(f"Depth scale: {self.depth_scale:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            return False
    
    def get_depth_at_pixel(self, depth_image, x, y, region_size=5):
        """Get depth at specific pixel with region averaging"""
        try:
            # Ensure coordinates are within bounds
            x = max(region_size, min(x, depth_image.shape[1] - region_size))
            y = max(region_size, min(y, depth_image.shape[0] - region_size))
            
            # Extract region around pixel
            region = depth_image[y-region_size:y+region_size, x-region_size:x+region_size]
            
            # Filter out zero values
            valid_depths = region[region > 0]
            
            if len(valid_depths) == 0:
                return 0.0
            
            # Use median for robustness
            depth_units = np.median(valid_depths)
            depth_meters = depth_units * self.depth_scale
            
            return float(depth_meters)
            
        except Exception as e:
            logger.error(f"Failed to get depth: {e}")
            return 0.0
    
    def detect_objects(self, color_image, depth_image, confidence=0.3):
        """Run YOLO detection on image"""
        try:
            # Run YOLO inference
            results = self.model(color_image, conf=confidence, verbose=False)
            
            detections = []
            for result in results:
                # Check for OBB (Oriented Bounding Boxes) first
                if hasattr(result, 'obb') and result.obb is not None and len(result.obb) > 0:
                    for obb in result.obb:
                        # Get detection data from OBB
                        conf = obb.conf[0].cpu().numpy()
                        cls_id = int(obb.cls[0].cpu().numpy())
                        cls_name = self.model.names[cls_id]
                        
                        # Get rotated bounding box coordinates
                        xywhr = obb.xywhr[0].cpu().numpy()  # [center_x, center_y, width, height, rotation]
                        center_x = int(xywhr[0])
                        center_y = int(xywhr[1])
                        width = xywhr[2]
                        height = xywhr[3]
                        rotation = xywhr[4]  # rotation in radians
                        
                        # Get OBB corner points (if available)
                        obb_points = None
                        if hasattr(obb, 'xyxyxyxy'):
                            obb_points = obb.xyxyxyxy[0].cpu().numpy()  # 4 corner points
                        
                        # Calculate approximate bounding box for display
                        x1 = int(center_x - width / 2)
                        y1 = int(center_y - height / 2)
                        x2 = int(center_x + width / 2)
                        y2 = int(center_y + height / 2)
                        
                        
                        # Get depth at object center
                        depth = self.get_depth_at_pixel(depth_image, center_x, center_y)
                        
                        detection = {
                            'class_id': cls_id,
                            'class_name': cls_name,
                            'confidence': float(conf),
                            'bbox': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            'depth': depth,
                            'obb_points': obb_points,
                            'rotation': rotation,
                            'width': width,
                            'height': height
                        }
                        detections.append(detection)
                # Fallback to regular boxes if available
                elif hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        # Get detection data
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = self.model.names[cls_id]
                        
                        x1, y1, x2, y2 = xyxy
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        # Get depth at object center
                        depth = self.get_depth_at_pixel(depth_image, center_x, center_y)
                        
                        detection = {
                            'class_id': cls_id,
                            'class_name': cls_name,
                            'confidence': float(conf),
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'center': (center_x, center_y),
                            'depth': depth
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def draw_detections(self, image, detections):
        """Draw detection results on image"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x, center_y = detection['center']
            
            # Choose color based on class
            if detection['class_name'] == 'brown_block':
                color = (19, 69, 139)  # Brown (BGR format)
            else:
                color = (0, 255, 0)  # Green
            
            # Draw oriented bounding box if available
            if 'obb_points' in detection and detection['obb_points'] is not None:
                # Draw rotated rectangle using corner points
                points = detection['obb_points'].reshape((-1, 2)).astype(np.int32)
                cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
            else:
                # Fallback to regular bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(image, (center_x, center_y), 5, color, -1)
            
            # Draw label with confidence and depth
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            if detection['depth'] > 0:
                label += f" | {detection['depth']:.3f}m"
            
            # Calculate text size and background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run_test(self):
        """Run YOLO detection test"""
        print("YOLO Detection Test")
        print("==================")
        print("Controls:")
        print("- 'q': Quit")
        print("- 's': Save current frame with detections")
        print("- 'c': Toggle confidence threshold (0.3/0.5/0.7)")
        print()
        
        if not self.start_camera():
            return
        
        confidence_levels = [0.3, 0.5, 0.7]
        current_conf_idx = 0
        current_confidence = confidence_levels[current_conf_idx]
        
        frame_count = 0
        detection_stats = {'total_frames': 0, 'frames_with_detections': 0, 'total_detections': 0}
        
        try:
            while True:
                # Get frames
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Run detection
                start_time = time.time()
                detections = self.detect_objects(color_image, depth_image, current_confidence)
                detection_time = (time.time() - start_time) * 1000  # ms
                
                # Update statistics
                detection_stats['total_frames'] += 1
                if detections:
                    detection_stats['frames_with_detections'] += 1
                    detection_stats['total_detections'] += len(detections)
                
                # Draw detections
                display_image = color_image.copy()
                self.draw_detections(display_image, detections)
                
                # Add status text
                status_text = f"Frame: {frame_count} | Conf: {current_confidence} | Time: {detection_time:.1f}ms"
                cv2.putText(display_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add detection info
                if detections:
                    y_offset = 60
                    for i, det in enumerate(detections[:3]):  # Show up to 3 detectionsc
                        det_text = f"{det['class_name']}: {det['confidence']:.2f} | Depth: {det['depth']:.3f}m"
                        cv2.putText(display_image, det_text, (10, y_offset + i*25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                else:
                    cv2.putText(display_image, "No objects detected", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Show image
                cv2.imshow('YOLO Detection Test', display_image)
                
                frame_count += 1
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f'yolo_detection_test_{timestamp}.jpg'
                    cv2.imwrite(filename, display_image)
                    print(f"Saved: {filename}")
                elif key == ord('c'):
                    current_conf_idx = (current_conf_idx + 1) % len(confidence_levels)
                    current_confidence = confidence_levels[current_conf_idx]
                    print(f"Confidence threshold changed to: {current_confidence}")
        
        finally:
            # Print statistics
            print("\nDetection Statistics:")
            print("====================")
            print(f"Total frames processed: {detection_stats['total_frames']}")
            print(f"Frames with detections: {detection_stats['frames_with_detections']}")
            print(f"Detection rate: {detection_stats['frames_with_detections']/detection_stats['total_frames']*100:.1f}%")
            print(f"Total detections: {detection_stats['total_detections']}")
            if detection_stats['frames_with_detections'] > 0:
                avg_detections = detection_stats['total_detections'] / detection_stats['frames_with_detections']
                print(f"Average detections per frame (when detected): {avg_detections:.1f}")
            
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("Test completed")

def main():
    try:
        tester = YOLOTester()
        tester.run_test()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    main()
    
    