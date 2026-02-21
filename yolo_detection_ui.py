import cv2
import numpy as np
import time
import pyrealsense2 as rs
from ultralytics import YOLO
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import queue
from datetime import datetime
import webbrowser
import subprocess
import platform
import os
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraThread(threading.Thread):
    """Thread for camera capture and YOLO detection"""
    
    def __init__(self, model_path, frame_queue, control_queue):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.frame_queue = frame_queue
        self.control_queue = control_queue
        self.running = True
        
        # Camera setup
        self.pipeline = None
        self.config = None
        self.align = None
        self.model = None
        self.color_intrinsics = None
        self.depth_scale = None
        self.confidence = 0.3
        
        # Statistics
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
    def setup_camera(self):
        """Initialize RealSense camera"""
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            self.align = rs.align(rs.stream.color)
            
            profile = self.pipeline.start(self.config)
            
            # Get camera intrinsics
            color_stream = profile.get_stream(rs.stream.color)
            self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            # Get depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            logger.info("RealSense camera initialized successfully")
            logger.info(f"Depth scale: {self.depth_scale:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup camera: {e}")
            return False
    
    def setup_model(self):
        """Load YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            logger.info("YOLO model loaded successfully")
            logger.info(f"Available classes: {list(self.model.names.values())}")
            return True
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False
    
    def get_depth_at_pixel(self, depth_image, x, y, region_size=5):
        """Get depth at specific pixel with region averaging"""
        try:
            x = max(region_size, min(x, depth_image.shape[1] - region_size))
            y = max(region_size, min(y, depth_image.shape[0] - region_size))
            
            region = depth_image[y-region_size:y+region_size, x-region_size:x+region_size]
            valid_depths = region[region > 0]
            
            if len(valid_depths) == 0:
                return 0.0
            
            depth_units = np.median(valid_depths)
            depth_meters = depth_units * self.depth_scale
            
            return float(depth_meters)
            
        except Exception as e:
            logger.error(f"Failed to get depth: {e}")
            return 0.0
    
    def detect_objects(self, color_image, depth_image):
        """Run YOLO detection"""
        try:
            results = self.model(color_image, conf=self.confidence, verbose=False)
            
            detections = []
            for result in results:
                if hasattr(result, 'obb') and result.obb is not None and len(result.obb) > 0:
                    for obb in result.obb:
                        conf = obb.conf[0].cpu().numpy()
                        cls_id = int(obb.cls[0].cpu().numpy())
                        cls_name = self.model.names[cls_id]
                        
                        xywhr = obb.xywhr[0].cpu().numpy()
                        center_x = int(xywhr[0])
                        center_y = int(xywhr[1])
                        width = xywhr[2]
                        height = xywhr[3]
                        rotation = xywhr[4]
                        
                        obb_points = None
                        if hasattr(obb, 'xyxyxyxy'):
                            obb_points = obb.xyxyxyxy[0].cpu().numpy()
                        
                        x1 = int(center_x - width / 2)
                        y1 = int(center_y - height / 2)
                        x2 = int(center_x + width / 2)
                        y2 = int(center_y + height / 2)
                        
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
                
                elif hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = self.model.names[cls_id]
                        
                        x1, y1, x2, y2 = xyxy
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
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
            
            if detection['class_name'] == 'brown_block':
                color = (19, 69, 139)  # Brown
            else:
                color = (0, 255, 0)  # Green
            
            if 'obb_points' in detection and detection['obb_points'] is not None:
                points = detection['obb_points'].reshape((-1, 2)).astype(np.int32)
                cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
            else:
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            cv2.circle(image, (center_x, center_y), 5, color, -1)
            
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            if detection['depth'] > 0:
                label += f" | {detection['depth']:.3f}m"
            
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    def run(self):
        """Main thread loop"""
        if not self.setup_camera():
            self.frame_queue.put({'status': 'error', 'message': 'Failed to setup camera'})
            return
        
        if not self.setup_model():
            self.frame_queue.put({'status': 'error', 'message': 'Failed to load model'})
            return
        
        self.frame_queue.put({'status': 'ready'})
        
        try:
            while self.running:
                # Check for control commands
                try:
                    cmd = self.control_queue.get_nowait()
                    if cmd['type'] == 'confidence':
                        self.confidence = cmd['value']
                    elif cmd['type'] == 'stop':
                        break
                except queue.Empty:
                    pass
                
                # Get frames
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Run detection
                start_time = time.time()
                detections = self.detect_objects(color_image, depth_image)
                detection_time = (time.time() - start_time) * 1000
                
                # Draw detections
                display_image = color_image.copy()
                display_image = self.draw_detections(display_image, detections)
                
                # Add FPS
                current_time = time.time()
                if current_time - self.last_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_time = current_time
                else:
                    self.frame_count += 1
                
                status_text = f"FPS: {self.fps} | Conf: {self.confidence:.2f} | Time: {detection_time:.1f}ms"
                cv2.putText(display_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Send frame to queue
                data = {
                    'frame': display_image,
                    'detections': detections,
                    'fps': self.fps,
                    'detection_time': detection_time
                }
                
                try:
                    self.frame_queue.put(data, block=False)
                except queue.Full:
                    pass  # Skip frame if queue is full
                
        except Exception as e:
            logger.error(f"Camera thread error: {e}")
            self.frame_queue.put({'status': 'error', 'message': str(e)})
        finally:
            if self.pipeline:
                self.pipeline.stop()


class YOLODetectionUI:
    """Tkinter UI for YOLO Detection"""
    
    def __init__(self, root, model_path):
        self.root = root
        self.model_path = model_path
        self.root.title("YOLO Detection Interface")
        self.root.geometry("1100x600")
        
        # Queues for thread communication
        self.frame_queue = queue.Queue(maxsize=1)
        self.control_queue = queue.Queue()
        
        # State
        self.running = False
        self.camera_thread = None
        self.detection_history = []
        self.detection_data = []  # For JSON saving
        self.total_frames = 0
        self.total_detections = 0
        self.frames_with_detections = 0
        self.session_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup UI
        self.setup_ui()
        
        # Start update loop using after instead of blocking
        self.update_loop()
    
    def setup_ui(self):
        """Create UI components"""
        # Main container - no padding
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Left panel - Video display (fixed width)
        left_frame = ttk.Frame(main_frame, width=750)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=0, pady=0)
        left_frame.pack_propagate(False)  # Keep fixed size
        
        # Video label without title, stretches fully
        self.video_label = ttk.Label(left_frame, background="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Controls
        right_frame = ttk.Frame(main_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        right_frame.pack_propagate(False)
        
        # Control sections
        self.create_status_section(right_frame)
        self.create_control_section(right_frame)
        self.create_stats_section(right_frame)
        self.create_detection_log_section(right_frame)
    
    def create_status_section(self, parent):
        """Status information panel"""
        status_frame = ttk.LabelFrame(parent, text="Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Model info
        ttk.Label(status_frame, text=f"Model: {self.model_path.split('/')[-1]}", 
                 font=("Arial", 9)).pack(anchor=tk.W)
        
        self.status_label = ttk.Label(status_frame, text="Status: Idle", font=("Arial", 9))
        self.status_label.pack(anchor=tk.W)
        
        self.fps_label = ttk.Label(status_frame, text="FPS: 0", font=("Arial", 9))
        self.fps_label.pack(anchor=tk.W)
    
    def create_control_section(self, parent):
        """Control buttons and sliders"""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Confidence slider
        ttk.Label(control_frame, text="Confidence Threshold:", font=("Arial", 9)).pack(anchor=tk.W)
        
        conf_container = ttk.Frame(control_frame)
        conf_container.pack(fill=tk.X, pady=(0, 10))
        
        self.confidence_slider = ttk.Scale(conf_container, from_=0.1, to=0.9, orient=tk.HORIZONTAL,
                                          command=self.on_confidence_change)
        self.confidence_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.conf_label = ttk.Label(conf_container, text="0.30", font=("Arial", 9), width=4)
        self.conf_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Set slider value AFTER label is created
        self.confidence_slider.set(0.3)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_btn = ttk.Button(button_frame, text="Start", command=self.start_detection)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop", command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        
        self.save_btn = ttk.Button(button_frame, text="Save Frame", command=self.save_frame, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Clear stats button
        ttk.Button(control_frame, text="Clear Statistics", command=self.clear_stats).pack(fill=tk.X, pady=(0, 5))
        
        # Save detections to JSON button
        ttk.Button(control_frame, text="Save Detections JSON", command=self.save_detections_to_json).pack(fill=tk.X)
    
    def create_stats_section(self, parent):
        """Statistics display"""
        stats_frame = ttk.LabelFrame(parent, text="Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.total_frames_label = ttk.Label(stats_frame, text="Total Frames: 0", font=("Arial", 9))
        self.total_frames_label.pack(anchor=tk.W)
        
        self.frames_with_det_label = ttk.Label(stats_frame, text="Frames w/ Detections: 0", font=("Arial", 9))
        self.frames_with_det_label.pack(anchor=tk.W)
        
        self.det_rate_label = ttk.Label(stats_frame, text="Detection Rate: 0%", font=("Arial", 9))
        self.det_rate_label.pack(anchor=tk.W)
        
        self.total_det_label = ttk.Label(stats_frame, text="Total Detections: 0", font=("Arial", 9))
        self.total_det_label.pack(anchor=tk.W)
    
    def create_detection_log_section(self, parent):
        """Recent detections log"""
        log_frame = ttk.LabelFrame(parent, text="Recent Detections", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create text widget with scrollbar
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.detection_text = tk.Text(log_frame, height=10, width=30, font=("Courier", 8),
                                      yscrollcommand=scrollbar.set, state=tk.DISABLED)
        self.detection_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.detection_text.yview)
    
    def on_confidence_change(self, value):
        """Handle confidence slider change"""
        conf_value = float(value)
        self.conf_label.config(text=f"{conf_value:.2f}")
        
        if self.running:
            self.control_queue.put({'type': 'confidence', 'value': conf_value})
    
    def start_detection(self):
        """Start detection"""
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        self.confidence_slider.config(state=tk.NORMAL)
        
        self.status_label.config(text="Status: Initializing...")
        
        self.camera_thread = CameraThread(self.model_path, self.frame_queue, self.control_queue)
        self.camera_thread.start()
    
    def stop_detection(self):
        """Stop detection"""
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        
        if self.camera_thread:
            self.control_queue.put({'type': 'stop'})
            self.camera_thread.join(timeout=2)
        
        self.status_label.config(text="Status: Stopped")
        
        # Auto-save detections when stopping
        if self.detection_data:
            auto_save = messagebox.askyesno("Auto Save", f"Save {len(self.detection_data)} detections to JSON?")
            if auto_save:
                self.save_detections_to_json()
    
    def save_frame(self):
        """Save current frame"""
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'yolo_detection_ui_{timestamp}.jpg'
            cv2.imwrite(filename, self.current_frame)
            messagebox.showinfo("Success", f"Frame saved as: {filename}")
    
    def clear_stats(self):
        """Clear statistics"""
        self.total_frames = 0
        self.total_detections = 0
        self.frames_with_detections = 0
        self.detection_history = []
        self.update_stats()
        self.update_detection_log()
    
    def save_detections_to_json(self):
        """Save detection data to JSON file"""
        if not self.detection_data:
            logger.info("No detections to save")
            messagebox.showinfo("Info", "No detections to save")
            return
        
        try:
            filename = f'detections_{self.session_start_time}.json'
            with open(filename, 'w') as f:
                json.dump(self.detection_data, f, indent=2)
            logger.info(f"Detections saved to {filename}")
            messagebox.showinfo("Success", f"Detections saved to {filename}\nTotal: {len(self.detection_data)} detections")
        except Exception as e:
            logger.error(f"Failed to save detections: {e}")
            messagebox.showerror("Error", f"Failed to save detections: {e}")
    

    def update_stats(self):
        """Update statistics display"""
        self.total_frames_label.config(text=f"Total Frames: {self.total_frames}")
        self.frames_with_det_label.config(text=f"Frames w/ Detections: {self.frames_with_detections}")
        
        det_rate = 0 if self.total_frames == 0 else (self.frames_with_detections / self.total_frames * 100)
        self.det_rate_label.config(text=f"Detection Rate: {det_rate:.1f}%")
        self.total_det_label.config(text=f"Total Detections: {self.total_detections}")
    
    def update_detection_log(self):
        """Update detection history"""
        self.detection_text.config(state=tk.NORMAL)
        self.detection_text.delete(1.0, tk.END)
        
        for det in self.detection_history[-10:]:
            self.detection_text.insert(tk.END, det + "\n")
        
        self.detection_text.see(tk.END)
        self.detection_text.config(state=tk.DISABLED)
    
    def update_loop(self):
        """Non-blocking update loop for UI using after()"""
        # Check camera frame queue
        try:
            data = self.frame_queue.get_nowait()
            
            if 'status' in data:
                if data['status'] == 'ready':
                    self.status_label.config(text="Status: Running")
                elif data['status'] == 'error':
                    messagebox.showerror("Error", data['message'])
                    self.running = False
                    self.start_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
            else:
                # Update frame
                self.current_frame = data['frame']
                image_rgb = cv2.cvtColor(data['frame'], cv2.COLOR_BGR2RGB)
                
                # Get the label size
                label_width = self.video_label.winfo_width()
                label_height = self.video_label.winfo_height()
                
                # Only resize if label has a valid size
                if label_width > 1 and label_height > 1:
                    # Resize image to stretch and fill the entire label
                    image_pil = Image.fromarray(image_rgb)
                    image_pil = image_pil.resize((label_width, label_height), Image.Resampling.LANCZOS)
                    image_tk = ImageTk.PhotoImage(image_pil)
                    
                    self.video_label.config(image=image_tk)
                    self.video_label.image = image_tk
                
                # Update FPS
                self.fps_label.config(text=f"FPS: {data['fps']}")
                
                # Update statistics
                self.total_frames += 1
                detections = data['detections']
                
                if detections:
                    self.frames_with_detections += 1
                    self.total_detections += len(detections)
                    
                    for det in detections:
                        log_entry = f"{det['class_name']}: {det['confidence']:.2f} | {det['depth']:.3f}m"
                        self.detection_history.append(log_entry)
                        
                        # Save detection data to JSON
                        bbox = det['bbox']
                        height = bbox[3] - bbox[1]  # y2 - y1
                        detection_record = {
                            'timestamp': datetime.now().isoformat(),
                            'class_name': det['class_name'],
                            'confidence': float(det['confidence']),
                            'height': float(height),
                            'depth': float(det['depth']),
                            'bbox': {
                                'x1': int(bbox[0]),
                                'y1': int(bbox[1]),
                                'x2': int(bbox[2]),
                                'y2': int(bbox[3])
                            }
                        }
                        self.detection_data.append(detection_record)
                
                # Update displays every 10 frames
                if self.total_frames % 10 == 0:
                    self.update_stats()
                    self.update_detection_log()
        
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(50, self.update_loop)


def main():
    try:
        root = tk.Tk()
        model_path = r"C:\Projects\Robotic vision\Green_Box\train\weights\best.pt"
        ui = YOLODetectionUI(root, model_path)
        root.mainloop()
    except Exception as e:
        logger.error(f"Application failed: {e}")
        messagebox.showerror("Error", f"Application failed: {e}")


if __name__ == "__main__":
    main()
