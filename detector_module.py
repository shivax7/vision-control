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
    