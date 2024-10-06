from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import math
from sort import *
import threading
import queue

app = Flask(__name__)

# Create separate queues for each video feed
frame_queues = [queue.Queue(maxsize=10) for _ in range(4)]
data_queues = [queue.Queue(maxsize=1) for _ in range(4)]


class VehicleDetector:
    def __init__(self):
        self.model = YOLO("yolov8l.pt")
        self.tracker = Sort(max_age=10, min_hits=1, iou_threshold=0.1)
        self.total_count = []

        # Constants
        self.MAX_SIGNAL_TIME = 120
        self.MIN_SIGNAL_TIME = 30
        self.MAX_TRAFFIC = 50

        # Class names for detection
        self.class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck"]
        self.target_classes = ["car", "truck", "bus", "motorbike"]

        # Detection lines (will be initialized with frame dimensions)
        self.limit_lines = None
        self.vehicle_counts_history = []

    def initialize_lines(self, frame):
        height, width = frame.shape[:2]
        y1 = height * 0.6
        y2 = y1 + 20

        self.limit_lines = [
            [int(width * 0.2), int(y1), int(width * 0.8), int(y1)],
            [int(width * 0.2), int(y2), int(width * 0.8), int(y2)]
        ]

    def calculate_green_time(self, vehicle_count):
        density_factor = self.MAX_SIGNAL_TIME / self.MAX_TRAFFIC

        self.vehicle_counts_history.append(vehicle_count)
        if len(self.vehicle_counts_history) > 3:
            self.vehicle_counts_history.pop(0)

        predicted = (sum(self.vehicle_counts_history) / len(self.vehicle_counts_history)
                     if self.vehicle_counts_history else 0)

        green_time = max(self.MIN_SIGNAL_TIME,
                         min(density_factor * predicted, self.MAX_SIGNAL_TIME))

        return int(green_time), predicted

    def process_frame(self, frame):
        if self.limit_lines is None:
            self.initialize_lines(frame)

        results = self.model(frame, stream=True)
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                current_class = self.class_names[cls]

                if current_class in self.target_classes and conf > 0.3:
                    current_array = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_array))

        tracked_objects = self.tracker.update(detections)

        for limit in self.limit_lines:
            cv2.line(frame, (limit[0], limit[1]), (limit[2], limit[3]),
                     (250, 182, 122), 2)

        for result in tracked_objects:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2,
                              colorR=(111, 237, 235))
            cvzone.putTextRect(frame, f'#{int(id)}', (max(0, x1), max(35, y1)),
                               scale=2, thickness=1, offset=10,
                               colorR=(56, 245, 213), colorT=(25, 26, 25))

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(frame, (cx, cy), 5, (22, 192, 240), cv2.FILLED)

            for limit in self.limit_lines:
                if (limit[0] < cx < limit[2] and
                        limit[1] - 15 < cy < limit[1] + 15 and
                        id not in self.total_count):
                    self.total_count.append(id)
                    cv2.line(frame, (limit[0], limit[1]), (limit[2], limit[3]),
                             (12, 202, 245), 3)

        current_count = len(self.total_count)
        if current_count >= 2:
            green_time, predicted_traffic = self.calculate_green_time(current_count)

            cvzone.putTextRect(frame, f'Count: {current_count}', (20, 40),
                               scale=3, thickness=3, offset=10,
                               colorR=(147, 245, 186), colorT=(15, 15, 15))
            cvzone.putTextRect(frame, f'Predicted: {predicted_traffic:.1f}', (20, 90),
                               scale=2, thickness=2, offset=10,
                               colorR=(147, 245, 186), colorT=(15, 15, 15))
            cvzone.putTextRect(frame, f'Green Time: {green_time}s', (20, 140),
                               scale=2, thickness=2, offset=10,
                               colorR=(147, 245, 186), colorT=(15, 15, 15))
        else:
            cvzone.putTextRect(frame, f'Count: {current_count}', (20, 40),
                               scale=3, thickness=3, offset=10,
                               colorR=(147, 245, 186), colorT=(15, 15, 15))

        return frame, current_count


# Create detector instances for each feed
detectors = [VehicleDetector() for _ in range(4)]


def video_processing_thread(feed_id):
    # Different video sources for each feed
    video_sources = [
        "static/videos/vehical.mp4",
        "static/videos/matte.mp4",
        "static/videos/ramaiya.mp4",
        "static/videos/matte.mp4"
    ]

    cap = cv2.VideoCapture(video_sources[feed_id])

    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        processed_frame, count = detectors[feed_id].process_frame(frame)

        data = {
            "count": count
        }

        try:
            data_queues[feed_id].put(data, block=False)
        except queue.Full:
            try:
                data_queues[feed_id].get_nowait()
                data_queues[feed_id].put(data, block=False)
            except queue.Empty:
                pass

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        try:
            frame_queues[feed_id].put(frame_bytes, block=False)
        except queue.Full:
            try:
                frame_queues[feed_id].get_nowait()
                frame_queues[feed_id].put(frame_bytes, block=False)
            except queue.Empty:
                pass


@app.route('/')
def index():
    return render_template('index.html')


def generate_frames(feed_id):
    while True:
        frame_bytes = frame_queues[feed_id].get()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed/<int:feed_id>')
def video_feed(feed_id):
    if 0 <= feed_id < 4:
        return Response(generate_frames(feed_id),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Invalid feed ID", 404


@app.route('/get_data/<int:feed_id>')
def get_data(feed_id):
    if 0 <= feed_id < 4:
        try:
            data = data_queues[feed_id].get_nowait()
            return jsonify(data)
        except queue.Empty:
            return jsonify({"count": 0})
    return jsonify({"error": "Invalid feed ID"}), 404


if __name__ == '__main__':
    # Start video processing threads for each feed
    for i in range(4):
        threading.Thread(target=video_processing_thread, args=(i,), daemon=True).start()

    app.run(debug=True, threaded=True)