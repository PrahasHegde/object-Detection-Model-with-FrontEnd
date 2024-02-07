from flask import Flask, render_template, Response
import cv2
import torch

app = Flask(__name__)
cap = cv2.VideoCapture(0)

# Load YOLOv8 model
model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s', pretrained=True)
model = model.autoshape()  # Autoshape input to CUDA FP16

# Set the desired window size
window_width = 1150
window_height = 650

import cv2
import torch

# Load YOLOv8 model
model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s', pretrained=True)
model = model.autoshape()  # Autoshape input to CUDA FP16

# Set the desired window size
window_width = 1150
window_height = 650

# Factor to reduce bounding box size
bounding_box_scale_factor = 0.8

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Perform inference
            results = model(frame)

            # Extract bounding box information
            bboxes = results.xyxy[0].cpu().numpy()

            # Iterate over detected objects
            for bbox in bboxes:
                class_id, confidence, x_min, y_min, x_max, y_max = map(int, bbox[:6])
                label = model.names[0]

                # Calculate new bounding box coordinates with reduced size
                box_width = x_max - x_min
                box_height = y_max - y_min
                x_min_new = int(x_min + box_width * (1 - bounding_box_scale_factor) / 2)
                y_min_new = int(y_min + box_height * (1 - bounding_box_scale_factor) / 2)
                x_max_new = int(x_max - box_width * (1 - bounding_box_scale_factor) / 2)
                y_max_new = int(y_max - box_height * (1 - bounding_box_scale_factor) / 2)

                # Draw bounding box with reduced size and label
                cv2.rectangle(frame, (x_min_new, y_min_new), (x_max_new, y_max_new), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x_min_new, y_min_new - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Resize the frame to the desired window size
            frame = cv2.resize(frame, (window_width, window_height))

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
