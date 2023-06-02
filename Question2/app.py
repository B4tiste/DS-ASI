import cv2
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

video_path = "../Sujet/voitures.mp4"

video_stream = cv2.VideoCapture(video_path)
skip_frames = 10

detections_infos = []


def process(frame):
    LABELS_FILE = '../Sujet/coco.names'
    CONFIG_FILE = '../Sujet/yolov3-tiny.cfg'
    WEIGHTS_FILE = '../Sujet/yolov3-tiny.weights'
    CONFIDENCE_THRESHOLD = 0.3
    H = None
    W = None
    LABELS = open(LABELS_FILE).read().strip().split("\n")
    np.random.seed(4)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    image = frame
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    if W is None or H is None:
        (H, W) = image.shape[:2]
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > CONFIDENCE_THRESHOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
                            CONFIDENCE_THRESHOLD)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Poster les informations textuelles de la détection sur localhost:5000/detections
    detections_infos.clear()
    for i in idxs.flatten():
        detections_infos.append({
            "label": LABELS[classIDs[i]],
            "confidence": confidences[i]
        })

    return image


def generate_frames():
    frame_count = 0
    while True:
        success, frame = video_stream.read()
        if not success:
            # If video finished, reset the video capture
            video_stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            frame_count += 1
            if frame_count % skip_frames == 0:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_processed_frames():
    frame_count = 0
    while True:
        success, frame = video_stream.read()
        if not success:
            # If video finished, reset the video capture
            video_stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            frame_count += 1
            if frame_count % skip_frames == 0:
                processed_frame = process(frame)
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_stream_file')
def video_stream_file():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_stream_file_processed')
def video_stream_file_processed():
    return Response(generate_processed_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detections')
def detections():
    return render_template('detections.html', detections=detections_infos)


@app.route('/')
def index():
    return render_template('index.html')


def cleanup():
    video_stream.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    app.run(debug=True)

    # call this function at the end of your program
    cleanup()
