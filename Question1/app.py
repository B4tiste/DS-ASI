import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)

video_path = "../Sujet/voitures.mp4"

# Skip every 2 frames to speed up processing.
def generate_frames(mode, skip_frames=2):
    print('Starting camera...')
    frame_count = 0
    while True:
        if mode == "webcam":
            camera = cv2.VideoCapture(0)
        elif mode == "file":
            camera = cv2.VideoCapture(video_path)
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                frame_count += 1
                if frame_count % skip_frames == 0: # Process only every 'skip_frames' frame
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # Don't forget to release the video capture at the end of the video.
        camera.release()

@app.route('/video_stream_file')
def video_stream_file():
    return Response(generate_frames("file"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)