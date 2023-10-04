import array
import base64
import json
import queue
import threading
import time

import cv2
import flask

from tracker import Tracker
from yolov5.utils.general import Profile
from flask import Flask, request, jsonify

video_path = "./person_480P.mp4"
weights_path = "./yolov5s.pt"
web_host = "0.0.0.0"
web_port = 5000


def read_video(que: queue.Queue):
    global video_path
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened() is not True:
        print("video not opened")
        return
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cnt = 0
    while True:
        r, f = cap.read()
        if r is not True:
            print("read failed")
            break
        cnt += 1
        if cnt == frame_num:
            cnt = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #print(f'QUEUE1 length{que.qsize()}')
        if que.full():
            #print("loss frame")
            que.get()
        que.put_nowait(f)
        time.sleep(0.015)
    cap.release()


def tracing(t: Tracker, que1: queue.Queue, que2: queue.Queue):
    dt = Profile()
    while True:
        frame = que1.get(block=True)
        if frame is not None:
            with dt:
                image = t.run(frame)
            if que2.full():
                que2.get()
            que2.put_nowait(image)


def flask_web(t:Tracker, que: queue.Queue):
    app = Flask(
        import_name="cross line count by yolov5",
        static_folder="./static",
        template_folder="./static"
    )

    def cv2_to_base64(image):
        image1 = cv2.imencode('.jpg', image)[1]
        image_code = str(base64.b64encode(image1))[2:-1]
        return "data:image/jpeg;base64," + image_code

    def event_proc():
        while True:
            image = que.get(block=True)
            if image is not None:
                code = cv2_to_base64(image)
                yield 'data: {}\n\n'.format(json.dumps(({'image': code})))

    @app.route("/")
    def index():
        return flask.render_template("index.html")

    @app.route("/event")
    def update():
        print("event-stream 连接成功")
        return flask.Response(event_proc(), mimetype="text/event-stream")

    @app.route("/config", methods=['POST'])
    def config():
        data = request.get_json()
        t.update_config(data['line'], data['arrow'])
        return jsonify({"message": "ok"})

    app.run(
        host=web_host,
        port=web_port,
        debug=False
    )


def play(que: queue.Queue):
    window_name = "test"
    cv2.namedWindow(window_name, 0)
    cv2.resizeWindow(window_name, 800, 600)
    while True:
        image = que.get(block=True)
        if image is not None:
            cv2.imshow(window_name, image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    Q1 = queue.Queue(maxsize=50)
    Q2 = queue.Queue(maxsize=50)
    tracker = Tracker(weights=weights_path)
    ths = [
        threading.Thread(target=read_video, args=(Q1,)),
        threading.Thread(target=tracing, args=(tracker, Q1, Q2,)),
        threading.Thread(target=flask_web, args=(tracker, Q2,))
    ]
    [t.setDaemon(True) for t in ths]
    [t.start() for t in ths]

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("exiting")
        exit(0)
