import cv2
import time
from flask import Flask, request, Response,render_template
import json
from cam.base_camera import BaseCamera
from deepsort.detector import build_detector
from deepsort.deep_sort import build_tracker
from deepsort.utils.draw import draw_boxes
from deepsort.detector.YOLOv3 import YOLOv3
from numba import njit
from vizer.draw import draw_boxes as db
# Initialize Flask application
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
app = Flask(__name__)

class_names = [c.strip() for c in open(r'cam/coco.names').readlines()]

yolo = YOLOv3(r"deepsort/detector/YOLOv3/cfg/yolo_v3.cfg", r"deepsort/detector/YOLOv3/weight/yolov3.weights",
                      r"cam/coco.names")
detector = build_detector(use_cuda=False)
deepsort = build_tracker(use_cuda=False)

file_name = ['jpg','jpeg','png']
video_name = ['mp4','avi']

# API that returns image with detections on it
@njit(parallel = True)
@app.route('/images', methods= ['POST'])
def get_image():
    image = request.files["images"]
    image_name = image.filename

    if image_name.split('.')[-1] in video_name:
        with open('./result.txt', 'w') as f:
            f.write(image_name)

    image.save(os.path.join(os.getcwd(), image_name))

    if image_name.split(".")[-1] in file_name:
        a = time.time()
        img = cv2.imread(image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bbox, cls_conf, cls_ids = yolo(img)
        if bbox is not None:
            img = db(img, bbox, cls_ids, cls_conf, class_name_map=class_names)
        img = img[:, :, (2, 1, 0)]
        _, img_encoded = cv2.imencode('.png', img)
        response = img_encoded.tobytes()
        print(time.time()-a)
        # os.remove(image_name)
        try:
            return Response(response=response, status=200, mimetype='image/png')
        except:
            return render_template('index1.html')
    else:
        return render_template('real-time.html')


class Camera(BaseCamera):
    @staticmethod
    def frames():
        with open('./result.txt', 'r') as f:
            image_name = f.read()
        fi_name = image_name
        cam = cv2.VideoCapture(image_name)
        y = 0
        sum = 0
        a = time.time()
        while True:
            b = time.time() - a
            if b > 80:
                break
            with open('./result.txt', 'r') as f:
                image_name = f.read()
            if image_name != fi_name:
                break
            ret,img = cam.read()
            if ret:
                if CameraParams.gray:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    yield cv2.imencode('.jpg', img)[1].tobytes()
                elif CameraParams.gaussian:
                    if y == 0:
                        cam = cv2.VideoCapture(image_name)
                        sum = sum - 1
                        y = 1
                    de_sum = []
                    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    bbox_xywh, cls_conf, cls_ids = detector(im)

                    mask = cls_ids == 0
                    new_bbox_xywh = bbox_xywh[mask]
                    new_bbox_xywh[:, 3:] *= 1.2

                    new_cls_conf = cls_conf[mask]
                    outputs = deepsort.update(new_bbox_xywh, new_cls_conf, im)
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        for id in identities:
                            # if id not in de_sum:
                            de_sum.append(id)

                        img = draw_boxes(img, bbox_xyxy, identities)
                    de_sum = set(de_sum)
                    text = "people "
                    if (len(de_sum) > 0):
                        text = text + str(len(de_sum))
                    else:
                        text = text + str(0)
                    cv2.putText(img, text, (30, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (250, 250, 0), 8)
                    yield cv2.imencode('.jpg', img)[1].tobytes()
                elif CameraParams.sobel:
                    img = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
                    img = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y
                    yield cv2.imencode('.jpg', img)[1].tobytes()
                elif CameraParams.canny:
                    img = cv2.Canny(img, 100, 200, 3, L2gradient=True)
                    yield cv2.imencode('.jpg', img)[1].tobytes()
            else:
                cam = cv2.VideoCapture(image_name)
                sum = sum + 1
            if sum == 2:
                break
class CameraParams():

    gray = False
    gaussian = False
    sobel = False
    canny = False

    @njit(parallel = True)
    def __init__(self, gray, gaussian, sobel, canny):
        self.gray = gray
        self.gaussian = gaussian
        self.sobel = sobel
        self.canny = canny

@njit(parallel = True)
@app.route('/')
def upload_file():
   return render_template('index1.html')

@njit(parallel = True)
@app.route('/cameraParams', methods=['GET', 'POST'])
def cameraParams():
    if request.method == 'GET':
        data = {
            'gray': CameraParams.gray,
            'gaussian': CameraParams.gaussian,
            'sobel': CameraParams.sobel,
            'canny': CameraParams.canny,
        }
        return app.response_class(response=json.dumps(data),
                                    status=200,
                                    mimetype='application/json')
    elif request.method == 'POST':
        try:
            data = request.form.to_dict()
            CameraParams.gray = str_to_bool(data['gray'])
            CameraParams.gaussian = str_to_bool(data['gaussian'])
            CameraParams.sobel = str_to_bool(data['sobel'])
            CameraParams.canny = str_to_bool(data['canny'])
            message = {'message': 'Success'}
            response = app.response_class(response=json.dumps(message),
                                    status=200,
                                    mimetype='application/json')
            return response
        except Exception as e:
            print(e)
            response = app.response_class(response=json.dumps(e),
                                    status=400,
                                    mimetype='application/json')
            return response
    else:
        data = { "error": "Method not allowed. Please GET or POST request!" }
        return app.response_class(response=json.dumps(data),
                                    status=400,
                                    mimetype='application/json')

@njit(parallel = True)
@app.route('/realtime')
def realtime():
    return render_template('real-time.html')

########get  path
@njit(parallel = True)
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(genWeb(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def genWeb(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@njit()
def str_to_bool(s):
    if s == "true":
        return True
    elif s == "false":
        return False
    else:
        raise ValueError

if __name__ == '__main__':
    app.run(debug=True, host = '127.0.0.1', port=5000)
