
from cam.base_camera import BaseCamera
import cv2
import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs

# customize your API through the following parameters
classes_path = 'coco.names'
weights_path = './weights/yolov3.tf'
tiny = False                    # set to True if using a Yolov3 Tiny model
size = 416                      # size images are resized to for model
output_path = './detections/'   # path to output folder where images with detections are saved
num_classes = 80                # number of classes in model

# load in weights and classes
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


yolo = YoloV3(classes=num_classes)

yolo.load_weights(weights_path).expect_partial()
print('weights loaded')

class_names = [c.strip() for c in open(classes_path).readlines()]
print('classes loaded')


class Camera(BaseCamera):

    @staticmethod
    def frames():
        cam = cv2.VideoCapture(r'./finish.mp4')
        if not cam.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = cam.read()
            try:
                if CameraParams.gray:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if CameraParams.gaussian:
                    img_raw = tf.convert_to_tensor(img)
                    img_raw = tf.expand_dims(img_raw, 0)
                    # img detect
                    img_raw = transform_images(img_raw, size)
                    boxes, scores, classes, nums = yolo(img_raw)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
                if CameraParams.sobel:
                    if(len(img.shape) == 3):
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
                    img = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y
                if CameraParams.canny:
                    img = cv2.Canny(img, 100, 200, 3, L2gradient=True)
            except Exception as e:
                print(e)
            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()

class CameraParams():

    gray = False
    gaussian = False
    sobel = False
    canny = False
    def __init__(self, gray, gaussian, sobel, canny, yolo):
        self.gray = gray
        self.gaussian = gaussian
        self.sobel = sobel
        self.canny = canny
        self.yolo
