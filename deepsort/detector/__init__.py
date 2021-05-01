from .YOLOv3 import YOLOv3


__all__ = ['build_detector']

def build_detector(use_cuda):
    return YOLOv3('./deepsort/detector/YOLOv3/cfg/yolo_v3.cfg', './deepsort/detector/YOLOv3/weight/yolov3.weights','./deepsort/detector/YOLOv3/cfg/coco.names',
                    score_thresh=0.5, nms_thresh=0.4,
                    is_xywh=True, use_cuda=use_cuda)
