# YOLOv3 and Deep Sort with PyTorch Run On The Web Page

##Computer terminal
[![asciicast](demo/output.gif)

##Mobile terminal

![](demo/mobile.gif)

## Introduction
Combined with yolov3, deep sort and flash, it is a target detection and multi-target tracking platform that can run on Web pages. You can upload pictures or videos. When the image is uploaded, target detection is carried out. When the video is uploaded, multi-target tracking is carried out (the default is pedestrian, which can be changed to other objects). The mobile terminal provides an online shooting interface for real-time target detection and multi-target tracking.
## Dependencies
- torch
- torchvision
- numpy
- opencv-python==4.1.2.30
- lxml
- tqdm
- flask
- seaborn
- pillow
- vizer
- numba

## Quick Start
##### 1. Check all dependencies installed
```bash
pip install -r requirements.txt
```

##### 2. Download YOLOv3 parameters
```
cd deepsort/detector/YOLOv3/weight/
wget https://pjreddie.com/media/files/yolov3.weights
cd ../../../
```

##### 3. Download deepsort parameters ckpt.t7
```
cd deep_sort/deep/checkpoint
# download ckpt.t7 from
https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6 to this folder
cd ../../../
```  

##### 4. Or download the weight file through baidu network disk
[提取码：hhbb](https://pan.baidu.com/s/1blu8U3wM4NN2TpDK3U5leA )
  
```angular2
yolov3.weight  put in  deepsort/detector/YOLOv3/weight/
ckpt.t7        put in  deep_sort/deep/checkpoint
```

##### 5. Compile nms module
```bash
cd detector/YOLOv3/nms
sh build.sh
cd ../../../
```
or
```bash
cd detector/YOLOv3/nms/ext
python build.py build_ext develop
cd ../../../../
```
##### 6. Run
```
python app.py
```

##### 7. If you want to configure to run on the server, please visit my blog [阿里云ECS部署python,flask项目，简单易懂，无需nginx和uwsgi](https://blog.csdn.net/qq_44523137/article/details/112676287)

##### 8. You can use yolov3 demo
```
python detector.py
```

##Result
![avatar](demo/1.png)

![avatar](demo/2.png)

![avatar](demo/3.png)
## References

- code: [ZQPei/deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch)
