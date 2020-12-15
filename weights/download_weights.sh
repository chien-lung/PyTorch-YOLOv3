#!/bin/bash
# Download weights for vanilla YOLOv3
wget -c https://pjreddie.com/media/files/yolov3.weights
# # Download weights for tiny YOLOv3
wget -c https://pjreddie.com/media/files/yolov3-tiny.weights
# Download weights for backbone network
wget -c https://pjreddie.com/media/files/darknet53.conv.74
# Download weight for YOLOv3 pretrained on GTSDB
wget -c https://www.dropbox.com/s/5i5q5opu6wch6qz/yolov3_GTSDB.pth
# Download weight for YOLOv3 pretrained on LISA
wget -c https://www.dropbox.com/s/z7q7q26dak9z9bg/yolov3_LISA.pth