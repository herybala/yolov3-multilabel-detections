# Yolov3 with multiple detections option on objects
As we know, yolov3 is able to make classify obejects with more than one label thanks to logistics regression classifier.
This is the first implementation of the multilabel object detection, the original code was from https://github.com/ultralytics/yolov3.
The modifications brought to the original code are from utils.py and detect.py.
I have trained my model with my own data for 8 classes : Square plate, Round plate, Tree, Metal rail, Bollar, Electric pole and parents classes are Pole and Plate.
