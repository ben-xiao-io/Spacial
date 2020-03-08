# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import numpy as np
import sys
import argparse
import time
import cv2
import os
import datetime
import serial
import io
import sql_connect
import asyncio
from multiprocessing import Process
from pynput import keyboard

cv2.imread("images/default.jpg")

def object_detection(confidence_arg = 0.5, threshold_arg = 0.3, image_source = cv2):

    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join(["yolo-coco/objects", "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join(["yolo-coco/objects", "yolov3.weights"])
    configPath = os.path.sep.join(["yolo-coco/objects", "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # load our input image and grab its spatial dimensions
    image = image_source
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidence_arg:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_arg,
        threshold_arg)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)

    # show the output image
    #xcv2.imshow("Image", image)

    #cv2.imwrite("output/output.jpg", image)
    return image

def camera_startup():
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    task_continue = True
    
    while task_continue:
        # init webcam
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)

        image_recogized = object_detection(0.5, 0.3, frame)
        
        cv2.imshow('Input', image_recogized)
        cv2.imwrite('images/temp.jpg', image_recogized)

        c = cv2.waitKey(1)
        
        if c == 27:
            task_continue = False
            if cap.isOpened():
                print ("Closing camera")
                cap.release()

            cv2.destroyAllWindows()
            break


    exit()

def arduino_read():
    arduino = serial.Serial('COM4', 115200, timeout=.1)

    while True:	

        # init arduino
        data = arduino.readline()[:-2] #the last bit gets rid of the new-line chars
        if data:
            distance = int(data.decode("utf-8"))

            if (distance <= 300 and distance != -1):
                print (distance)
                # record and send image
                sql_connect.upload()
                
           

def begin(): 
    if __name__ == '__main__':
        p1 = Process(target=camera_startup)
        p2 = Process(target=arduino_read)

        p1.start()
        p2.start()
        
        p1.join()
        p2.join()

(begin())