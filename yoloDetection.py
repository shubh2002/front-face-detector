import cv2
import numpy as np

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416

def refined_box(left, top, width, height):
    right = left + width
    bottom = top + height
    original_ver_height = bottom - top
    top = int(top + original_ver_height * 0.15)
    bottom = int(bottom - original_ver_height * 0.05)

    margin = ((bottom - top) - (right - left)) // 2
    left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1
    right = right + margin
    
    return left, top, right, bottom








def get_output_names(net):
    #name of layers
    layers_name = net.getLayerNames()
    #name of output layers
    return [layers_name[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def post_process(frame, outs, cnf_threshold, nms_threshold):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    #scan through all the bounding boxes output from the network and keep only    
    #the with high confidence
    confidences = []
    boxes = []
    final_boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > cnf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width/2)
                top = int(center_y - height/2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    #perforn non max suppression to eliminate redudant overlapping boxes 
    indices = cv2.dnn.NMSBoxes(boxes, confidences, cnf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[0]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        #left, top, right, bottom = refined_box(left, top, width, height)
        final_boxes.append(box)
    
    return final_boxes


def yolo_detection(frame):
    net = cv2.dnn.readNetFromDarknet('yolov3_face_detector.cfg', 'yolov3_face_detector.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    blob = cv2.dnn.blobFromImage(frame, 1/255, (IMG_WIDTH, IMG_HEIGHT), [0,0,0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_names(net))
    faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)

    return faces


