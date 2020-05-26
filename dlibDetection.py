import dlib
import cv2 
import yoloDetection 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def frontal_face(frame):
    front_faces = []
    faces = yoloDetection.yolo_detection(frame)
    if len(faces) > 0:
        for face in faces:
            ret=dlib.rectangle(left=face[0], top=face[1], right=face[2], bottom=face[3])
            landmarks = predictor(frame, ret)
            left_point = [landmarks.part(36).x, landmarks.part(36).y]
            right_point = [landmarks.part(45).x, landmarks.part(45).y]
            if left_point and right_point is not None:
                front_faces.append(face)
            else:
                continue
    else:
        print('No face detected')

    return front_faces


