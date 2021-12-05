import cv2
import dlib

def detect_faces(img):
    """
    input:
        img: input image
    output:
        dets: a list of dlib rectangles
    """

    cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
    dets = cnn_face_detector(img, 1)

    #transfer from cnn_rectangles to dlib rectangles
    rects = dlib.rectangles()
    rects.extend([d.rect for d in dets])

    return rects