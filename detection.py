import cv2
import dlib

def detect_faces(img):
    """
    input:
        img: input image
    output:
        dets: a list of dlib rectangles
    """

    detector = dlib.get_frontal_face_detector()
    dets = detector(img)
    return dets