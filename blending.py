import cv2
import numpy as np

def blending(result, source_frame, convexhull):
    """
    input:
        result: morphing result
        source_frame
        convexhull
    output:
        seamless_output
    """
    # get face
    frame1_head_mask = cv2.fillConvexPoly(np.zeros_like(source_frame[:, :, 0]), convexhull, 255)
    (x, y, w, h) = cv2.boundingRect(convexhull) # face position
    center_face = (int((x + x + w) / 2), int((y + y + h) / 2)) 
    # result is source face on the target image, second parameter is target image, third image is target head
    seamless_output = cv2.seamlessClone(result, source_frame, frame1_head_mask, center_face, cv2.NORMAL_CLONE)
    return seamless_output