import cv2
import numpy as np


def blending(result, source_frame, convexhull, method=cv2.NORMAL_CLONE):
    """
    input:
        result: morphing result
        source_frame
        convexhull
    output:
        seamless_output
    """
    frame1_head_mask = cv2.fillConvexPoly(np.zeros_like(result[:, :, 0]), convexhull, 255)
    (x, y, w, h) = cv2.boundingRect(convexhull)
    center_face = (int(x + w / 2), int(y + h / 2))
    seamless_output = cv2.seamlessClone(np.uint8(result), source_frame, frame1_head_mask, center_face, cv2.NORMAL_CLONE)
    return seamless_output