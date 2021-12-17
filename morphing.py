import numpy as np
import cv2
from scipy.spatial import Delaunay
from utils import matrixABC, generate_warp


def ImageMorphingTriangulation(source_frame, target_frame, feature_source, feature_target):
    Tri = Delaunay(feature_source)
    nTri = Tri.simplices.shape[0]

    ABC_Inter_inv_set = np.zeros((nTri, 3, 3))
    ABC_im1_set = np.zeros((nTri, 3, 3))
    ABC_im2_set = np.zeros((nTri, 3, 3))

    for ii, element in enumerate(Tri.simplices):
        ABC_Inter_inv_set[ii, :, :] = np.linalg.inv(matrixABC(feature_source, element))
        ABC_im1_set[ii, :, :] = matrixABC(feature_source, element)
        ABC_im2_set[ii, :, :] = matrixABC(feature_target, element)

    size_H, size_W = source_frame.shape[:2]
    warp_im1 = generate_warp(size_H, size_W, Tri, ABC_Inter_inv_set, ABC_im1_set, source_frame)
    warp_im2 = generate_warp(size_H, size_W, Tri, ABC_Inter_inv_set, ABC_im2_set, target_frame)

    dissolved_pic2 = warp_im2.astype(np.uint8)

    return dissolved_pic2


def face_swap(frame, frame2, landmarks_pts, landmarks_pts2, dissolved_pic2):
    mask1 = np.zeros_like(frame[:, :, 0])
    pts = np.array(landmarks_pts, np.int32)
    convexhull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask1, convexhull, 255)

    mask2 = np.zeros_like(frame2[:, :, 0])
    pts2 = np.array(landmarks_pts2, np.int32)
    convexhull2 = cv2.convexHull(pts2)
    cv2.fillConvexPoly(mask2, convexhull2, 255)

    # 把target人脸的位置都圈出来
    face_masked2 = cv2.bitwise_and(dissolved_pic2, dissolved_pic2, mask=mask1)

    # Face swapped (place 1st face onto the 2nd face)
    frame1_head_mask = cv2.fillConvexPoly(np.zeros_like(mask1), convexhull, 255)
    frame1_face_mask = cv2.bitwise_not(frame1_head_mask)
    frame1_no_face = cv2.bitwise_and(frame, frame, mask=frame1_face_mask)
    result = cv2.add(frame1_no_face, face_masked2)

    return convexhull, result