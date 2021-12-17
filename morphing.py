import numpy as np
import cv2
from scipy.spatial import Delaunay
from utils import matrixABC, generate_warp
from matplotlib import pyplot as plt


def ImageMorphingTriangulation(source_frame, target_frame, feature_source, feature_target):
    Tri = Delaunay(feature_target)
    nTri = Tri.simplices.shape[0]

    ABC_Inter_inv_set = np.zeros((nTri, 3, 3))
    ABC_im1_set = np.zeros((nTri, 3, 3))
    ABC_im2_set = np.zeros((nTri, 3, 3))

    for ii, element in enumerate(Tri.simplices):
        ABC_Inter_inv_set[ii, :, :] = np.linalg.inv(matrixABC(feature_target, element))
        ABC_im1_set[ii, :, :] = matrixABC(feature_source, element)
        ABC_im2_set[ii, :, :] = matrixABC(feature_target, element)

    size_H, size_W = target_frame.shape[:2]
    warp_im1 = generate_warp(size_H, size_W, Tri, ABC_Inter_inv_set, ABC_im1_set, source_frame)
    # warp_im2 = generate_warp(size_H, size_W, Tri, ABC_Inter_inv_set, ABC_im2_set, target_frame)

    dissolved_pic1 = warp_im1.astype(np.uint8)

    return dissolved_pic1


def face_swap(source_frame, target_frame, source_features, target_features, dissolved_pic1):
    mask_source = np.zeros_like(source_frame[:, :, 0])
    source_features = np.array(source_features, np.int32)
    source_convexhull = cv2.convexHull(source_features)
    cv2.fillConvexPoly(mask_source, source_convexhull, 255)
    # plt.imshow(mask_source)
    # plt.show()

    mask_target = np.zeros_like(target_frame[:, :, 0])
    target_features = np.array(target_features, np.int32)
    target_convexhull = cv2.convexHull(target_features)
    cv2.fillConvexPoly(mask_target, target_convexhull, 255)
    # plt.imshow(mask_target)
    # plt.show()

    # 把target人脸的位置都圈出来
    source_face_on_target = cv2.bitwise_and(dissolved_pic1, dissolved_pic1, mask=mask_target)  # source face on target position
    # plt.imshow(source_face_on_target)
    # plt.show()

    # combine the source face with target environment
    target_face_mask = cv2.fillConvexPoly(np.zeros_like(mask_target), source_convexhull, 255)
    inv_target_face_mask = cv2.bitwise_not(target_face_mask)
    target_no_face = cv2.bitwise_and(target_frame, target_frame, mask=inv_target_face_mask)
    result = cv2.add(target_no_face, source_face_on_target)
    # plt.imshow(result)
    # plt.show()

    return target_convexhull, result