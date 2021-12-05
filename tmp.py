import sys
import cv2
import dlib
import numpy as np
from scipy.spatial.qhull import Delaunay


def matrixABC(sparse_control_points, elements):
    output = np.zeros((3, 3))
    for i, element in enumerate(elements):
        output[0:2, i] = sparse_control_points[element]
    output[2, :] = 1
    return output


def interp2(v, xq, yq):
    dim_input = 1
    if len(xq.shape) == 2 or len(yq.shape) == 2:
        dim_input = 2
        q_h = xq.shape[0]
        q_w = xq.shape[1]
        xq = xq.flatten()
        yq = yq.flatten()

    h = v.shape[0]
    w = v.shape[1]
    if xq.shape != yq.shape:
        raise ('query coordinates Xq Yq should have same shape')

    x_floor = np.floor(xq).astype(np.int32)
    y_floor = np.floor(yq).astype(np.int32)
    x_ceil = np.ceil(xq).astype(np.int32)
    y_ceil = np.ceil(yq).astype(np.int32)

    x_floor[x_floor < 0] = 0
    y_floor[y_floor < 0] = 0
    x_ceil[x_ceil < 0] = 0
    y_ceil[y_ceil < 0] = 0

    x_floor[x_floor >= w - 1] = w - 1
    y_floor[y_floor >= h - 1] = h - 1
    x_ceil[x_ceil >= w - 1] = w - 1
    y_ceil[y_ceil >= h - 1] = h - 1

    v1 = v[y_floor, x_floor]
    v2 = v[y_floor, x_ceil]
    v3 = v[y_ceil, x_floor]
    v4 = v[y_ceil, x_ceil]

    lh = yq - y_floor
    lw = xq - x_floor
    hh = 1 - lh
    hw = 1 - lw

    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw

    interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

    if dim_input == 2:
        return interp_val.reshape(q_h, q_w)
    return interp_val


def generate_warp(size_H, size_W, Tri, A_Inter_inv_set, A_im_set, image):
    x, y = np.meshgrid(np.arange(size_W), np.arange(size_H))
    x = x.ravel()
    y = y.ravel()

    pts = zip(x, y)
    pts_list = []
    for p in pts:
        pts_list.append(p)

    simplices = Tri.find_simplex(pts_list)

    px = np.ndarray((3, size_H * size_W))
    zind = np.ones((1, size_H * size_W))  # 1xN

    px[0, :] = x
    px[1, :] = y
    px[2, :] = zind

    all_inv_ABC = A_Inter_inv_set[simplices]  # 3 x 3 x size_H*size_W
    alphas = all_inv_ABC[:, 0, 0] * px[0] + all_inv_ABC[:, 0, 1] * px[1] + all_inv_ABC[:, 0, 2] * 1
    betas = all_inv_ABC[:, 1, 0] * px[0] + all_inv_ABC[:, 1, 1] * px[1] + all_inv_ABC[:, 1, 2] * 1
    gammas = all_inv_ABC[:, 2, 0] * px[0] + all_inv_ABC[:, 2, 1] * px[1] + all_inv_ABC[:, 2, 2] * 1

    bary = np.ndarray((3, size_H * size_W))
    bary[0, :] = alphas
    bary[1, :] = betas
    bary[2, :] = gammas

    all_im_set = A_im_set[simplices]  # 3 x 3 x size_H*size_W
    px_x = all_im_set[:, 0, 0] * bary[0] + all_im_set[:, 0, 1] * bary[1] + all_im_set[:, 0, 2] * bary[2]
    px_y = all_im_set[:, 1, 0] * bary[0] + all_im_set[:, 1, 1] * bary[1] + all_im_set[:, 1, 2] * bary[2]
    px_z = all_im_set[:, 2, 0] * bary[0] + all_im_set[:, 2, 1] * bary[1] + all_im_set[:, 2, 2] * bary[2]

    final_px = np.ndarray((3, size_H * size_W))
    final_px[0, :] = px_x
    final_px[1, :] = px_y
    final_px[2, :] = px_z

    generated_pic = np.zeros((size_H, size_W, 3), dtype=np.uint8)

    image_r = interp2(image[:, :, 0], final_px[0], final_px[1])
    image_g = interp2(image[:, :, 1], final_px[0], final_px[1])
    image_b = interp2(image[:, :, 2], final_px[0], final_px[1])

    image_r = np.reshape(image_r, (size_H, size_W))
    image_g = np.reshape(image_g, (size_H, size_W))
    image_b = np.reshape(image_b, (size_H, size_W))

    generated_pic[:, :, 0] = image_r
    generated_pic[:, :, 1] = image_g
    generated_pic[:, :, 2] = image_b
    if len(generated_pic[:,:,0]) < 1:
        print("no generated image")
    return generated_pic

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


# vid1 = (str(sys.argv[1]))
# vid2 = (str(sys.argv[2]))

vid1 = './1.mp4'
vid2 = './2.mp4'
cap2 = cv2.VideoCapture(vid1)
cap1 = cv2.VideoCapture(vid2)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap1.read()
    if frame is None:
        cap1 = cv2.VideoCapture(vid2)
        _, frame = cap1.read()

    _, frame2 = cap2.read()
    if frame2 is None:
        cap2 = cv2.VideoCapture(vid1)
        _, frame2 = cap2.read()

    f_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f_gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Feature detection and face extraction for face 1 --------
    dets = detector(f_gray)
    for k, d in enumerate(dets):
        landmarks = predictor(f_gray, d)
        landmarks_pts = []
        for n in range(0, 68):
            list = [landmarks.part(n).x, landmarks.part(n).y]
            landmarks_pts.append(list)

    # Feature detection and face extraction for face 2
    dets2 = detector(f_gray2,1)
    for k, d in enumerate(dets2):
        landmarks2 = predictor(f_gray2, d)
        landmarks_pts2 = []
        for n in range(0, 68):
            list = [landmarks2.part(n).x, landmarks2.part(n).y]
            landmarks_pts2.append(list)

    Tri = Delaunay(landmarks_pts)
    nTri = Tri.simplices.shape[0]

    ABC_Inter_inv_set = np.zeros((nTri, 3, 3))
    ABC_im1_set = np.zeros((nTri, 3, 3))
    ABC_im2_set = np.zeros((nTri, 3, 3))

    for ii, element in enumerate(Tri.simplices):
        ABC_Inter_inv_set[ii, :, :] = np.linalg.inv(matrixABC(landmarks_pts, element))
        ABC_im1_set[ii, :, :] = matrixABC(landmarks_pts, element)
        ABC_im2_set[ii, :, :] = matrixABC(landmarks_pts2, element)

    size_H, size_W = frame.shape[:2]
    warp_im1 = generate_warp(size_H, size_W, Tri, ABC_Inter_inv_set, ABC_im1_set, frame)
    warp_im2 = generate_warp(size_H, size_W, Tri, ABC_Inter_inv_set, ABC_im2_set, frame2)

    dissolved_pic = warp_im1.astype(np.uint8)
    dissolved_pic2 = warp_im2.astype(np.uint8)

    # Mask the face for extraction and everything else is black
    mask1 = np.zeros_like(f_gray)
    pts = np.array(landmarks_pts, np.int32)
    convexhull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask1, convexhull, 255)

    height1, width1, channels1 = frame.shape
    frame1_new_face = np.zeros((height1, width1, channels1), np.uint8)
    face_masked = cv2.bitwise_and(dissolved_pic2, dissolved_pic2, mask=mask1)

    mask2 = np.zeros_like(f_gray2)
    pts2 = np.array(landmarks_pts2, np.int32)
    convexhull2 = cv2.convexHull(pts2)
    cv2.fillConvexPoly(mask2, convexhull2, 255)

    height2, width2, channels2 = frame.shape
    frame2_new_face = np.zeros((height2, width2, channels2), np.uint8)
    face_masked2 = cv2.bitwise_and(dissolved_pic2, dissolved_pic2, mask=mask1)

    # Face swapped (place 1st face onto the 2nd face)
    frame1_head_mask = cv2.fillConvexPoly(np.zeros_like(f_gray), convexhull, 255)
    frame1_face_mask = cv2.bitwise_not(frame1_head_mask)
    frame1_no_face = cv2.bitwise_and(frame, frame, mask=frame1_face_mask)
    result = cv2.add(frame1_no_face, face_masked2)

    # seamless cloning with OpenCV2
    (x, y, w, h) = cv2.boundingRect(convexhull)
    center_face = (int((x + x + w) / 2), int((y + y + h) / 2))
    seamless_output = cv2.seamlessClone(result, frame, frame1_head_mask, center_face, cv2.NORMAL_CLONE)
    cv2.imshow("Result", seamless_output)

    key = cv2.waitKey(1)
    if key == 0:
        break
