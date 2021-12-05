import numpy as np


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
    # Generate x,y meshgrid
    x = np.linspace(0, size_W - 1, size_W)
    y = np.linspace(0, size_H - 1, size_H)
    xv, yv = np.meshgrid(x, y)

    # Flatten the meshgrid
    x_reshaped = xv.reshape(1, xv.shape[0] * xv.shape[1]).squeeze().tolist()
    y_reshaped = yv.reshape(1, yv.shape[0] * yv.shape[1]).squeeze().tolist()

    # Zip the flattened x, y and Find Simplices (hint: use list and zip)
    flatten_xy = zip(x_reshaped, y_reshaped)
    temp = np.array(list(flatten_xy))

    tri_idx = []
    for i in range(temp.shape[0]):
        tri_idx.append(Tri.find_simplex(temp[i]).item())

    # compute alpha, beta, gamma for all the color layers(3)
    ones_ = np.ones((size_H * size_W, 1))
    y_expand = np.concatenate((temp, ones_), -1)[:, :, None]
    brc_coor = A_Inter_inv_set[tri_idx] @ y_expand
    beta = brc_coor[:, 1, :]

    # Find all x and y co-ordinates
    pt_xy = A_im_set[tri_idx] @ brc_coor

    all_z_coor = np.ones(beta.size)

    # Divide all x and y with z
    pt_xy[:, 0:2, :] = pt_xy[:, 0:2, :] / pt_xy[:, -1, :].reshape((-1, 1, 1))

    # Generate Warped Images (Use function interp2) for each of 3 layers
    generated_pic = np.zeros((size_H, size_W, 3), dtype=np.uint8)

    # IMPLEMENT HERE
    pt_x = pt_xy[:, 0, :].reshape(size_H, size_W)
    pt_y = pt_xy[:, 1, :].reshape(size_H, size_W)

    for i in range(3):
        pixel = interp2(image[:, :, i], pt_x, pt_y)
        generated_pic[:, :, i] = pixel

    return generated_pic