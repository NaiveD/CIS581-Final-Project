import cv2
import numpy as np

from skimage.color import rgb2gray
from skimage import feature

from utils import laplacian_pyramid, gaussian_pyramid, combine, collapse
from utils import getIndexes, getCoefficientMatrix, getSolutionVect
from utils import getSolutionVectTexture, reconstructImg, solveEqu

def blending(result, source_frame, convexhull):
    """
    input:
        result: morphing result
        source_frame
        convexhull
    output:
        seamless_output
    """
    frame1_head_mask = cv2.fillConvexPoly(np.zeros_like(source_frame[:, :, 0]), convexhull, 255)
    (x, y, w, h) = cv2.boundingRect(convexhull)
    center_face = (int((x + x + w) / 2), int((y + y + h) / 2))
    seamless_output = cv2.seamlessClone(result, source_frame, frame1_head_mask, center_face, cv2.NORMAL_CLONE)
    return seamless_output

def seamlessCloningPoisson(sourceImg, targetImg, convexhull):
    """
    Wrapper function to put all steps together
    Args:
    - sourceImg, targetImg: source and targe image
    - mask: masked area in the source image
    - offsetX, offsetY: offset of the mask in the target image
    Returns:
    - ResultImg: result image
    """
    mask = cv2.fillConvexPoly(np.zeros_like(sourceImg[:, :, 0]), convexhull, 255)
    # step 1: index replacement pixels
    indexes = getIndexes(mask, targetImg.shape[0], targetImg.shape[1])
    # step 2: compute the Laplacian matrix A
    A = getCoefficientMatrix(indexes)

    # step 3: for each color channel, compute the solution vector b
    red, green, blue = [
        getSolutionVect(indexes, sourceImg[:, :, i], targetImg[:, :, i],
                        0, 0).T for i in range(3)
    ]

    # step 4: solve for the equation Ax = b to get the new pixels in the replacement area
    new_red, new_green, new_blue = [
        solveEqu(A, channel)
        for channel in [red, green, blue]
    ]

    # step 5: reconstruct the image with new color channel
    resultImg = reconstructImg(indexes, new_red, new_green, new_blue,
                               targetImg)
    return resultImg

def PoissonTextureFlattening(result, convexhull):
    mask = cv2.fillConvexPoly(np.zeros_like(result[:, :, 0]), convexhull, 255)
    edges = feature.canny(rgb2gray(result))

    # step 1: index replacement pixels
    indexes = getIndexes(mask, result.shape[0], result.shape[1])
    # step 2: compute the Laplacian matrix A
    A = getCoefficientMatrix(indexes)

    # step 3: for each color channel, compute the solution vector b
    red, green, blue = [
        getSolutionVectTexture(indexes, result[:, :, i], mask, edges).T for i in range(3)
    ]

    # step 4: solve for the equation Ax = b to get the new pixels in the replacement area
    new_red, new_green, new_blue = [
        solveEqu(A, channel)
        for channel in [red, green, blue]
    ]

    # step 5: reconstruct the image with new color channel
    resultImg = reconstructImg(indexes, new_red, new_green, new_blue,
                               result)
    return resultImg

def laplacianBlending(result, source, convexhull):
    origin_shape = result.shape

    a1 = np.zeros((640, 640, result.shape[2]), dtype=float)
    a1[:result.shape[0], :result.shape[1], :result.shape[2]] = result
    result = a1

    a1 = np.zeros((640, 640, source.shape[2]), dtype=float)
    a1[:source.shape[0], :source.shape[1], :source.shape[2]] = source
    source = a1

    mask = cv2.fillConvexPoly(np.zeros_like(result[:, :, 0]), convexhull, 1)
    mask = np.tile(mask[:, :, np.newaxis], (1, 1, result.shape[2]))

    # depth of the pyramids
    depth = 5

    # 1) we build the Laplacian pyramids of the two images
    l_result = laplacian_pyramid(result, depth)
    l_source = laplacian_pyramid(source, depth)

    # 2) we build the Gaussian pyramid of the selected region
    Gmask = gaussian_pyramid(mask, depth)

    # 3) we combine the two pyramids using the nodes of GR as weights
    LS = combine(l_result, l_source, Gmask)

    # 4) we collapse the output pyramid to get the final blended image
    blended_image = collapse(LS).astype(np.uint8)

    return blended_image[:origin_shape[0], :origin_shape[1], :]


