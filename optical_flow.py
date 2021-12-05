import numpy as np
import cv2
from skimage.transform import SimilarityTransform
from morphing import ImageMorphingTriangulation
    
lk_params = dict(winSize  = (15,15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def optical_flow(prev_output, prev_target_features, target_frame, prev_target_frame):
    """
        input:
            prev_output: last output frame image
            prev_target_features: feature points in the last target frame
            target_frame: current target frame image
            prev_target_frame: last target frame image
            
        output:
            output: current output frame image
            feature_target: feature points in the current target frame      
    """
    
    feature_points = np.asarray(prev_target_features).astype(np.float32)[:, :, None]
    feature_points = np.transpose(feature_points, (0, 2, 1))

    prev_gray = cv2.cvtColor(prev_target_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
    
    new_features, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, feature_points, None, **lk_params) # Gunnar Farneback's algorithm
    
    good_new = new_features[st==1]
    good_old = feature_points[st==1]
    
    new_output = np.copy(prev_output)
    
    transform = SimilarityTransform()
    
    if transform.estimate(good_old, good_new):
        new_output = transform_image(good_old, good_new, prev_output, target_frame)
    
    good_new_features = []
    for each in good_new.tolist():
        s = []
        for i in range(len(each)):
            s.append(each[i])
        good_new_features.append(tuple(s))

    return new_output, good_new_features

def transform_image(good_old, good_new, prev_output, target_frame):
    warped_img1 = np.copy(target_frame)

    hullIndex = cv2.convexHull(np.array(good_new).astype(np.int32), returnPoints=False)
    
    hull1 = [good_old[int(hullIndex[i])] for i in range(0, len(hullIndex))]
    hull2 = [good_new[int(hullIndex[i])] for i in range(0, len(hullIndex))]

    # hull1, hull2 = convex_hull(points1.tolist(), points2.tolist())
    hull1 = np.array(hull1).astype(np.float32)
    hull2 = np.array(hull2).astype(np.float32)

    if (len(hull1) == 0 or len(hull2) == 0):
        return target_frame
        
    hull2 = np.asarray(hull2)
    hull2[:, 0] = np.clip(hull2[:, 0], 0, target_frame.shape[1] - 1)
    hull2[:, 1] = np.clip(hull2[:, 1], 0, target_frame.shape[0] - 1)
    hull2_list = []
    for each in hull2.astype(np.float32).tolist():
        s = []
        for i in range(len(each)):
            s.append(each[i])
        hull2_list.append(tuple(s))
    hull2 = hull2_list

    new_image = ImageMorphingTriangulation(prev_output, target_frame, good_old, good_new)
    return new_image
    
    
    