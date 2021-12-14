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
    
    
    good_new_features = []
    for each in good_new.tolist():
        s = []
        for i in range(len(each)):
            s.append(each[i])
        good_new_features.append(tuple(s))

    return good_new_features