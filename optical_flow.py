import numpy as np
import cv2
lk_params = dict( winSize  = (15, 15), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def optical_flow(prev_output, target_features, curr_target_frame, prev_target_frame,frame_no):
    """
        input:
            prev_output: output from last frame
            target_features：feature points of target frame
            target_frame: target frame image
            prev_target_frame: previous target frame image
        output:
            output
        """

    """ 
        Parameters for calOpticalFlowPyrLK():
            prevImg	:first 8-bit input image or pyramid constructed by buildOpticalFlowPyramid.
            nextImg	:second input image or pyramid of the same size and the same type as prevImg.
            prevPts	:vector of 2D points for which the flow needs to be found; point coordinates must be single-precision floating-point numbers.
            nextPts	:output vector of 2D points (with single-precision floating-point coordinates) containing the calculated new positions of input features in the second image;
    """

    # convert frame into gray color
    prev_img_gray = cv2.cvtColor(prev_target_frame, cv2.COLOR_BGR2GRAY)
    curr_img_gray = cv2.cvtColor(curr_target_frame, cv2.COLOR_BGR2GRAY)

    # convert feature_points into 2D (68,2,1) and float-point number
    prev_point = np.asarray(target_features).astype(np.float32)[:, :, None]  # shape (68,2,1)
    p0 = np.transpose(prev_point, (0, 2, 1)) # requirement is (68,1,2)

    # p1 = feature points in the next frame, shape(68,1,2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_img_gray, curr_img_gray, p0, None, **lk_params)

    good_p1 = p1[st==1]
    good_p0 = p0[st==1]

    curr_output = np.copy(prev_output)
    

    pass