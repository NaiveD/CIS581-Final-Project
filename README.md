# CIS581-Final-Project

Face Swapping in Two Videos

Members: Jiaqi Xie, Keyan Zhai, Yu-Chia Shen, Zhi Zheng, Li-Pu Chen

## How to run the code

1. Put two test videos inside the root directory.

2. Set the source video and target video in `main.py` (line 8).

3. Run `python main.py`.

4. The results will be `Results/ouput.avi`.

## Implementation details

The main framework is implemented in `main.py`. We process each frame of the videos in a while loop. We perform feature detection at a `frame_rate` of 5. For the other frames, we perform optical flow to track the feature points. 

Feature detection is implemented in `detection.py` and `feature_extraction.py` where we use the face detector in the library `dlib` to obtain the feature points in the souce frame and the target frame.

For the frames that we are not detecting and extracting features points, we use optical flow implemented in `optical flow.py` to track the moving face and return the feature points. We set a window size of 15 by 15 and use the `cv2.calcOpticalFlowPyrLK()` function to perform optical flow. 

Once we have the feature points of the faces in the source and target frame, we perform image morphing and warp the source face to the target frame. We implemented tiangulation and the morphing functions in `morphing.py`. 

Finally, to make the swapped face more natural, we perform image blending with `cv2.seamlessClone()` in `blendidng.py` to fuse the  face in the source video with the background in the target video.
