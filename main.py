import cv2
from detection import detect_faces
from feature_extraction import extract_feature
from morphing import ImageMorphingTriangulation, face_swap
from blending import blending
from optical_flow import optical_flow

video_source = './2.mp4' # MrRobot 640*360 
video_target = './1.mp4' # FrankUnderwood 1280*720 

frame_rate = 5
needDetection = True

def main():
    cap_source = cv2.VideoCapture(video_source) # changed
    cap_target = cv2.VideoCapture(video_target) # changed

    source_num_frame = int(cap_source.get(cv2.CAP_PROP_FRAME_COUNT)) # number of total frames
    target_num_frame = int(cap_target.get(cv2.CAP_PROP_FRAME_COUNT)) # number of total frames
    num_frame = min(source_num_frame, target_num_frame)

    # For writing the video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = cap_source.get(cv2.CAP_PROP_FPS)
    size = (int(cap_source.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_source.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("input size = ", size)
    writer = cv2.VideoWriter('Results/output.avi', fourcc, fps, size)

    count_frame = 0
    while True:
        # Read next frame
        is_source, source_frame = cap_source.read()
        is_target, target_frame = cap_target.read()

        if is_source and is_target and (count_frame < num_frame):
            count_frame += 1
            target_pos_frame = cap_target.get(cv2.CAP_PROP_POS_FRAMES) # index of the current target frame
            print("current target frame: %d" % target_pos_frame)
            
            # Feature detection
            if (target_pos_frame - 1) % frame_rate == 0 or needDetection:
                print("Performing feature detection")
                
                # STEP 1: Do detection
                det_source = detect_faces(source_frame)
                det_target = detect_faces(target_frame)

                # STEP 2: Extract feature points
                feature_source = extract_feature(source_frame, det_source)[0]
                feature_target = extract_feature(target_frame, det_target)[0]

                # STEP 3: Face Warping
                dissolved_pic = ImageMorphingTriangulation(source_frame, target_frame, feature_source, feature_target)
                convexhull, result = face_swap(source_frame, target_frame, feature_source, feature_target, dissolved_pic)

                # STEP 4: Seamless Blending
                output = blending(result, source_frame, convexhull)
                cv2.imshow("Result", output)
                print("output size = ", output.shape)
                writer.write(output) # write the output to the video

                # prev_target_frame = target_frame
                prev_source_frame = source_frame
                
                needDetection = False
                
            else:
                # Perform optical flow
                print("Performing optical flow")
                feature_source = optical_flow(output, feature_source, source_frame, prev_source_frame)
                
                dissolved_pic = ImageMorphingTriangulation(source_frame, target_frame, feature_source, feature_target)
                convexhull, result = face_swap(source_frame, target_frame, feature_source, feature_target, dissolved_pic)

                output = blending(result, source_frame, convexhull)
                
                cv2.imshow("Result", output)
                print("output size = ", output.shape)
                writer.write(output)
                prev_source_frame = source_frame
        else:
            break


        cv2.waitKey(1)

    cv2.destroyAllWindows()
    cap_source.release()
    cap_target.release()
    writer.release()


if __name__ == '__main__':
    main()




