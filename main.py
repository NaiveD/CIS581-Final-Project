import cv2
from detection import detect_faces
from feature_extraction import extract_feature
from morphing import ImageMorphingTriangulation, face_swap
from blending import blending
from optical_flow import optical_flow

video_1 = './1.mp4'  # FrankUnderwood 1280*720
video_2 = './2.mp4'  # MrRobot 640*360


frame_rate = 5
needDetection = True


def main():
    cap_source = cv2.VideoCapture(video_1)  # changed
    cap_target = cv2.VideoCapture(video_2)  # changed

    source_num_frame = int(cap_source.get(cv2.CAP_PROP_FRAME_COUNT))  # number of total frames
    target_num_frame = int(cap_target.get(cv2.CAP_PROP_FRAME_COUNT))  # number of total frames
    num_frame = min(source_num_frame, target_num_frame)

    # For writing the video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = cap_target.get(cv2.CAP_PROP_FPS)
    size = (int(cap_target.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_target.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("input size = ", size)
    writer = cv2.VideoWriter('Results/output.avi', fourcc, fps, size)

    count_frame = 0
    while True:
        # Read next frame
        is_source, source_frame = cap_source.read()
        is_target, target_frame = cap_target.read()

        if is_source and is_target and (count_frame < num_frame):
            source_gray = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
            target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
            
            count_frame += 1
            target_pos_frame = cap_target.get(cv2.CAP_PROP_POS_FRAMES)  # index of the current target frame
            print("current target frame: %d" % target_pos_frame)

            # Feature detection
            if (target_pos_frame - 1) % frame_rate == 0 or needDetection:
                print("Performing feature detection")

                # STEP 1: Do detection
                det_source = detect_faces(source_gray)
                det_target = detect_faces(target_gray)

                # STEP 2: Extract feature points
                feature_source = extract_feature(source_gray, det_source)[0]
                feature_target = extract_feature(target_gray, det_target)[0]

                # STEP 3: Face Warping
                dissolved_pic = ImageMorphingTriangulation(source_frame, target_frame, feature_source, feature_target)
                target_convexhull, result = face_swap(source_frame, target_frame, feature_source, feature_target,
                                                      dissolved_pic)

                # STEP 4: Seamless Blending
                output = blending(result, target_frame, target_convexhull)
                cv2.imshow("Result", output)
                print("output size = ", output.shape)
                writer.write(output)  # write the output to the video

                prev_target_frame = target_frame
                needDetection = False

            else:
                print("this is in the optical flow", target_pos_frame)
                output, feature_target = optical_flow(output, feature_target, target_frame, prev_target_frame)
                cv2.imshow("Result", output)
                writer.write(output) # write the output to the video
                prev_target_frame = target_frame

        else:
            break

        cv2.waitKey(1)

    cv2.destroyAllWindows()
    cap_source.release()
    cap_target.release()
    writer.release()


if __name__ == '__main__':
    main()




