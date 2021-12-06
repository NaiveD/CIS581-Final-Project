import cv2
from detection import detect_faces
from feature_extraction import extract_feature
from morphing import ImageMorphingTriangulation, face_swap
from blending import blending
from optical_flow import optical_flow

video_source = './2.mp4'
video_target = './1.mp4'

frame_rate = 5
needDetection = True

def main():
    cap_source = cv2.VideoCapture(video_source)
    cap_target = cv2.VideoCapture(video_target)

    source_num_frame = int(cap_source.get(cv2.CAP_PROP_FRAME_COUNT))
    target_num_frame = int(cap_target.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frame = min(source_num_frame, target_num_frame)

    count_frame = 0
    while True:
        is_source, source_frame = cap_source.read()
        is_target, target_frame = cap_target.read()
        if is_source and is_target and count_frame < num_frame:
            count_frame += 1
            target_pos_frame = cap_target.get(cv2.CAP_PROP_POS_FRAMES)
            if (target_pos_frame - 1) % frame_rate == 0 or needDetection:
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

                prev_target_frame = target_frame
                needDetection = False
            else:
                print("this is in the optical flow", target_pos_frame)
                output, feature_target = optical_flow(output, feature_target, target_frame, prev_target_frame)
                cv2.imshow("Result", output)
                prev_target_frame = target_frame

        key = cv2.waitKey(1)
        if key == 0:
            break

    cv2.destroyAllWindows()
    cap_source.release()
    cap_target.release()


if __name__ == '__main__':
    main()




