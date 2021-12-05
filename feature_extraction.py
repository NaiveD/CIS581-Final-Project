import dlib

def extract_feature(img, dets, num_features=68):
    """
    input:
        img: input image
        dets: a list of dlib rectangles
        num_features: number of feature points to extract
    output:
        landmarks: a list, each component is a list of feature points of each face
    """
    landmarks = []
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    for i, det in enumerate(dets):
        features = predictor(img, det)
        landmark = []
        for n in range(0, num_features):
            tmp = [features.part(n).x, features.part(n).y]
            landmark.append(tmp)
        landmarks.append(landmark)
    return landmarks