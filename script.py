import numpy as np
import argparse
import cv2

# function to detect face contour using Haar cascade (https://cs.stackexchange.com/questions/109227/how-to-detect-facial-landmarks-using-haar-and-other-way)
# detect_face_contour() is inaccurate b/c it assumes the face is the most defined contour
# Haar features (https://en.wikipedia.org/wiki/Haar-like_feature) are read using the cv::CascadeClassifier::load
# sources: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

# load pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

# # function to detect largest contour
# def detect_face_contour(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (11,11), 0)
#     edged = cv2.Canny(blurred, 30, 150)

#     (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     face_contour = max(cnts, key=cv2.contourArea) # assuming largest contour is the face
#     # this may be a simplification (we will see as we input different images)

#     return face_contour

# function to swap face region
def swap_faces(img1, img2):
    face_countour1 = detect_face_contour(img1)
    face_countour2 = detect_face_contour(img2)

    if face_countour1 is None or face_countour2 is None:
        print("Couldn't detect face contour in image(s).")
        return None
    
    # bounding rectangles/mask around contours
    x1, y1, w1, h1 = cv2.boundingRect(face_countour1)
    x2, y2, w2, h2 = cv2.boundingRect(face_countour2)

    # crop face region from mask
    face1 = img1[y1:y1+h1, x1:x1+w1]
    face2 = img2[y2:y2+h2, x2:x2+w2]

    # resize to match target bounds
    face1_resize = cv2.resize(face1, (w2,h2), interpolation=cv2.INTER_AREA)
    face2_resize = cv2.resize(face2, (w1,h1), interpolation=cv2.INTER_AREA)

    # replace face regions
    img1[y1:y1+h1, x1:x1+w1] = face2_resize
    img2[y2:y2+h2, x2:x2+w2] = face1_resize

    return img1, img2

# main code
ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--image1", required=True, # for multiple images, use specific flags
                help="Path to the Image")
ap.add_argument("-i2", "--image2", required=True,
                help="Path to the Image")
args = vars(ap.parse_args())

# load images
image1 = cv2.imread(args["image1"])
image2 = cv2.imread(args["image2"])

# perform face swap using function and display windows
result_image1, result_image2 = swap_faces(image1, image2)
cv2.imshow("1", result_image1)
cv2.imshow("2", result_image2)

cv2.waitKey(0)
