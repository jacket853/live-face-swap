import numpy as np
import argparse
import cv2

# function to detect largest contour
def detect_face_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11,11), 0)
    edged = cv2.Canny(blurred, 30, 150)

    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    face_contour = max(cnts, key=cv2.contourArea) # assuming largest contour is the face
    # this may be a simplification (we will see as we input different images)

    return face_contour

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