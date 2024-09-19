import numpy as np
import argparse
import cv2

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