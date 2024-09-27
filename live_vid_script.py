import numpy as np
import argparse
import cv2

# load pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

    if len(faces) < 2:
        return None
    
    return faces # return all detected faces as bounding rects

# function to swap 2 face regions in the same image
def swap_faces(img):
    faces = detect_faces(img)

    if faces is None:
        print("Couldn't detect face contour in image(s).")
        return None
    
    faces = sorted(faces, key=lambda x:x[0])# sort faces by x-coordinate (l to r)

    # extract faces
    (x1, y1, w1, h1) = faces[0]
    (x2, y2, w2, h2) = faces[1]

    # crop
    face1_region = img[y1:y1+h1, x1:x1+w1]
    face2_region = img[y2:y2+h2, x2:x2+w2]

    # resize to match target bounds
    face1_resize = cv2.resize(face1_region, (w2,h2), interpolation=cv2.INTER_AREA)
    face2_resize = cv2.resize(face2_region, (w1,h1), interpolation=cv2.INTER_AREA)

    # replace face regions
    img[y1:y1+h1, x1:x1+w1] = face2_resize
    img[y2:y2+h2, x2:x2+w2] = face1_resize

    return img

# main code
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, # for multiple images, use specific flags
                help="Path to the Image")
args = vars(ap.parse_args())

# load images
img = cv2.imread(args["image"])

# perform face swap using function and display windows
result_image = swap_faces(img)
cv2.imshow("Face swap", result_image)

cv2.waitKey(0)
