import numpy as np
import argparse
import cv2

###### JACK'S CODE ######

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


###### CASPAR'S CODE ######
def get_image(frame):
    cv2.imshow("img",frame)
    # return image

# Start video capture from webcam, 0 argument opens default camera
cap = cv2.VideoCapture(0)

while True:
    # Capture each frame, loops through the frames
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # display the frames in real time
    cv2.imshow("Original", frame)

    # waits for the key press
    key = cv2.waitKey(1) &0xFF

    if key == ord('p'):
        # create a filename for each saved frame
        filename = "saved_frame.png"
        # save the current frame to a file with the corrsponding name
        cv2.imwrite(filename, frame)

        break

    # Break the loop on 'q' key press
    if key == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# parsing
ap = argparse.ArgumentParser()
ap.add_argument("saved_frame.png")
args = vars(ap.parse_args())

# load image
img = cv2.imread(args["saved_frame.png"])

# perform face swap using function and display windows
result_image = swap_faces(img)
cv2.imshow("Face swap", result_image)

cv2.waitKey(0)
