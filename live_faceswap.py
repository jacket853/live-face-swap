import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

# function that updates the HSV values of the trackbars 
def nothing(x):
    pass

# function to display a histogram 
def display_histogram(frame):
    color = ('b','g','r')

    # loops through frames and colors to get individual pixel values
    for i, col in enumerate(color):
        histr = cv2.calcHist([frame], [i], None, [256], [0, 256])
        plt.plot(histr, color = col)
        plt.xlim([0, 256])
    
    plt.show()

def get_image(frame):
    cv2.imshow("img",frame)
    # return image


# Start video capture from webcam, 0 argument opens default camera
cap = cv2.VideoCapture(0)

# keeps track of saved images
image_counter = 0
saved_images = []

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

    # displays histogram on the press of h
    if key == ord('h'):
        display_histogram(frame)

    if key == ord('p'):
        # create a filename for each saved frame
        filename = f'saved_frame_{image_counter}.png'
        # save the current frame to a file with the corrsponding name
        cv2.imwrite(filename, frame)
        
        saved_images.append(filename)
        image_counter += 1


    # Break the loop on 'q' key press
    if key == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# display the saved images
for image_file in saved_images:
    img = cv2.imread(image_file)

    if img is not None:
        cv2.imshow(f"{image_file}", img)

        cv2.waitKey(0)
        cv2.destroyWindow(f"{image_file}")

cv2.destroyAllWindows()

# cv2.imshow("img", image)
# cv2.waitKey(0)

