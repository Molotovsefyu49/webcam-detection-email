import cv2
import time
import os
import glob
from emailing import send_email
from threading import Thread

# initialize the video capture object to capture frames from the default camera (0)
video = cv2.VideoCapture(0)

# wait for 1 second to allow the camera to warm up
time.sleep(1)

first_frame = None
# to keep track of motion detected in the previous two iterations
status_list = []
# to count the number of images saved in the "images" folder
count = 1


# function to clean the "images" folder by removing all the PNG files
def clean_folder():
    print("clean_folder function started")
    images = glob.glob("images/*.png")
    for image in images:
        os.remove(image)
    print("clean_folder function ended")


# infinite loop to capture video frames, process them and detect motion
while True:
    status = 0  # initialize the status to 0 (no motion detected)
    # read a frame from the video stream
    check, frame = video.read()

    # convert the frame to grayscale and apply Gaussian blur to reduce noise
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame_gau = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # display the grayscale frame with Gaussian blur applied
    cv2.imshow("My video", gray_frame_gau)

    # if this is the first frame, set it as the reference frame
    if first_frame is None:
        first_frame = gray_frame_gau

    # calculate the absolute difference between the reference frame and the current frame
    delta_frame = cv2.absdiff(first_frame, gray_frame_gau)

    # threshold the delta frame to get the foreground objects and dilate them for better visibility
    thresh_frame = cv2.threshold(delta_frame, 60, 255, cv2.THRESH_BINARY)[1]
    dil_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # display the dilated frame with the foreground objects
    cv2.imshow("My video", dil_frame)

    # find contours (boundaries) of the foreground objects
    contours, check = cv2.findContours(dil_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop through each contour and check if it's big enough to be considered as motion
    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        rectangle = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

        # if a rectangle is drawn around a foreground object, set the status to 1 (motion detected)
        if rectangle.any():
            status = 1
            # save the current frame as a PNG image in the "images" folder
            cv2.imwrite(f"images/{count}.png", frame)
            count = count + 1

            # find the index of the latest image saved in the "images" folder
            all_images = glob.glob("images/*.png")
            index = int(len(all_images) / 2)
            image_with_object = all_images[index]

    # add the current status to the status_list and keep only the last two statuses
    status_list.append(status)
    status_list = status_list[-2:]

    # If an object is detected after a period of no detection, send an email
    if status_list[0] == 1 and status_list[1] == 0:
        email_thread = Thread(target=send_email, args=(image_with_object, ))
        email_thread.daemon = True
        clean_thread = Thread(target=clean_folder)
        clean_thread.daemon = True

        email_thread.start()

    # Show the current frame and wait for a key press
    cv2.imshow("Video", frame)
    key = cv2.waitKey(1)

    # If the 'q' key is pressed, break the loop and release the camera device
    if key == ord("q"):
        break

video.release()

clean_thread.start()