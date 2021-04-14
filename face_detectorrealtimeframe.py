#for a stationary img ucan read the image manually
import cv2
from random import randrange

# load some pre trained data
trained_face_data=cv2.CascadeClassifier('haaracascades.xml')

# choose an image to detect faces
#img=cv2.imread('namo1.png')

#to capture video from webcam
webcam=cv2.VideoCapture(0) #0 stands for default webcam...for detecting faces in a video write name.mp4 instead of 0

#iterate forever over frames
while True:

    #read the current fame
    successful_frame_read,frame=webcam.read()  #we only want the frame


    #converting a colour image to greyscale s that we dont have to deal with lot of colours
    grayscaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


    #detect faces
    face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)   
 

    #draw rectangles around the faces(a&y coordinates,colour,thickness of the rectangle)
    #w&h are width and height of the rectangle
    for (x,y,w,h) in face_coordinates:
         cv2.rectangle(frame , (x,y), (x+w,y+h), (randrange(256), randrange(256),randrange(256)), 2) 
 

    #poping up the image to be detected
    cv2.imshow('AI based face detector by Rathin', frame)
    key=cv2.waitKey(1)

    #stop if Q key is pressed
    if key==81 or key==113:
         break

#release the videocapture object
webcam.release()

