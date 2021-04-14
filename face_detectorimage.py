#for a stationary img ucan read the image manually
import cv2
from random import randrange

# load some pre trained data
trained_face_data=cv2.CascadeClassifier('haaracascades.xml')

#choose an image to detect faces
img=cv2.imread('namo1.png')




#converting a colour image to greyscale s that we dont have to deal with lot of colours
grayscaled_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detect faces
face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)   
 

#draw rectangles around the faces(a&y coordinates,colour,thickness of the rectangle)
#w&h are width and height of the rectangle
for (x,y,w,h) in face_coordinates:
     cv2.rectangle(img, (x,y), (x+w,x+h), (randrange(256), randrange(256),randrange(256)), 2) 
 

#poping up the image to be detected
cv2.imshow('AI based face detector by Rathin', img)
cv2.waitKey()

