
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import dlib


class AlignFace():

    def __init__(self,path_to_landmark_detector):

        self.face_landmarks_detector=dlib.shape_predictor(path_to_landmark_detector)

    def Shape2NP(self,shapes):

        coords=np.zeros((68,2),dtype=np.int32)
        for i in range(0,68):
            coords[i]=(shapes.part(i).x,shapes.part(i).y)

        return coords

    def Align(self,bbox,image):
        
        x1,y1,x2,y2=bbox
        bbox=dlib.rectangle(x1,y1,x2,y2)
        gray_img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #face_landmarks_detector=dlib.shape_predictor(path_to_landmark_detector)
        landmarks=self.face_landmarks_detector(gray_img,bbox)
        landmarks=self.Shape2NP(landmarks)
        
        # left eye coordinates are from 37-42, right eye coordinates are from 43-48 in 1-index format
        left_eye_ptns=landmarks[36:42]
        right_eye_ptns=landmarks[42:48]
        
        # caluclate angle between eyes
        left_eye_center=left_eye_ptns.mean(axis=0).astype('int')
        right_eye_center=right_eye_ptns.mean(axis=0).astype('int')
    
        
        dX=right_eye_center[0]-left_eye_center[0]
        dY=right_eye_center[1]-left_eye_center[1]
        angle=np.degrees(np.arctan2(dY,dX))

        eyes_center=((right_eye_center[0]+left_eye_center[0])//2,(right_eye_center[1]+left_eye_center[1])//2)
    
        # Get rotation matrix keeping scale as 1
        M=cv2.getRotationMatrix2D(eyes_center,angle,1)
    
        output=cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
        
        return output

# face_aline=AlignFace('/home/overlord/Documents/Face Recognition/shape_predictor_68_face_landmarks.dat')
# bbox=dlib.rectangle(223,388,713,977) #223,388,713,977 ,x1,y1,x2,y2
# img=cv2.imread('/home/overlord/Documents/Face Recognition/tiltedhead.jpg')
# face_aline.Align(bbox,img)
