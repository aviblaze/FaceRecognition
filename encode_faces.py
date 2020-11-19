# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 04:53:24 2020

@author: MY PC
"""

import numpy as np
from PIL import Image
import pickle
import face_recognition

class GetFaceEncodings():

    def __init__(self,detection_model,landmarks_model):

        self.face_detection_model=detection_model
        self.face_alignment_model=landmarks_model
       
    def ImageToNumpyArray(self,path):
            return np.array(Image.open(path))

    def encode_faces(self,paths_to_images,output_path=None,return_encodings=False):
        
        paths_to_images=paths_to_images
        train_images_labels=[]
        train_images_vectors=[]
        for image_path in paths_to_images:

            image=self.ImageToNumpyArray(image_path)
            # Detect Face
            face_bbox=self.face_detection_model.DetectFaces(image)

            # Face Alignment
            aligned_face=self.face_alignment_model.Align(face_bbox,image)
            
            # Detect face on the aligned image
            aligned_face_bbox=self.face_detection_model.DetectFaces(aligned_face)
            
            # extract features
            face_encoding=face_recognition.face_encodings(aligned_face,[aligned_face_bbox],model='large',num_jitters=10)
            # store the encodings
            
            train_images_labels.append(image_path.split('/')[-2])
            train_images_vectors.append(face_encoding)

        if return_encodings==False:
            # write encodings to a file
            print('Info : dumping encodings ...')
            data={'encodings':train_images_vectors,'labels':train_images_labels}
            f=open(output_path+'/encodings.pickle','wb')
            f.write(pickle.dumps(data))
            f.close()

            print("Done.")
            return output_path+'/encodings.pickle'
        else:
            return train_images_vectors
    # ##################################################################

# faceencodings=GetFaceEncodings(['/home/overlord/Documents/Face Recognition/tiltedhead.jpg'],'/home/overlord/Documents/Face Recognition/my_model/saved_model',
#                                 '/home/overlord/Documents/Face Recognition/shape_predictor_68_face_landmarks.dat'
#                                 )
# faceencodings.encode_faces('/home/overlord/Documents/Face Recognition')