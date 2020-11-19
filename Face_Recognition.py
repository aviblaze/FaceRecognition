from matplotlib.pyplot import title
from tensorflow.python.framework import c_api_util
from encode_faces import GetFaceEncodings
from tkinter import filedialog
import tkinter
import argparse
import cv2
import pickle
import sys
import numpy as np
from face_detection import FaceDetection
from face_align import AlignFace

ap=argparse.ArgumentParser()

ap.add_argument("-m","--model",required=True,help="path to tensorflow face detection saved model")
ap.add_argument("-l","--landmarks",required=True,help="path to face landmarks detection model")
ap.add_argument("-o","--output",required=True,help="directory path for saving encodings")

args=vars(ap.parse_args())

def compare_faces(train_enc,test_enc,tol=0.3):
    train_enc=np.array(train_enc)
    train_enc=train_enc.reshape(train_enc.shape[0],train_enc.shape[2])
    test_enc=np.array(list(test_enc))
    test_enc=test_enc.reshape(test_enc.shape[0],test_enc.shape[2])
    euc_dist=np.linalg.norm(train_enc-test_enc,axis=1)
    #print(euc_dist)
    return np.where(euc_dist <= tol)[0]

# Load required models
face_detection_model_path=args['model']
landmarks_model_path=args['landmarks']

print('Info : Loading required models ...')
face_detection_model=FaceDetection(face_detection_model_path)
face_alignment_model=AlignFace(landmarks_model_path)
print('Done.')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# use gui to let user choose train images
main_win=tkinter.Tk()
main_win.withdraw()
images=filedialog.askopenfilenames(title='choose train images')

if len(images) > 0:
    encoder=GetFaceEncodings(detection_model=face_detection_model,landmarks_model=face_alignment_model)
    encodings_path=encoder.encode_faces(paths_to_images=images,output_path=args['output'])
else:
    print("Error : 0 files selected")
    sys.exit(0)

# use gui to let user choose test images
test_images=filedialog.askopenfilenames(title='choose test images')

if len(test_images) > 0:
    for test_image in test_images:

        name=(test_image.split('/')[-1]).split('.')[0]
        
        #test_image=cv2.cvtColor(cv2.imread(test_image),cv2.COLOR_BGR2RGB)
        test_image_enc=encoder.encode_faces(paths_to_images=[test_image],return_encodings=True)

        # load train images encodings pickle file
        encodings=pickle.loads(open(encodings_path,'rb').read())
        train_labels=encodings['labels']
        train_encodings=encodings['encodings']
       
        # Get matches for the test_image
        matches=compare_faces(train_encodings,test_image_enc)
        
        if len(matches)==0:
            label='Unknown'
        else:
            vote_count={}
            for match in matches:
                val=vote_count.get(train_labels[match],0)
                vote_count[train_labels[match]]=val+1
            label=max(vote_count,key=vote_count.get)

        test_image=cv2.imread(test_image)
        # detect face on test image
        y1,x1,y2,x2=face_detection_model.DetectFaces(test_image)

        # draw bounding box around detected face
        cv2.rectangle(test_image,(x1,y1),(x2,y2),(0,255,0),10)

        # write the label name
        cv2.putText(test_image,label,(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),2,bottomLeftOrigin=False)
        cv2.namedWindow(name,cv2.WINDOW_NORMAL)
        cv2.imshow(name,test_image)
        cv2.waitKey(0)
else:
    print('0 images selected')
    sys.exit(0)

