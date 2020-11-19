## This file is to test the Exported model from the notebook
# Load th Model

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from object_detection.utils import label_map_util

class FaceDetection():

    def __init__(self,model_path):

        SAVED_MODEL_PATH=model_path #'/home/overlord/Documents/Face Recognition/my_model/saved_model'
        # Load model

        self.detect_fn=self.LoadModel(SAVED_MODEL_PATH)

    def LoadModel(self,path):

        start_time=time.time()

        model=tf.saved_model.load(path)
        end_time=time.time()

        elapsed_time=end_time-start_time
        #print('Done.Elapsed time : {} seconds'.format(elapsed_time))

        return model 

    def DetectFaces(self,image):

        ######################Detection#########################
        hei,wid=image.shape[:2]
        image_tnsr=tf.convert_to_tensor(image)

        image_tnsr=image_tnsr[tf.newaxis,...]

        detections=self.detect_fn(image_tnsr)

        num_detections=int(detections.pop('num_detections'))
        detections={key : value[0, : num_detections].numpy() for key,value in detections.items()}

        detections['num_detections']=num_detections

        detections['detection_classes']=detections['detection_classes'].astype(np.int64)
        
        y1,x1,y2,x2=detections['detection_boxes'][np.argmax(detections['detection_scores'])]
        
        high_score_rect=(int(y1*hei),int(x1*wid),int(y2*hei),int(x2*wid))
        
        return high_score_rect


# face_detection_model=FaceDetection('/home/overlord/Documents/Face Recognition/my_model/saved_model')

# x1,y1,x2,y2=face_detection_model.DetectFaces('/home/overlord/Documents/Face Recognition/tiltedhead.jpg')
# print(x1,y1,x2,y2)
# # reload matplotlib as the TF2.0 changes the engine
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

# img=cv2.imread('/home/overlord/Documents/Face Recognition/tiltedhead.jpg')
# cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),5)
# plt.imshow(img)
# plt.show()

# x1,y1,x2,y2=face_detection_model.DetectFaces('/home/overlord/Documents/Face Recognition/aligned_tiltedhead.jpg')
# img=cv2.imread('/home/overlord/Documents/Face Recognition/aligned_tiltedhead.jpg')
# cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),5)
# plt.imshow(img)
# plt.show()