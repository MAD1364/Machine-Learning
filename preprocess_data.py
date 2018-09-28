#-*-coding:utf8-*-

import os
import cv2
import numpy as np

def read_data():
    '''
    read training data from EgoK-360 Dataset
    '''
   
    img_rows,img_cols,img_depth=128,128,16#resize parameters
    X_tr=[] 

    #Reading walking action class
    listing_in_dataset = os.listdir('../EgoK360_Data')
    listing_in_dataset.sort()
    nb_subdirectories = len(listing_in_dataset)
    for i in range(nb_subdirectories):
        print(listing_in_dataset[i])
        sub_directory = os.listdir('../EgoK360_Data/' + listing_in_dataset[i])
        print("Action Class " + listing_in_dataset[i])
        for vid in sub_directory:
            vid = '../EgoK360_Data/' + listing_in_dataset[i] + "/" + vid
            print("Video in current action class: " + vid)
            frames = []
            cap = cv2.VideoCapture(vid)
            fps = cap.get(5) # cv2.cv.CV_CAP_PROP_FPS ; cv2.CAP_PROP_FPS
            ### frames
            # **CHANGE HERE 
            for k in range(img_depth):
                ret, frame = cap.read()
                if (not ret):
                    cap.release()
                    cap = cv2.VideoCapture(vid)
                    ret, frame = cap.read()
                frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #break
            cap.release()
            #cv2.destroyAllWindows()
            input=np.array(frames)
            ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
            X_tr.append(ipt)
        print(listing_in_dataset[i] + " done")

    np.save('out'+str(img_rows)+'x'+str(img_cols)+'-'+str(img_depth)+'F', X_tr)
    return X_tr

