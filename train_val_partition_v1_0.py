import os
import json
import numpy as np
import cv2

super_actions = os.listdir("../EgoK360_Data/")

action_dictionary = json.load(open("actions.txt"))

partition = {'train':[], 'validation':[]}
labels = {}

#num_actions = 0

train_directory = "./EgoK-360_training_0/"

try:
    os.mkdir(train_directory)
except OSError:
    print("Creation of the directory %s failed." % train_directory)
else:
    print("Successfully created the directory %s." % train_directory)

img_rows, img_cols, img_depth = 224, 224, 64

for super_action in sorted(super_actions):
    actions = os.listdir("../EgoK360_Data/" + super_action)
    print(super_action)
    for action in sorted(actions):
        print(action)
        videos = os.listdir("../EgoK360_Data/" + super_action + "/" + action)
        print("Processing " + action + " in " + super_action + "...")
        num_videos = len(videos)
        num_validation_videos = int(num_videos * 0.2)
        for i, video in enumerate(videos):
            if (i <= num_videos - num_validation_videos):
                partition['train'].append(str(i) + "_" + super_action + "__" + action)
            else:
                partition['validation'].append(str(i) + "_" + super_action + "__" + action)

            vid = '../EgoK360_Data/' + super_action + "/" + action + "/" + video
            print("Video in current action class: " + vid)
            frames = []
            cap = cv2.VideoCapture(vid)
            #fps = cap.get(5) # cv2.cv.CV_CAP_PROP_FPS ; cv2.CAP_PROP_FPS
            ### frames
            # **CHANGE HERE 
            for k in range(img_depth):
               ret, frame = cap.read()
               if (not ret):
                   cap.release()
                   cap = cv2.VideoCapture(vid)
                   ret, frame = cap.read()
               frame = cv2.resize(frame, (img_rows,img_cols),interpolation=cv2.INTER_AREA)
               gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
               frames.append(gray)
                   #if cv2.waitKey(1) & 0xFF == ord('q'):
                        #break
            cap.release()
            #cv2.destroyAllWindows()
            input=np.array(frames)
            ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
            ipt=np.rollaxis(ipt,2,0)
            np.save(train_directory + str(i) + "_" + super_action + "_" + action, ipt)
            print(video + " done")

            labels[str(i) + "_" + super_action + "__" + action] = action_dictionary[action]
        #num_actions += 1

#for set, action in partition.items():
 #   print(set + str(action))

#for key, value in labels.items():
 #   print(key + ": " + str(value))

print(len(labels))

json.dump(partition, open("partition_dict.txt", 'w'))
json.dump(labels, open("labels_dict.txt", 'w'))
