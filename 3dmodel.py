#-*-coding:utf8-*-

# Source: http://learnandshare325.blogspot.hk/2032/06/3d-cnn-in-keras-action-recognition.html

import os
#Use for debugging and not immediately running on GPU server
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers import BatchNormalization
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from keras.callbacks import TensorBoard

import tensorflow as tf
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import datetime
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pdb
import json

import i3d_inception

from preprocess_data import read_data
from generator import Generator


#read ids and labels from the respective files as dictionaries
partition = json.load(open("partition_dict.txt"))
labels = json.load(open("labels_dict.txt"))

params = {'dim': (64, 224, 224),
          'batch_size': 40,
          'n_classes': 45,
          'n_channels': 3,
          'shuffle': True}

#Use generators to deal with data that becomes too large for cpu to send to gpu
training_generator = Generator(partition['train'], labels, **params)
validation_generator = Generator(partition['validation'], labels, **params)

nb_classes = 45

model = i3d_inception.Inception_Inflated3d(include_top=True, weights=None, input_tensor=None, input_shape=(64, 224, 224, 3), dropout_prob = 0.5, classes=nb_classes)
sgd = SGD(lr=0.01, momentum = 0.9, nesterov=True) #Can choose to use this form of gradient descent
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['mse', 'accuracy'])
model.summary()#print the model
hist = model.fit_generator(generator=training_generator,
                           steps_per_epoch=len(training_generator),
                           validation_data=validation_generator,
                           workers=4,
                           use_multiprocessing=True)


# Save model
now = str(datetime.datetime.now()).split('.')[0]
model.save('./models/'+now+"-model.h5")
# Evaluate the model
score = model.evaluate(X_val_new,y_val_new,batch_size=batch_size)
# Print the results
print('**********************************************')
print('Test score:', score)
print('History', hist.history)





