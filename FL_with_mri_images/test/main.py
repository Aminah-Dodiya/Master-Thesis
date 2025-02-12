import available_datasets
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import keras
from keras.utils import to_categorical
from keras import layers
from keras import models
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

#print(tf._version_)

# disable tensorflow deprecation warnings
import logging
logging.getLogger('tensorflow').disabled=True

# only allocate the GPU RAM actually required
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

# load the datasets
#flavor = 'N4cor_Warped_bet'
flavor = 'N4cor_WarpedSegmentationPosteriors2'
adni2 = available_datasets.load_ADNI2_data(x_range_from = 32, x_range_to = 161,
    y_range_from = 90, y_range_to = 135,
    z_range_from = 20, z_range_to = 159, flavor = flavor)

# display distribution of diagnoses
print(adni2['groups']['Group'].value_counts())

#%%
# combine datasets
images = adni2['images']
labels = adni2['labels']
labels = to_categorical(labels)
groups = adni2['groups']
covariates = adni2['covariates']
numfiles = labels.shape[0]

# Split data into training/validation and holdout test data
from sklearn.model_selection import train_test_split
import numpy as np

train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=0.2, stratify = labels, random_state=1)

import pandas as pd
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc
import keras
from keras import layers
from keras.layers import Input, Conv3D, BatchNormalization, Dense
from keras.layers import AveragePooling3D, GlobalAveragePooling3D, MaxPooling3D
from keras.models import Model
from keras.layers import ReLU, concatenate
import keras.backend as K

from keras import models
from scipy import ndimage

def DenseNet(ip_shape, op_shape, filters = 3):
   '''
   declaring a DenseNet model.
   Paper: https://arxiv.org/pdf/1608.06993.pdf
   Code adapted from https://towardsdatascience.com/creating-densenet-121-with-tensorflow-edbc08a956d8

   ip_shape: expected input shape
   op_shape: expected output shape
   filters: number of filters to be used
   '''

   #batch norm + relu + conv
   def bn_rl_conv(x,filters,kernel=1,strides=1):

       x = BatchNormalization()(x)
       x = ReLU()(x)
       x = Conv3D(filters, kernel, strides=strides,padding = 'same')(x)
       return x

   def dense_block(x, repetition=4):

       for _ in range(repetition):
           y = bn_rl_conv(x, filters=8)
           y = bn_rl_conv(y, filters=8, kernel=3)
           x = concatenate([y,x])
       return x

   def transition_layer(x):

       x = bn_rl_conv(x, K.int_shape(x)[-1] //2 )
       x = AveragePooling3D(2, strides = 2, padding = 'same')(x)
       return x



   input = Input(ip_shape)
   x = Conv3D(10, 7, strides = 2, padding = 'same')(input)
   x = MaxPooling3D(3, strides = 2, padding = 'same')(x)

   brc_in_blocks = [3,3]
   for repetition in brc_in_blocks:                      #[6,12,24,16]:
       d = dense_block(x, repetition)
       x = transition_layer(d)

   #x = GlobalAveragePooling3D()(d)

   x = layers.MaxPooling3D(pool_size=(2, 2, 2))(d)
   #Notice the input to pooling layer. d. this overwrites the last x variable,
   #and nullifies the transition layer computations done on last dense blocks output.
   # i.e last transition layer is not connected to the graph
   #TLDR: No transition layer after last dense block.

   # FC layer
   # added on 3.3.23
   x=layers.Activation('relu')(x)
   #x=layers.Dropout(rate = 0.3)(x)
   x=layers.Dropout(rate = 0.4)(x)

   x = layers.Flatten()(x)
   output = Dense(op_shape, activation = 'softmax')(x)

   model = Model(input, output)
   return model

model = DenseNet(images.shape[1:], labels.shape[1])
opt = keras.optimizers.Adam(learning_rate=0.001)
#opt = keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])

# Fit model to training data
batch_size = 8
epochs = 25
hist = model.fit(train_x, train_y, batch_size=batch_size,
                        epochs=epochs, verbose=1,)

mymodel = hist.model

# Calculate accuracy for holdout test data
scores = mymodel.evaluate(test_x, test_y, batch_size=batch_size) #, verbose=0
print("Test %s: %.2f%%" % (mymodel.metrics_names[1], scores[1]*100))

