import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import ML_my_func as my
import sys
import numpy as np

from solve_cudnn_error import *

solve_cudnn_error()

'''
def creat_valid_set(image_data, label_data, train_percentage=0.8):
    np.random.seed(123)
    mark = np.random.choice(image_data.shape[0], image_data.shape[0], replace=False)
    length = int(mark.shape[0] * train_percentage)
    mark_train = mark[0:length]
    mark_valid = mark[length:]
    img_train_data = image_data[mark_train]
    img_valid_data = image_data[mark_valid]
    label_train_data = label_data[mark_train]
    label_valid_data = label_data[mark_valid]

    return img_train_data, img_valid_data, label_train_data, label_valid_data
'''

(train_image, train_label), (test_image, test_label) = mnist.load_data()
print(len(train_image))
tmp_lenght = len(train_image)
print("\t[Info] train data={:7,}".format(len(train_image)))  
print("\t[Info] test  data={:7,}".format(len(test_image)))  

print("\t[Info] Shape of train data=%s" % (str(train_image.shape)))  
print("\t[Info] Shape of train label=%s" % (str(train_label.shape))) 
Label = np.hstack((train_label, test_label))
one_hot = my.one_hot_encoder(Label)
one_hot_train = one_hot[:len(train_image), :]
one_hot_test = one_hot[len(train_image):, :]

train_image = train_image / 127.5 - 1.
train_image = np.expand_dims(train_image, axis=3)
test_image = test_image / 127.5 - 1.
test_image = np.expand_dims(test_image, axis=3)
print(train_image.shape)

# exit() 
Input_shape = (train_image.shape[1], train_image.shape[2], train_image.shape[3])
Inputs_data = Input(shape=Input_shape, name='cnn_input')
Cnn_model = Conv2D(64, (6, 6), activation='relu', padding='same')(Inputs_data)
Cnn_model = Conv2D(64, (6, 6), activation='relu', padding='same')(Cnn_model)
Cnn_model = MaxPooling2D((2, 2), padding='same')(Cnn_model)
Cnn_model = Dropout(0.2)(Cnn_model)
Cnn_model = Conv2D(32, (6, 6), activation='relu', padding='same')(Cnn_model)
Cnn_model = Conv2D(32, (6, 6), activation='relu', padding='same')(Cnn_model)
Cnn_model = MaxPooling2D((2, 2), padding='same')(Cnn_model)
Cnn_model = Dropout(0.2)(Cnn_model)
Cnn_model = Conv2D(16, (6, 6), activation='relu', padding='same')(Cnn_model)
Cnn_model = Dropout(0.2)(Cnn_model)
Cnn_model = Conv2D(8, (6, 6), activation='relu', padding='same')(Cnn_model)
Cnn_model = Flatten()(Cnn_model)
# Cnn_model = Dense(512, activation='relu')(Cnn_model)
Cnn_model = Dense(256, activation='relu')(Cnn_model)
Cnn_model = Dropout(0.2)(Cnn_model)
Cnn_model = Dense(128, activation='relu')(Cnn_model)
# Cnn_model = Dropout(0.2)(Cnn_model)
Output = Dense(one_hot_train.shape[1], activation='softmax')(Cnn_model)

Classificator = Model(Inputs_data, Output)
Classificator.summary()

opt = Adam(lr=0.001, decay=0.01)
Classificator.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
Classificator.fit(train_image, one_hot_train, epochs=50, batch_size=200)
Classificator.save("mnist_cnn_classificator_10_06_e50_b200_tf2.h5")

score_train = Classificator.evaluate(train_image, one_hot_train)
score_valid = Classificator.evaluate(test_image, one_hot_test)

print ('\nTrain Acc:', score_train[1])
print ('\nValid Acc:', score_valid[1])
exit()
