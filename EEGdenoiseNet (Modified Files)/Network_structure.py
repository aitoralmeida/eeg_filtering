import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, Input, Sequential
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras

# Author: Haoming Zhang

def fcNN(datanum):
  model = tf.keras.Sequential()
  model.add(Input(shape=(datanum,)))
  model.add(layers.Dense(datanum, activation=tf.nn.relu ))
  model.add(layers.Dropout(0.3))


  model.add(layers.Dense(datanum))
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Dense(datanum))
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Dense(datanum))
  model.summary()
  return model


def RNN_lstm(datanum):
  model = tf.keras.Sequential()
  model.add(Input(shape=(datanum,1)))
  model.add(layers.LSTM(1,return_sequences = True ))

  model.add(layers.Flatten())

  model.add(layers.Dense(datanum))
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Dense(datanum))
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Dense(datanum))
  model.summary()
  return model


def simple_CNN(datanum):
  model = tf.keras.Sequential()

  model.add(layers.Conv1D(64, 3, strides=1, padding='same',input_shape=[ datanum, 1]))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv1D(64, 3, strides=1, padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv1D(64, 3, strides=1, padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  #num4
  model.add(layers.Conv1D(64, 3, strides=1, padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Flatten())
  model.add(layers.Dense(datanum))

  model.build(input_shape=[ 1,datanum, 1] )
  model.summary()

  return model


# Resnet Basic Block module。
class Res_BasicBlock(layers.Layer):
  def __init__(self,kernelsize, stride=1):
    super(Res_BasicBlock, self).__init__()
    self.bblock = Sequential([layers.Conv1D(32,kernelsize,strides=stride,padding="same"),
                              layers.BatchNormalization(),
                              layers.ReLU(),
                              layers.Conv1D(16,kernelsize,strides=1,padding="same"),
                              layers.BatchNormalization(),
                              layers.ReLU(),
                              layers.Conv1D(32,kernelsize,strides=1,padding="same"),
                              layers.BatchNormalization(),
                              layers.ReLU()])
                              
    self.jump_layer = lambda x:x


  def call(self, inputs, training=None):

    #Through the convolutional layer
    out = self.bblock(inputs)

    #skip
    identity = self.jump_layer(inputs)

    output = layers.add([out, identity])  #layers下面有一个add，把这2个层添加进来相加。
    
    return output


class BasicBlockall(layers.Layer):
  def __init__(self, stride=1):
    super(BasicBlockall, self).__init__()

    self.bblock3 = Sequential([Res_BasicBlock(3),
                              Res_BasicBlock(3)
                              ])                      
    
    self.bblock5 = Sequential([Res_BasicBlock(5),
                              Res_BasicBlock(5)
                              ])                      

    self.bblock7 = Sequential([Res_BasicBlock(7),
                              Res_BasicBlock(7)
                              ])
                              
    self.downsample = lambda x:x


  def call(self, inputs, training=None):
 
    out3 = self.bblock3(inputs)
    out5 = self.bblock5(inputs)
    out7 = self.bblock7(inputs)

    out = tf.concat( values = [out3,out5,out7] , axis = -1)

    return out


def Complex_CNN(datanum):
  model = Sequential()
  model.add(layers.Conv1D(32 ,5,strides=1,padding="same",input_shape=[ datanum, 1]))
  model.add(layers.BatchNormalization())
  model.add( layers.ReLU())

  model.add(BasicBlockall())

  model.add(layers.Conv1D(32 ,1,strides=1,padding="same"))
  model.add(layers.BatchNormalization())
  model.add( layers.ReLU())

  model.add(layers.Flatten())
  model.add(layers.Dense(datanum))

  model.build(input_shape=[ 1,datanum, 1] )
  model.summary()
  
  return model

def Novel_CNN(input_size = ( 1024, 1)):
    inputs = Input(input_size)
    conv1 = layers.Conv1D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = layers.Conv1D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = AveragePooling1D(pool_size= 2)(conv1)

    conv2 = layers.Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = layers.Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = AveragePooling1D(pool_size= 2)(conv2)

    conv3 = layers.Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = layers.Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = AveragePooling1D(pool_size= 2)(conv3) #9

    conv4 = layers.Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = layers.Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = AveragePooling1D(pool_size = 2)(drop4)  #13

    conv5 = layers.Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)  #14
    conv5 = layers.Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)
    ###
    pool5 = AveragePooling1D(pool_size = 2)(drop5)

    conv6 = layers.Conv1D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool5)  #18
    conv6 = layers.Conv1D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    drop6 = layers.Dropout(0.5)(conv6)

    pool6 = AveragePooling1D(pool_size = 2)(drop6)

    conv7 = layers.Conv1D(2048, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool6)  #22
    conv7 = layers.Conv1D(2048, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    drop7 = layers.Dropout(0.5)(conv7)
    #######
    flatten1 = layers.Flatten()(drop7)
    #output1 = layers.Dense(2048)(flatten1)
    output1 = layers.Dense(1024)(flatten1)
    model = Model(inputs = inputs, outputs = output1)

    model.summary()
    return model