import json
import sys
#sys.tracebacklimit = 0
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import tensorflow as tf
import time
import enum
import zipfile
import matplotlib.pyplot as plt
import random
import glob

class MNIST:
    def __init__(self):
        (x,_),(y,_) = tf.keras.datasets.mnist.load_data()
    
        self.dataset = tf.expand_dims(tf.cast(tf.concat([x,y],axis=0),dtype=tf.float32)*(1.0/255.0)*2.0-1.0,axis=-1)
        self.dataset = tf.pad(self.dataset, [[0,0],[2,2],[2,2],[0,0]], constant_values = -1.0) #pad to 32x32
        
        self.stddev = tf.math.reduce_std(self.dataset,axis=[0,1,2],keepdims=True)
        self.mean = tf.math.reduce_mean(self.dataset,axis=[0,1,2],keepdims=True)
        
        self.dataset = (self.dataset-self.mean)/self.stddev
        #return dataset, self.stddev, self.mean
        
        self.shape = self.dataset.shape

    def gather(self, indices):
        return tf.gather(self.dataset, indices)

class Directory:
    def __init__(self, directory, extension, pad=None):
        pngs = glob.glob(f"{directory}/*.{extension}")
        self.dataset = [tf.io.decode_png(tf.io.read_file(png)) for png in pngs]
        self.dataset = tf.stack(self.dataset,axis=0)
        self.dataset = tf.cast(self.dataset,dtype=tf.float32)*(1.0/255.0)*2.0-1.0
        
        if pad is not None:
            self.dataset = tf.pad(self.dataset,pad,constant_values=-1.0) #pad to 128x80
        
        self.stddev = tf.math.reduce_std(self.dataset,axis=[0,1,2],keepdims=True)
        self.mean = tf.math.reduce_mean(self.dataset,axis=[0,1,2],keepdims=True)
        
        self.dataset = (self.dataset-self.mean)/self.stddev
        
        self.shape = self.dataset.shape

    def gather(self, indices):
        return tf.gather(self.dataset, indices)
    
class UltimateDoom:
    def __init__(self, file, pad=None):
        self.mean = tf.convert_to_tensor([-0.50083236, -0.58777652, -0.68647362])
        self.mean = tf.reshape(self.mean, [1,1,1,3])

        self.stddev = tf.convert_to_tensor([0.32425563, 0.31570265, 0.29139458])
        self.stddev = tf.reshape(self.stddev, [1,1,1,3])
        
        self.pad = pad
        
        with tf.device('/cpu:0'):
            self.dataset = tf.convert_to_tensor(np.load(file))
            self.shape = self.dataset.shape
            if pad is not None:
                self.shape = [self.shape[0], self.shape[1]+pad[1][0]+pad[1][1], self.shape[2]+pad[2][0]+pad[2][1], self.shape[3]]

    def gather(self, indices):
        data = tf.gather(self.dataset, indices)
        data = tf.cast(data,dtype=tf.float32)*(1.0/255.0*2.0)-1.0

        if self.pad is not None:
            data = tf.pad(data, self.pad, constant_values = -1.0)
        
        data = (data-self.mean)/self.stddev
        return data
