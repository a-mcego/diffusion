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

import posenc

from PIL import Image

if not os.path.exists("outs"):
    os.mkdir("outs")

class RangeShuffler:
    def __init__(self, minval, maxval):
        self.minval = minval
        self.maxval = maxval
        self.current = tf.zeros(shape=[0],dtype=tf.int64)
    
    def new_shuffled(self):
        newrange = tf.random.shuffle(tf.range(start=self.minval, limit=self.maxval, dtype=tf.int64))
        self.current = tf.concat([self.current, newrange],axis=-1)
    
    def get_batch(self,batch_size):
        while self.current.shape[0] < batch_size:
            self.new_shuffled()
    
        ret = self.current[:batch_size]
        self.current = self.current[batch_size:]
        return ret

#returns float32, pixel colors in range [-1.0,1.0]
def get_mnist_pics_only():
    (x,_),(y,_) = tf.keras.datasets.mnist.load_data()
    return tf.expand_dims(tf.cast(tf.concat([x,y],axis=0),dtype=tf.float32)*(1.0/255.0)*2.0-1.0,axis=-1)

dataset = get_mnist_pics_only()

prm = {} #parameters
prm['N_STEPS'] = 99 #how many diffusion steps we use
prm['SIZE_Y'] = dataset.shape[-3]
prm['SIZE_X'] = dataset.shape[-2]
prm['CHANNELS'] = dataset.shape[-1]
prm['IMAGE_SIZE'] = prm['SIZE_Y']*prm['SIZE_X']*prm['CHANNELS']
prm['LAYERS'] = 6
prm['MODEL_SIZE'] = 128
prm['BATCH_SIZE'] = 32
prm['LEARNING_RATE'] = 0.0005
prm['LEARNING_RATE_MIN'] = 0.00001
prm['ADAM_EPSILON'] = 1e-4
prm['N_TEST_GENERATIONS'] = 49
prm['FILTER_SIZE'] = 3
prm['USE_RANGE_SHUFFLER'] = True

for key in prm:
    print(f"{key}={prm[key]} ",end="")
print()

class NoisingProcess:
    def __init__(self, timesteps):
        self.timesteps = timesteps
        
        self.image_coef = tf.linspace(start=1.0, stop=0.0, num=timesteps+1)
        self.noise_coef = tf.linspace(start=0.0, stop=1.0, num=timesteps+1)

    def direct(self, step, image, noise=None):
        assert step>=0 and step<=self.timesteps
        
        if noise is None:
            noise = tf.random.normal(shape=image.shape, stddev=1.0)
        
        ret = self.image_coef[step]*image + noise*self.noise_coef[step]
        return ret, noise
        
    def get_noise_stddev_for_step(self, step):
        return self.noise_coef[step]


def reduce(matrix, axis):
    mean = tf.math.reduce_mean(matrix, axis=axis, keepdims=True)
    stddev = tf.math.reduce_std(matrix, axis=axis, keepdims=True)
    return (matrix - mean)/(stddev + 1e-2)

#pad image *amount* amount on all sides, up/down and left/right, such that it repeats
def pad_image_repeat(img, amount):
    left =  img[:,:, -amount: ,:]
    right = img[:,:, :amount  ,:]
    img = tf.concat([left,img,right],axis=2)

    top =    img[:, -amount: ,:,:]
    bottom = img[:, :amount  ,:,:]
    img = tf.concat([top,img,bottom],axis=1)
    return img


class NoisePredictorConv(tf.keras.Model):
    def __init__(self):
        super(NoisePredictorConv,self).__init__()
        self.timestep_posenc = posenc.get_zero_to_one_posenc(prm['N_STEPS'], prm['MODEL_SIZE'])
        
        self.image_posenc = tf.keras.layers.Embedding(prm['SIZE_X']*prm['SIZE_Y'], prm['MODEL_SIZE'])

        self.prologue = tf.keras.layers.Conv2D(prm['MODEL_SIZE'], (1,1), use_bias=False)
        self.convs = [tf.keras.layers.Conv2D(prm['MODEL_SIZE'], (prm['FILTER_SIZE'],prm['FILTER_SIZE']), activation=tf.nn.gelu, padding='valid') for _ in range(prm['LAYERS'])]
        self.epilogue = tf.keras.layers.Conv2D(prm['CHANNELS'], (1,1), use_bias=False)

    def call(self, data, timestep_ns):
        #data shape:        BATCH, SIZEY, SIZEX, CHANNELS
        #timestep_ns shape: BATCH
        
        orig_shape = data.shape
        
        image_posenc = self.image_posenc(tf.range(prm['SIZE_X']*prm['SIZE_Y'],dtype=tf.int64))
        #image_posenc shape: 784, 256
        image_posenc = tf.reshape(image_posenc, [1,prm['SIZE_Y'],prm['SIZE_X'],prm['MODEL_SIZE']])
        #image_posenc shape: 1, 28, 28, 256
        
        
        timesteps = tf.gather(self.timestep_posenc, timestep_ns)
        #shape: BATCH, MODEL_SIZE
        timesteps = tf.expand_dims(timesteps,axis=1)
        timesteps = tf.expand_dims(timesteps,axis=1)
        #shape: BATCH, 1, 1, MODEL_SIZE
        output = data
        output = self.prologue(output)
        output += timesteps
        #20.0 is an arbitrary constant to make it actually see the posenc.
        #i want to normalize these things somehow at some point.
        output += image_posenc*20.0 
        for layer in self.convs:
            output = pad_image_repeat(output, prm['FILTER_SIZE']//2)
            output = layer(output)
        output = self.epilogue(output)
        return output
        
step=1

def lr_scheduler():
    global step
    minimum = prm['LEARNING_RATE_MIN']
    calculated = prm['LEARNING_RATE']*(2.0**(-step/3000.0))
    return max(minimum,calculated)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler,amsgrad=True,epsilon=prm['ADAM_EPSILON'])
noising_process = NoisingProcess(prm['N_STEPS'])
noise_predictor = NoisePredictorConv()

@tf.function
def do_step(inputdata, timesteps, target, training):
    losses = []
    
    if training:
        with tf.GradientTape() as tape:
            output = noise_predictor(inputdata, timesteps)
            loss = tf.reduce_mean(tf.square(target-output))
            losses.append(loss)
        gradients = tape.gradient(losses, noise_predictor.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        optimizer.apply_gradients(zip(gradients, noise_predictor.trainable_variables))
    else:
        #TODO: not implemented yet
        pass
        
    return losses

#organize pictures in a neat grid
def gridify(allpics):
    n_pics = allpics.shape[0]
    height = tf.cast(tf.sqrt(tf.cast(n_pics,tf.float32)),dtype=tf.int64)
    width = n_pics // height + (1 if (n_pics % height > 0) else 0)
    
    total = height*width
    missing = total-n_pics
    if missing > 0:
        allpics = tf.concat([allpics,tf.zeros(shape=[missing,allpics.shape[-3],allpics.shape[-2]])],axis=0)
    allpics = tf.reshape(allpics, shape=[height, width, allpics.shape[-3],allpics.shape[-2]])
    allpics = tf.transpose(allpics, [0,2,1,3])
    allpics = tf.reshape(allpics, shape=[allpics.shape[0]*allpics.shape[1], allpics.shape[2]*allpics.shape[3]])
    return allpics

training_sess_id = str(random.randrange(1000000000))

losses = []

if prm['USE_RANGE_SHUFFLER']:
    data_shuffler = RangeShuffler(0,dataset.shape[0])
    timestep_shuffler = RangeShuffler(1,prm['N_STEPS']+1)

starttime = time.time()

#training loop:
while True:
    if prm['USE_RANGE_SHUFFLER']:
        ids = data_shuffler.get_batch(prm['BATCH_SIZE'])
        timestep_ids = timestep_shuffler.get_batch(prm['BATCH_SIZE'])
    else:
        ids = tf.random.uniform(shape=[prm['BATCH_SIZE']], maxval=dataset.shape[0], dtype=tf.int64)
        timestep_ids = tf.random.uniform(shape=[prm['BATCH_SIZE']], minval=1, maxval=prm['N_STEPS']+1, dtype=tf.int64)
    
    originals = tf.gather(dataset, ids)
    
    targets = []
    batch = []
    
    for ex_id in range(prm['BATCH_SIZE']):
        ex, target = noising_process.direct(timestep_ids[ex_id], originals[ex_id])
        
        batch.append(ex)
        targets.append(target)
    
    batch = tf.stack(batch,axis=0)
    targets = tf.stack(targets,axis=0)
    
    losses.extend(do_step(batch, timestep_ids, targets, training=True))
    
    if step%128 == 0:
    
        totaltime = time.time()-starttime
        starttime = time.time()
        
        losses = tf.reduce_mean(tf.stack(losses,axis=0))
        
        print(f"{totaltime} {step} {step*prm['BATCH_SIZE']} ",end="")
        
        print(f"{losses.numpy()} {lr_scheduler()}",end="")
        print()
        losses = []
        
        if step % 1024 == 0:
            #we generate pictures from noise here
            n_gen = prm['N_TEST_GENERATIONS']

            picture = tf.random.normal(shape=[n_gen,dataset.shape[1],dataset.shape[2],dataset.shape[3]], stddev=1.0, mean=0.0)
            for ts in range(prm['N_STEPS'], 0, -1):
                ts_tf = tf.expand_dims(tf.convert_to_tensor(ts),axis=0)
                ts_tf = tf.broadcast_to(ts_tf,[n_gen])
                picture -= noise_predictor(picture, ts_tf)*noising_process.get_noise_stddev_for_step(ts)
                picture,_ = noising_process.direct(ts-1, picture)
            
            picture = gridify(picture)
            picture = tf.cast(tf.clip_by_value(((picture+1.0)*0.5)*255.0,0.0,255.0),dtype=tf.uint8)
            
            im = Image.fromarray(picture.numpy())
            im.save(f"outs/sess{training_sess_id}_{step}.png")
    
    step += 1
