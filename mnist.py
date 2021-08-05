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
import glob

def get_files_from_directory(directory, extension):
    pngs = glob.glob(f"{directory}/*.{extension}")
    imgs = [tf.io.decode_png(tf.io.read_file(png)) for png in pngs]
    imgs = tf.stack(imgs,axis=0)
    imgs = tf.cast(imgs,dtype=tf.float32)*(1.0/255.0)*2.0-1.0
    return imgs

dataset = get_mnist_pics_only()
dataset = tf.pad(dataset, [[0,0],[2,2],[2,2],[0,0]], constant_values = -1.0) #pad to 32x32

#dataset = get_files_from_directory("Q:\\doom\\out", "png")
#dataset = tf.pad(dataset,[[0,0],[0,8],[0,0],[0,0]],constant_values=-1.0) #pad to 128x80

dataset_stddev = tf.math.reduce_std(dataset,axis=[0,1,2],keepdims=True)
dataset_mean = tf.math.reduce_mean(dataset,axis=[0,1,2],keepdims=True)
dataset = (dataset-dataset_mean)/dataset_stddev

print(f"Dataset shape: {dataset.shape}")

prm = {} #parameters
prm['N_STEPS'] = 199 #how many diffusion steps we use
prm['SIZE_Y'] = dataset.shape[-3]
prm['SIZE_X'] = dataset.shape[-2]
prm['CHANNELS'] = dataset.shape[-1]
prm['IMAGE_SIZE'] = prm['SIZE_Y']*prm['SIZE_X']*prm['CHANNELS']
prm['BATCH_SIZE'] = 32
prm['LEARNING_RATE_START'] = 0.001
prm['LEARNING_RATE_MIN'] = 0.0001
prm['ADAM_EPSILON'] = 1e-4
prm['N_TEST_GENERATIONS'] = 49
prm['PRINT_TIME'] = 256 #how many steps between prints
prm['GENERATE_TIME'] = prm['PRINT_TIME']*8 # how many steps between generated outputs saved to .png
prm['USE_TIMESTEP_EMBEDDINGS'] = True
prm['USE_IMAGE_POSENC'] = False

for key in prm:
    print(f"{key}={prm[key]} ",end="")
print()

class NoisingProcess:
    def __init__(self, timesteps):
        self.timesteps = timesteps
        
        #sqrt() so that when we add the two kinds of noise together,
        #we get an image that has stddev=1
        self.image_coef = tf.math.sqrt(tf.linspace(start=1.0, stop=0.0, num=timesteps+1))
        self.noise_coef = tf.math.sqrt(tf.linspace(start=0.0, stop=1.0, num=timesteps+1))

    def direct(self, step, image, noise=None):
        assert step>=0 and step<=self.timesteps
        
        if noise is None:
            noise = tf.random.normal(shape=image.shape, stddev=1.0)
        
        ret = self.image_coef[step]*image + noise*self.noise_coef[step]
        return ret, noise
        
    def get_noise_stddev_for_step(self, step):
        return self.noise_coef[step]
        
    def get_image_stddev_for_step(self, step):
        return self.image_coef[step]


def reduce(matrix, axis):
    mean = tf.math.reduce_mean(matrix, axis=axis, keepdims=True)
    stddev = tf.math.reduce_std(matrix, axis=axis, keepdims=True)
    return (matrix - mean)/(stddev + 1e-3)

#pad image *amount* amount on all sides, up/down and left/right, such that it repeats
def pad_image_repeat(img, amount):
    left =  img[:,:, -amount: ,:]
    right = img[:,:, :amount  ,:]
    img = tf.concat([left,img,right],axis=2)

    top =    img[:, -amount: ,:,:]
    bottom = img[:, :amount  ,:,:]
    img = tf.concat([top,img,bottom],axis=1)
    return img

def normalize_features(x):
    oldshape = x.shape
    x = tf.reshape(x,[x.shape[0],-1,x.shape[3]])
    x = reduce(x, axis=-2)
    x = tf.reshape(x,oldshape)
    return x

#the resolution-preseving parts of U-NET
class MultiConv(tf.keras.Model):
    def __init__(self, model_size):
        super(MultiConv,self).__init__()
        
        self.filter_size = 3
        
        self.convs = [tf.keras.layers.Conv2D(model_size, self.filter_size, activation=tf.nn.gelu, padding='valid') for _ in range(2)]
        
    def call(self, data):
        output = data
        for layer in self.convs:
            orig = output
            output = pad_image_repeat(output, self.filter_size//2)
            output = orig+layer(output)
            #output = layer(output)
            output = normalize_features(output)
        return output
        

class NoisePredictorUNet(tf.keras.Model):
    def __init__(self):
        super(NoisePredictorUNet,self).__init__()

        #layer amount hardcoded for now
        self.features = [64,128,256,512,1024,512,256,128,64]
        self.concat   = [-1, -1, -1, -1,  -1,  3,  2,  1, 0]
        
        self.n_layers = len(self.features)
        
        downsamples = [tf.keras.layers.Conv2D(self.features[n+1], (2,2), strides=(2,2), activation=tf.nn.gelu) for n in range(self.n_layers//2)]
        upsamples = [tf.keras.layers.Conv2DTranspose(self.features[self.n_layers//2+1+n], (1,1), strides=(2,2), activation=tf.nn.gelu) for n in range(self.n_layers//2)]
        
        self.samples = []
        self.samples.extend(downsamples)
        self.samples.extend(upsamples)
        self.samples.extend([None])

        if prm['USE_TIMESTEP_EMBEDDINGS']:
            self.timestep_posenc = posenc.get_zero_to_one_posenc(prm['N_STEPS'], self.features[0])
        
        self.image_posenc = tf.keras.layers.Embedding(prm['SIZE_X']*prm['SIZE_Y'], self.features[0])
        self.prologue = tf.keras.layers.Conv2D(self.features[0], (1,1), use_bias=False)
        self.convs = [MultiConv(self.features[n]) for n in range(self.n_layers)]
        self.epilogue = tf.keras.layers.Conv2D(prm['CHANNELS'], (1,1), use_bias=False)

    def call(self, data, timestep_ns):
        #data shape:        BATCH, SIZEY, SIZEX, CHANNELS
        #timestep_ns shape: BATCH
        
        orig_input = data
        
        orig_shape = data.shape
        
        output = data
        output = self.prologue(output)
        
        if prm['USE_TIMESTEP_EMBEDDINGS']:
            timesteps = tf.gather(self.timestep_posenc, timestep_ns)
            #shape: BATCH, MODEL_SIZE
            timesteps = tf.expand_dims(timesteps,axis=1)
            timesteps = tf.expand_dims(timesteps,axis=1)
            #shape: BATCH, 1, 1, MODEL_SIZE
            output += timesteps

        image_posenc = self.image_posenc(tf.range(prm['SIZE_X']*prm['SIZE_Y'],dtype=tf.int64))
        #image_posenc shape: 784, 256
        image_posenc = tf.reshape(image_posenc, [1,prm['SIZE_Y'],prm['SIZE_X'],self.features[0]])
        #image_posenc shape: 1, 28, 28, 256
        #20.0 is an arbitrary constant to make it actually see the posenc.
        #i want to normalize these things somehow at some point.
        output += image_posenc*20.0
        
        #now we do the actual network
        outputs = []
        for n in range(self.n_layers):
            if self.concat[n] != -1:
                output += outputs[self.concat[n]] #can't concat because of residual network, so we just add.
            output = self.convs[n](output)
            outputs.append(output)
            
            if self.samples[n] is not None:
                output = self.samples[n](output)
            
        output = self.epilogue(output)
        output = normalize_features(output)
        return output

step=1

def lr_scheduler():
    global step
    minimum = prm['LEARNING_RATE_MIN']
    calculated = prm['LEARNING_RATE_START']*(2.0**(-step/5000.0))
    return max(minimum,calculated)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler,amsgrad=True,epsilon=prm['ADAM_EPSILON'])
noising_process = NoisingProcess(prm['N_STEPS'])
noise_predictor = NoisePredictorUNet()

@tf.function
def do_step(inputdata, timesteps, target):
    losses = []
    
    with tf.GradientTape() as tape:
        output = noise_predictor(inputdata, timesteps)
        loss = tf.reduce_mean(tf.square(target-output))
        losses.append(loss)
    gradients = tape.gradient(losses, noise_predictor.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    optimizer.apply_gradients(zip(gradients, noise_predictor.trainable_variables))

    return losses

#organize pictures in a neat grid
def gridify(allpics):
    n_pics = allpics.shape[0]
    height = tf.cast(tf.sqrt(tf.cast(n_pics,tf.float32)),dtype=tf.int64)
    width = n_pics // height + (1 if (n_pics % height > 0) else 0)
    
    total = height*width
    missing = total-n_pics
    if missing > 0:
        allpics = tf.concat([allpics,tf.zeros(shape=[missing,allpics.shape[-3],allpics.shape[-2],allpics.shape[-1]])],axis=0)
    allpics = tf.reshape(allpics, shape=[height, width, allpics.shape[-3],allpics.shape[-2],allpics.shape[-1]])
    allpics = tf.transpose(allpics, [0,2,1,3,4])
    allpics = tf.reshape(allpics, shape=[allpics.shape[0]*allpics.shape[1], allpics.shape[2]*allpics.shape[3],allpics.shape[4]])
    return allpics

training_sess_id = str(random.randrange(1000000000))

losses = []

data_shuffler = RangeShuffler(0,dataset.shape[0])
timestep_shuffler = RangeShuffler(1,prm['N_STEPS']+1)

starttime = time.time()

#training loop:
while True:
    ids = data_shuffler.get_batch(prm['BATCH_SIZE'])
    timestep_ids = timestep_shuffler.get_batch(prm['BATCH_SIZE'])
    
    originals = tf.gather(dataset, ids)
    
    targets = []
    batch = []
    
    for ex_id in range(prm['BATCH_SIZE']):
        ex, target = noising_process.direct(timestep_ids[ex_id], originals[ex_id])
        
        batch.append(ex)
        targets.append(target)
    
    batch = tf.stack(batch,axis=0)
    targets = tf.stack(targets,axis=0)
    
    losses.extend(do_step(batch, timestep_ids, targets))
    
    if step % prm['PRINT_TIME'] == 0:
    
        totaltime = time.time()-starttime
        starttime = time.time()
        
        losses = tf.reduce_mean(tf.stack(losses,axis=0))
        
        print(f"{totaltime} {step} {step*prm['BATCH_SIZE']} ",end="")
        
        print(f"{losses.numpy()} {lr_scheduler()}",end="")
        print()
        losses = []
        
        if step % prm['GENERATE_TIME'] == 0:
            #we generate pictures from noise here
            n_gen = prm['N_TEST_GENERATIONS']

            picture = tf.random.normal(shape=[n_gen,dataset.shape[1],dataset.shape[2],dataset.shape[3]], stddev=1.0, mean=0.0)
            for ts in range(prm['N_STEPS'], 0, -1):
                ts_tf = tf.expand_dims(tf.convert_to_tensor(ts),axis=0)
                ts_tf = tf.broadcast_to(ts_tf,[picture.shape[0]])
                
                oldnoise = noising_process.get_noise_stddev_for_step(ts)
                newnoise = noising_process.get_noise_stddev_for_step(ts-1)
                
                denoised_picture = picture-noise_predictor(picture, ts_tf)*oldnoise
                picture = denoised_picture + tf.random.normal(shape=picture.shape, mean=0, stddev=newnoise)
                
            picture = picture*dataset_stddev+dataset_mean
            picture = gridify(picture)
            picture = tf.cast(tf.clip_by_value(((picture+1.0)*0.5)*255.0,0.0,255.0),dtype=tf.uint8)
            
            if picture.shape[-1] == 1:
                picture = tf.squeeze(picture,axis=-1)
            
            im = Image.fromarray(picture.numpy())
            im.save(f"outs/sess{training_sess_id}_{step}.png")
    
    step += 1
