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
import glob
import datasets

from PIL import Image

if not os.path.exists("outs"):
    os.mkdir("outs")

def jsonload(filename):
    return json.load(open(filename,"r"))
    
def jsonsave(obj, filename):
    json.dump(obj, open(filename, "w"))

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
        
    def get_state(self):
        return self.current.numpy()
        
    def set_state(self, s):
        self.current = tf.convert_to_tensor(s)

#dataset = datasets.MNIST()

#dataset = datasets.Directory("Q:\\doom\\out", "png", pad=[[0,0],[0,8],[0,0],[0,0]])

dataset = datasets.UltimateDoom("c:\\datasets\\ultimate_doom.npy", pad=[[0,0],[0,8],[0,0],[0,0]])

print(f"Dataset shape: {dataset.shape}")

prm = {} #parameters
prm['N_GENERATE_STEPS'] = 399 #how many diffusion steps we use
prm['SIZE_Y'] = dataset.shape[-3]
prm['SIZE_X'] = dataset.shape[-2]
prm['CHANNELS'] = dataset.shape[-1]
prm['IMAGE_SIZE'] = prm['SIZE_Y']*prm['SIZE_X']*prm['CHANNELS']
prm['BATCH_SIZE'] = 32
prm['LEARNING_RATE_START'] = 0.001
prm['LEARNING_RATE_MIN'] = 0.0001
prm['ADAM_EPSILON'] = 1e-4
prm['N_TEST_GENERATIONS'] = 9
prm['PRINT_TIME'] = 128 #how many steps between prints
prm['GENERATE_TIME'] = prm['PRINT_TIME']*8 # how many steps between generated outputs saved to .png
prm['N_BUCKETS'] = 1024#prm['BATCH_SIZE']
prm['USE_BUCKETS'] = True

prm['MODEL_SAVE_DIR'] = "Q:\\model_saves"
prm['MODEL_SAVE_TIME'] = prm['PRINT_TIME']*32 #how many steps between model saves.

for key in prm:
    print(f"{key}={prm[key]} ",end="")
print()

class NoisingProcess:
    def curvify(self, x):
        #sqrt() so that when we add the two kinds of noise together,
        #we get an image that has stddev=1
        return tf.math.sqrt(x)

    def __init__(self, timesteps):
        self.timesteps = timesteps
        
        self.image_coef = self.curvify(tf.linspace(start=1.0, stop=0.0, num=timesteps+1))
        self.noise_coef = self.curvify(tf.linspace(start=0.0, stop=1.0, num=timesteps+1))
        
    def get_continuous_learning_batch(batch_size):
        return self.curvify(tf.random.uniform([batch_size], minval=0.0, maxval=1.0))

    def direct_continuous(self, step, image):
        assert step>=0.0 and step<=1.0
        noise_c = self.curvify(step)
        image_c = self.curvify(1.0-step)
        noise = tf.random.normal(shape=image.shape, stddev=1.0)
        ret = image*image_c + noise*noise_c
        return ret, noise

        
    def get_noise_stddev_for_step(self, step):
        return self.noise_coef[step]


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
            output = normalize_features(output)
        return output
        

class NoisePredictorUNet(tf.keras.Model):
    def __init__(self):
        super(NoisePredictorUNet,self).__init__()
        
        #layer amount hardcoded for now
        self.features = [64,128,256,512,512,512,256,128,64]
        self.concat   = [-1, -1, -1, -1, -1,  3,  2,  1, 0]
        
        #self.features = [64,128,256,512,256,128,64]
        #self.concat   = [-1, -1, -1, -1,  2,  1, 0]

        #self.features = [64,128,64]
        #self.concat   = [-1, -1, 0]
        
        self.n_layers = len(self.features)
        
        downsamples = [tf.keras.layers.Conv2D(self.features[n+1], (2,2), strides=(2,2), activation=tf.nn.gelu) for n in range(self.n_layers//2)]
        upsamples = [tf.keras.layers.Conv2DTranspose(self.features[self.n_layers//2+1+n], (1,1), strides=(2,2), activation=tf.nn.gelu) for n in range(self.n_layers//2)]
        
        self.samples = []
        self.samples.extend(downsamples)
        self.samples.extend(upsamples)
        self.samples.extend([None])

        self.image_posenc = tf.keras.layers.Embedding(prm['SIZE_X']*prm['SIZE_Y'], self.features[0])
        self.prologue = tf.keras.layers.Conv2D(self.features[0], (1,1), use_bias=False)
        self.convs = [MultiConv(self.features[n]) for n in range(self.n_layers)]
        self.epilogue = tf.keras.layers.Conv2D(prm['CHANNELS'], (1,1), use_bias=False)

    def call(self, data):
        #data shape:        BATCH, SIZEY, SIZEX, CHANNELS
        
        orig_input = data
        
        orig_shape = data.shape
        
        output = data
        output = self.prologue(output)
        
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
noising_process = NoisingProcess(prm['N_GENERATE_STEPS'])
noise_predictor = NoisePredictorUNet()

@tf.function
def do_step(inputdata, target):
    losses = []
    
    with tf.GradientTape() as tape:
        output = noise_predictor(inputdata)
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
bucket_shuffler = RangeShuffler(0, prm['N_BUCKETS'])

starttime = time.time()

if os.path.isfile(prm['MODEL_SAVE_DIR']+'/settings.txt'):
    settings = jsonload(prm['MODEL_SAVE_DIR']+'/settings.txt')
    
    checkpoint = tf.train.Checkpoint(opt=optimizer, noisepred=noise_predictor)
    status = checkpoint.restore(tf.train.latest_checkpoint(prm['MODEL_SAVE_DIR']+'/weights'))
    
    step = settings['step']
    data_shuffler.set_state(np.array(settings['data_shuffler_state'],dtype=np.int64))
    bucket_shuffler.set_state(np.array(settings['bucket_shuffler_state'],dtype=np.int64))
    training_sess_id = settings['training_sess_id']

#training loop:
while True:
    ids = data_shuffler.get_batch(prm['BATCH_SIZE'])
    
    if prm['USE_BUCKETS']:        
        timestep_vals = bucket_shuffler.get_batch(prm['BATCH_SIZE'])
        timestep_vals = tf.cast(timestep_vals,dtype=tf.float32)*(1.0/float(prm['N_BUCKETS']))
        timestep_vals += tf.random.uniform(shape=timestep_vals.shape, minval=0.0, maxval=1.0/float(prm['N_BUCKETS']))
    else:
        timestep_vals = tf.random.uniform(shape=[prm['BATCH_SIZE']], minval=0.0,maxval=1.0)
    
    originals = dataset.gather(ids)
    
    targets = []
    batch = []
    for ex_id in range(prm['BATCH_SIZE']):
        ex, target = noising_process.direct_continuous(timestep_vals[ex_id], originals[ex_id])
        batch.append(ex)
        targets.append(target)
    
    batch = tf.stack(batch,axis=0)
    targets = tf.stack(targets,axis=0)
    
    losses.extend(do_step(batch, targets))
    
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
            ts_tf = tf.convert_to_tensor(prm['N_GENERATE_STEPS'])
            
            @tf.function
            def ret_pic(picture, oldnoise, newnoise):
                denoised_picture = picture-noise_predictor(picture)*oldnoise
                picture = denoised_picture + tf.random.normal(shape=picture.shape, mean=0.0, stddev=newnoise)
                return picture
            
            for ts in range(prm['N_GENERATE_STEPS'], 0, -1):
                oldnoise = noising_process.get_noise_stddev_for_step(ts)
                newnoise = noising_process.get_noise_stddev_for_step(ts-1)
                picture = ret_pic(picture, oldnoise, newnoise)
                
            picture = picture*dataset.stddev+dataset.mean
            picture = gridify(picture)
            picture = tf.cast(tf.clip_by_value(((picture+1.0)*0.5)*255.0,0.0,255.0),dtype=tf.uint8)
            
            if picture.shape[-1] == 1:
                picture = tf.squeeze(picture,axis=-1)
            
            im = Image.fromarray(picture.numpy())
            im.save(f"outs/sess{training_sess_id}_{step}.png")
    

    if step % prm['MODEL_SAVE_TIME'] == 0:
        checkpoint = tf.train.Checkpoint(opt=optimizer, noisepred=noise_predictor)
        ckptfolder = checkpoint.save(file_prefix=prm['MODEL_SAVE_DIR']+'/weights/ckpt')
        
        sets = {
            'step':step+1, 
            'folder':ckptfolder, 
            'data_shuffler_state':data_shuffler.get_state().tolist(), 
            'bucket_shuffler_state':bucket_shuffler.get_state().tolist(),
            'training_sess_id': training_sess_id
        }
        
        jsonsave(sets, prm['MODEL_SAVE_DIR']+'/settings.txt')
        
        print("Saved.", end="    \r")
        
    step += 1
