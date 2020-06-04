from __future__ import absolute_import, division, print_function

import os
import shutil
import re
import pdb

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

image_width  = 224 
image_height = 224 
batch_size   = 32

train_dir = 'D:/NEWTRAP/train/'
validation_dir = 'D:/NEWTRAP/val/'

# Rescale all images by 1./255 and apply image augmentation (limit overfitting)
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                            rotation_range=35,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=[0.8, 1.5],
                            horizontal_flip=True,
                            vertical_flip=False,
                            fill_mode='nearest',
                            brightness_range=[0.85, 1.15])
validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
                train_dir,  # Source directory for the training images
                target_size=(image_height, image_width),  
                batch_size=batch_size,
                class_mode='categorical')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = validation_datagen.flow_from_directory(
                validation_dir, # Source directory for the validation images
                target_size=(image_height, image_width),
                batch_size=batch_size,
                class_mode='categorical')
				
IMG_SHAPE = (image_height, image_width, 3)

# Create the base model from the pre-trained model Inceptionv3
base_model = keras.applications.inception_v3.InceptionV3(input_shape=IMG_SHAPE,
                                               include_top=False, 
                                               weights='imagenet')

base_model.trainable = False
# Let's take a look at the base model architecture
base_model.summary()

model = tf.keras.Sequential([
  base_model,
  keras.layers.GlobalAveragePooling2D(name='GAP'),
  keras.layers.Dense(2, name='out_dense'),
  #keras.layers.Dense(1, name='out_dense'),
  keras.layers.Activation('softmax', name='out_activation')
  #keras.layers.Activation('sigmoid', name='out_activation')
])

model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.summary()

modpath = 'D:/NEWTRAP/sex-inception.h5'

ckpt = tf.keras.callbacks.ModelCheckpoint(modpath, monitor='val_acc', verbose=0, 
                                save_best_only=True, save_weights_only=False, mode='max')
epochs = 50
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

class_weight = {0: 1.,
                1: 2.
                }

history = model.fit_generator(train_generator, 
                              steps_per_epoch = steps_per_epoch,
                              epochs=epochs, 
                              workers=4,
                              class_weight=class_weight,
                              validation_data=validation_generator, 
                              validation_steps=validation_steps,
                              callbacks=[ckpt])
							  
basemod = model.layers[0] # fetch layers from component base model
basemod.trainable = True

fine_tune_at = 100 

# Freeze all the layers before the `fine_tune_at` layer
for layer in basemod.layers[:fine_tune_at]:
    layer.trainable =  False

model.compile(loss='categorical_crossentropy',
              optimizer = tf.keras.optimizers.RMSprop(lr=1e-5),
              metrics=['accuracy'])
model.summary()

modpath = 'D:/NEWTRAP/sex-inception-finetune.h5'
ckpt = keras.callbacks.ModelCheckpoint(modpath, monitor='val_acc', verbose=0, 
                                save_best_only=True, save_weights_only=False, mode='max')#, save_freq='epoch')

history_fine = model.fit_generator(train_generator, 
                                   steps_per_epoch = steps_per_epoch,
                                   epochs=epochs, 
                                   workers=4,
                                   class_weight=class_weight,
                                   validation_data=validation_generator, 
                                   validation_steps=validation_steps,
                                   callbacks=[ckpt])
								   

def generate_cam(base_path,output_path,img_name,target_class):
  img = cv2.imread(base_path+img_name)
  img = cv2.resize(img, (image_width,image_height), interpolation = cv2.INTER_LANCZOS4)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = img / 255.
  
  class_weights = model.layers[-2].get_weights()[0]
  conv_model = keras.Model(inputs=model.input, outputs=model.get_layer('GAP').input)
  conv_outputs = conv_model.predict(np.array([img]))[0]
  
  cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
  target_class = target_class
  for i, w in enumerate(class_weights[:, target_class]):
    cam += w * conv_outputs[:, :, i]
        
  cam = np.maximum(cam, 0.)
  #cam = 1.- (cam / np.max(cam)) # 0 is red in COLORMAP_JET, revert
  cam = cam / np.max(cam) # 0 is red in COLORMAP_JET, revert
  cam = np.array(cam * 255, dtype = np.uint8)
  heatmap_cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)/255.
  heatmap_cam = cv2.resize(heatmap_cam, (224,224), interpolation = cv2.INTER_LANCZOS4)
  merged = cv2.addWeighted(heatmap_cam, 0.5, img, 0.5, 0)
  cv2.imwrite(output_path+img_name,merged*255)
