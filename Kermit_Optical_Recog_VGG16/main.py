# set the matplotlib backend so figures can be saved in the background
import matplotlib
from sklearn.metrics import confusion_matrix, f1_score

matplotlib.use("Agg")

# Matplot Imports

import matplotlib.pyplot as plt
from model_evaluation_utils import get_metrics

import sklearn.metrics
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths

import numpy as np
import argparse
import random
import cv2
import os

import tensorflow as tf
from keras import callbacks
from keras import optimizers
from keras.datasets import cifar10
from keras.engine import Model
from keras.applications import vgg16 as vgg
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

# %% Load Data
BATCH_SIZE = 32
EPOCHS = 40
NUM_CLASSES = 2
LEARNING_RATE = 1e-4
MOMENTUM = 0.9

print("[INFO] loading images...")
data = []
labels = []
picPaths = sorted(list(paths.list_images("data/TRAIN/")))
random.seed(123)
random.shuffle(picPaths)

for picPath in picPaths:
    img = cv2.imread(picPath)
    img = cv2.resize(img, (128, 128))
    data.append(img)

    label = picPath.split("/")[2].split("\\")[0]
    label = 1 if label == "KERMIT" else 0
    labels.append(label)

# %% scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# %% Train Test Split
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# %%
base_model = vgg.VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(128, 128, 3))

last = base_model.get_layer('block3_pool').output
x = GlobalAveragePooling2D()(last)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.6)(x)
pred = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(base_model.input, pred)

for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=LEARNING_RATE),
              metrics=['accuracy'])
# model.summary()

train_datagen = ImageDataGenerator(
    # rescale=1. / 255,
    horizontal_flip=False)
train_datagen.fit(trainX)
train_generator = train_datagen.flow(trainX, trainY, batch_size=BATCH_SIZE)
val_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=False)

val_datagen.fit(testX)
val_generator = val_datagen.flow(testX, testY, batch_size=BATCH_SIZE)

# %% Train the model
train_steps_per_epoch = trainX.shape[0] // BATCH_SIZE
val_steps_per_epoch = trainY.shape[0] // BATCH_SIZE

history = model.fit_generator(train_generator,
                              steps_per_epoch=train_steps_per_epoch,
                              validation_data=val_generator,
                              validation_steps=val_steps_per_epoch,
                              epochs=EPOCHS,
                              verbose=1)

model.save("models/")

# %% Evaluate Performance
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
t = f.suptitle('Deep Neural Net Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epochs = list(range(1, EPOCHS + 1))
#%%
ax1.plot(epochs, history.history['acc'], label='Train Accuracy')
ax1.plot(epochs, history.history['val_acc'], label='Validation Accuracy')
ax1.set_xticks(epochs)
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epochs, history.history['loss'], label='Train Loss')
ax2.plot(epochs, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(epochs)
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")



#%% Predictions

predictions = model.predict(testX, batch_size=32)
pred_trans = (predictions > 0.5)
pred_trans = np.argmax(pred_trans, axis=1)
test_trans = (testY > 0.5)
test_trans = np.argmax(test_trans, axis=1)
conf_matrix = confusion_matrix(test_trans, pred_trans)
f1Score =f1_score(test_trans, pred_trans)

