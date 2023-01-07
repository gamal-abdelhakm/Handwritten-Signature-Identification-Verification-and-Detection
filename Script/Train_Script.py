import os
import cv2
import time
import random
import numpy as np
import pandas as pd
import keras.api._v2.keras as keras
import tensorflow as tf
tf.__version__, np.__version__
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras import backend, layers, metrics

from tensorflow.keras.optimizers.experimental import SGD
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.utils import plot_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import seaborn as sns
import matplotlib.pyplot as plt
from Classes import *
from Functions import *


# loading the training and testing images
train_images_dict,train_image_count = read_images('Train')
test_images_dict,test_image_count = read_images('Test')
train_triplets = create_triplets(train_images_dict)
test_triplets = create_triplets(test_images_dict)

# Create a plot from a sample of training images
num_plots = 6
f, axes = plt.subplots(num_plots, 3, figsize=(15, 20))
for x in get_batch(train_triplets, batch_size=num_plots, preprocess=False):
    a,p,n = x
    for i in range(num_plots):
        axes[i, 0].imshow(a[i])
        axes[i, 1].imshow(p[i])
        axes[i, 2].imshow(n[i])
        i+=1
    break

# loading the siamese network
siamese_network = get_siamese_network()
siamese_network.summary()

# loading a siamese network
siamese_model = SiameseModel(siamese_network)

# setting optimizer to SGD with nesterov momentum (best performance)
optimizer = SGD(learning_rate=1e-3, momentum = 0.9, nesterov = True)
siamese_model.compile(optimizer=optimizer)

# --------------------------- Training the model ----------------------------------

with tf.device('/device:GPU:0'): # Use GPU
    save_all = False
    epochs = 10
    batch_size = 4

    max_acc = 0
    train_loss = []
    test_metrics = []

    for epoch in range(1, epochs + 1):
        t = time.time()

        # Training the model on train data
        epoch_loss = []
        for data in get_batch(train_triplets, batch_size=batch_size):
            loss = siamese_model.train_on_batch(data)
            epoch_loss.append(loss)
        epoch_loss = sum(epoch_loss) / len(epoch_loss)
        train_loss.append(epoch_loss)

        print(f"\nEPOCH: {epoch} \t (Epoch done in {int(time.time() - t)} sec)")
        print(f"Loss on train    = {epoch_loss:.5f}")

        # Testing the model on test data
        metric = test_on_triplets(test_triplets,siamese_model, batch_size=batch_size)
        test_metrics.append(metric)
        accuracy = metric[0]

        # Saving the model weights
        if save_all or accuracy >= max_acc:
            siamese_model.save_weights("siamese_model")
            max_acc = accuracy

    # Saving the model after all epochs run
    siamese_model.save_weights("siamese_model-final")

# ------------------------------------------------------------------------------------------

# extract encoder weights
encoder = extract_encoder(siamese_model)
encoder.save_weights("encoder")
encoder.summary()

# Test the model on test data
pos_list = np.array([])
neg_list = np.array([])
for data in get_batch(test_triplets, batch_size=256):
    a, p, n = data
    pos_list = np.append(pos_list, classify_images(encoder, a, p))
    neg_list = np.append(neg_list, classify_images(encoder, a, n))
    break

# compute confusion matrix
ModelMetrics(pos_list, neg_list)