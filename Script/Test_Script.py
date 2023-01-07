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
import joblib
import warnings
warnings.filterwarnings("ignore")

img1= input("Enter Image 1 Name: \n")
img2= input("Enter Image 2 Name: \n")

image1 = cv2.imread(img1)
image2 = cv2.imread(img2)

cluster = joblib.load("cluster.h5")
scalar = joblib.load("scaler.h5")
model = joblib.load("classifier.h5")

class1 = predict(cluster, scalar, model, image1)
print("First Signature is " + class1)

class2 = predict(cluster, scalar, model, image1)
print("Second Signature is " + class2)


siamese_network = get_siamese_network(mode='test')

siamese_model = SiameseModel(siamese_network)

encoder = extract_encoder(siamese_model)

for data in Testimages(img1,img2):
    img1,img2=data
    img1=img1.reshape(1,128,128,3)
    img2=img2.reshape(1,128,128,3)

verification =  classify_images(encoder, img1, img2, mode='test')
print("The second Signature is " + verification[0])