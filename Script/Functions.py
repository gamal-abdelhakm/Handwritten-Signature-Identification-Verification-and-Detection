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
# a function that reads images and put them in a dictionary for each person
def read_images(split_type = 'Train'):
    images_dict={}
    image_count = 0
    for person in ['personA','personB','personC','personD','personE']:
        csv_path = f"{person + '/' + split_type+'/'}{person}_SigVerification{split_type}Labels.csv"
        df = pd.read_csv(csv_path)
        images_dict[person]={'forged':[],'real':[]}
        for index, row in df.iterrows():
            folder= person + '/' + split_type
            image = row['image_name']
            if os.path.exists(f'{folder}'+f'/{image}'):
                if row['label'] == 'forged':
                    images_dict[person]['forged'].append([folder,image])
                else:
                    images_dict[person]['real'].append([folder,image])
                image_count +=1
    return images_dict , image_count


# a function that creates triplets to use for training
def create_triplets(images_dict):
    triplets=[]
    for person in images_dict:
        for i in range(len(images_dict[person]['real'])):
            for j in range(i+1,len(images_dict[person]['real'])):
                anchor = (images_dict[person]['real'][i][0] , images_dict[person]['real'][i][1])
                positive = (images_dict[person]['real'][j][0] , images_dict[person]['real'][j][1])
                k = random.randint(0, len(images_dict[person]['forged'])-1)
                negative = (images_dict[person]['forged'][k][0],images_dict[person]['forged'][k][1])
                triplets.append((anchor,positive,negative))
    random.shuffle(triplets)
    return triplets

# a Function that samples the data accordingly
def get_batch(triplet_list, batch_size=256, preprocess=True):
    batch_steps = len(triplet_list) // batch_size

    for i in range(batch_steps + 1):
        anchor = []
        positive = []
        negative = []

        j = i * batch_size
        while j < (i + 1) * batch_size and j < len(triplet_list):
            a, p, n = triplet_list[j]
            a = cv2.imread(f"{a[0]}/{a[1]}")
            p = cv2.imread(f"{p[0]}/{p[1]}")
            n = cv2.imread(f"{n[0]}/{n[1]}")
            a = cv2.resize(a, (128, 128))
            p = cv2.resize(p, (128, 128))
            n = cv2.resize(n, (128, 128))
            anchor.append(a)
            positive.append(p)
            negative.append(n)
            j += 1
        anchor = np.array(anchor)
        positive = np.array(positive)
        negative = np.array(negative)

        if preprocess:
            anchor = preprocess_input(anchor)
            positive = preprocess_input(positive)
            negative = preprocess_input(negative)

        yield ([anchor, positive, negative])


# a Function that returns a pretrained Xception encoder
def get_encoder(input_shape):
    """ Returns the image encoding model """

    pretrained_model = Xception(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False,
        pooling='avg',
    )

    for i in range(len(pretrained_model.layers) - 27):
        pretrained_model.layers[i].trainable = False

    encode_model = Sequential([
        pretrained_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ], name="Encode_Model")
    return encode_model


# a Function that encodes the inputs and computes the distances using distancelayer()
def get_siamese_network(mode = 'train',input_shape=(128, 128, 3)):
    encoder = get_encoder(input_shape)

    if mode != 'train':
        encoder.load_weights("encoder")

    # Input Layers for the images
    anchor_input = layers.Input(input_shape, name="Anchor_Input")
    positive_input = layers.Input(input_shape, name="Positive_Input")
    negative_input = layers.Input(input_shape, name="Negative_Input")

    ## Generate the encodings (feature vectors) for the images
    encoded_a = encoder(anchor_input)
    encoded_p = encoder(positive_input)
    encoded_n = encoder(negative_input)

    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    distances = DistanceLayer()(
        encoder(anchor_input),
        encoder(positive_input),
        encoder(negative_input)
    )

    # Creating the Model
    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=distances,
        name="Siamese_Network"
    )
    return siamese_network

# a Function to test the model
def test_on_triplets(test_triplets,siamese_model,batch_size=256):
    pos_scores, neg_scores = [], []

    for data in get_batch(test_triplets, batch_size=batch_size):
        prediction = siamese_model.predict(data)
        pos_scores += list(prediction[0])
        neg_scores += list(prediction[1])

    accuracy = np.sum(np.array(pos_scores) < np.array(neg_scores)) / len(pos_scores)
    ap_mean = np.mean(pos_scores)
    an_mean = np.mean(neg_scores)
    ap_stds = np.std(pos_scores)
    an_stds = np.std(neg_scores)

    print(f"Accuracy on test = {accuracy:.5f}")
    return (accuracy, ap_mean, an_mean, ap_stds, an_stds)


# a Function that saves the encoder weights
def extract_encoder(model):
    encoder = get_encoder((128, 128, 3))
    i=0
    for e_layer in model.layers[0].layers[3].layers:
        layer_weight = e_layer.get_weights()
        encoder.layers[i].set_weights(layer_weight)
        i+=1
    return encoder

# a Function that takes two lists of images and classifies them
def classify_images(encoder,sig_list1, sig_list2, threshold=1.3, mode = 'train'):
    # Getting the encodings for the passed faces
    tensor1 = encoder.predict(sig_list1)
    tensor2 = encoder.predict(sig_list2)

    distance = np.sum(np.square(tensor1 - tensor2), axis=-1)
    prediction = np.where(distance <= threshold, 0, 1)

    if mode != 'trian':
        prediction = np.where(distance <= threshold, 'Real', "Forged")

    return prediction


# a Function that computes the confusion matrix
def ModelMetrics(pos_list, neg_list):
    true = np.array([0] * len(pos_list) + [1] * len(neg_list))
    pred = np.append(pos_list, neg_list)

    # Compute and print the accuracy
    print(f"\nAccuracy of model: {accuracy_score(true, pred)}\n")

    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(true, pred)

    categories = ['Similar', 'Different']
    names = ['True Similar', 'False Similar', 'False Different', 'True Different']
    percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(names, percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='',
                xticklabels=categories, yticklabels=categories)

    plt.xlabel("Predicted", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)


def Testimages(img1,img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    img1 = cv2.resize(img1, (128, 128))
    img2 = cv2.resize(img2, (128, 128))
    img1 = np.array(img1)
    img2 = np.array(img2)
    img1 = preprocess_input(img1)
    img2 = preprocess_input(img2)
    yield ([img1, img2])

def predict(cluster, scalar, model, img):

    labels = {0:"Person A", 1:"Person B", 2:"Person C", 3:"Person D", 4:"Person E"}
    number_of_clusters = 10
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)  # 1- extract features
    cluster_result = cluster.predict(descriptors)  # 2- predict cluster

    # 3- build vocabulary
    vocabulary = np.array([[0 for i in range(number_of_clusters)]], 'float32')
    for each in cluster_result:
        vocabulary[0][each] += 1

    # vocabulary = reweight_tf_idf(vocabulary) ### tf_idf
    vocabulary = scalar.transform(vocabulary)  # 4- normalization
    prediction = model.predict(vocabulary)  # 5 - classification
    return labels[prediction[0]]