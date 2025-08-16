#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 22:15:02 2024

@author: harisushehu
"""

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import cv2
import os
import glob
import csv
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Set GPU configuration if necessary
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)


# Define functions for weights and layers
def get_weights(shape):
    data = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(data)


def get_biases(shape):
    data = tf.constant(0.1, shape=shape)
    return tf.Variable(data)


def create_layer(shape):
    W = get_weights(shape)
    b = get_biases([shape[-1]])
    return W, b


def convolution_2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pooling(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# Define placeholders
x = tf.placeholder(tf.float32, [None, 784])  # 28x28 images flattened to 784
y_loss = tf.placeholder(tf.float32, [None, 3])  # 3 classes, adjust as needed
keep_prob = tf.placeholder(tf.float32)  # Dropout rate placeholder

# Build the model
x_image = tf.reshape(x, [-1, 28, 28, 1])  # Reshape input to 28x28 grayscale

# First convolutional layer
W_conv1, b_conv1 = create_layer([5, 5, 1, 32])
h_conv1 = tf.nn.relu(convolution_2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pooling(h_conv1)

# Second convolutional layer
W_conv2, b_conv2 = create_layer([5, 5, 32, 64])
h_conv2 = tf.nn.relu(convolution_2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pooling(h_conv2)

# Fully connected layer
W_fc1, b_fc1 = create_layer([7 * 7 * 64, 1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout layer
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2, b_fc2 = create_layer([1024, 3])  # 3 output classes
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_loss, logits=y_conv))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

# Define accuracy
predicted = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_loss, 1))
accuracy = tf.reduce_mean(tf.cast(predicted, tf.float32))

# Reading train and test data
print("Reading train images...")
train_images = []
train_labels = []

# Read train images
rootdir_train = 'data/train/train'  # Update your training directory

for label in os.listdir(rootdir_train):
    label_path = os.path.join(rootdir_train, label)
    if os.path.isdir(label_path):
        for filename in glob.glob(os.path.join(label_path, '*.*')):
            im = cv2.imread(filename)
            if im is not None:
                im = cv2.resize(im, (28, 28))
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                train_images.append(im)
                train_labels.append(label)

print("Reading test images...")
test_images = []
test_labels = []

rootdir_test = 'data/train/test'  # Update your testing directory

for label in os.listdir(rootdir_test):
    label_path = os.path.join(rootdir_test, label)
    if os.path.isdir(label_path):
        for filename in glob.glob(os.path.join(label_path, '*.*')):
            im = cv2.imread(filename)
            if im is not None:
                im = cv2.resize(im, (28, 28))
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                test_images.append(im)
                test_labels.append(label)

# Preprocess data
train_images = np.array(train_images)
test_images = np.array(test_images)

le = LabelEncoder()
train_labels_encoded = le.fit_transform(train_labels)
test_labels_encoded = le.transform(test_labels)

train_images = np.reshape(train_images, (-1, 784))
test_images = np.reshape(test_images, (-1, 784))

train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

train_labels = to_categorical(train_labels_encoded, num_classes=3)
test_labels = to_categorical(test_labels_encoded, num_classes=3)

# Initialize variables
sess.run(tf.global_variables_initializer())


# Function to get next batch
def next_batch(num, data, labels):
    idx = np.random.choice(len(labels), num, replace=False)
    data_shuffle = data[idx]
    labels_shuffle = labels[idx]
    return data_shuffle, labels_shuffle


# Training parameters
epochs = 100
batch_size = 128
test_batch_size = 4
num_iterations = len(train_labels) // batch_size

# Training loop
for epoch in range(epochs):
    print(f"Epoch: {epoch + 1}")

    for i in range(num_iterations):
        batch_x, batch_y = next_batch(batch_size, train_images, train_labels)

        if i % 50 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                x: batch_x,
                y_loss: batch_y,
                keep_prob: 1.0
            })
            print(f"Iteration {i}, Training Accuracy = {train_accuracy:.4f}")

        sess.run(optimizer, feed_dict={
            x: batch_x,
            y_loss: batch_y,
            keep_prob: 0.5
        })

    # Evaluate on test data
    test_accuracy = sess.run(accuracy, feed_dict={
        x: test_images,
        y_loss: test_labels,
        keep_prob: 1.0
    })
    print(f'Test Accuracy after epoch {epoch + 1}: {test_accuracy:.4f}')

    # Save results to CSV
    csv_filename = 'results.csv'
    row_contents = [str(epoch + 1), str(test_accuracy)]
    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Accuracy'])
            writer.writerow(row_contents)
    else:
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_contents)
