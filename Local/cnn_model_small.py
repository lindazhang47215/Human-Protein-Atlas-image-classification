# import keras
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten, Dropout 

def cnn_model_small(inputs, training_flag):

    training_flag = training_flag
    # Input Layer
    x = tf.reshape(inputs, [-1, 128, 128, 4])

    l2_scale = 0.00001
    # Convolutional Layer #1/2
    with tf.name_scope("conv1"):
        x = tf.layers.conv2d(
                inputs=x,
                filters=8,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu,
                kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=l2_scale))
        x = tf.layers.batch_normalization(x, training=training_flag)

        x = tf.layers.conv2d(
                inputs=x,
                filters=16,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu,
                kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=l2_scale))
        x = tf.layers.batch_normalization(x, training=training_flag)

        # pooling and dropout (64 x64 x 16)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)
        dropout = tf.layers.dropout(inputs=x, rate=0.4, training=training_flag)

    # conv 2 (64 * 64 * 16) -> (32 * 32 * 32)
    with tf.name_scope("conv2"):
        x = tf.layers.conv2d(
                inputs=x,
                filters=32,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu,
                kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=l2_scale))
        x = tf.layers.batch_normalization(x, training=training_flag)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)
        dropout = tf.layers.dropout(inputs=x, rate=0.4, training=training_flag)

    # conv 3 (32 * 32 * 32) -> (16 * 16 * 64)
    with tf.name_scope("conv3"):
        x = tf.layers.conv2d(
                inputs=x,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu,
                kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=l2_scale))
        x = tf.layers.batch_normalization(x, training=training_flag)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)
        dropout = tf.layers.dropout(inputs=x, rate=0.4, training=training_flag)

    # conv 4 (16 * 16 * 64) -> (8*8*128)
    with tf.name_scope("conv4"):
        x = tf.layers.conv2d(
                inputs=x,
                filters=128,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu,
                kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=l2_scale))
        x = tf.layers.batch_normalization(x, training=training_flag)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)
        x = tf.layers.dropout(inputs=x, rate=0.4, training=training_flag)

    # Dense Layer
    with tf.name_scope("dense1"):
        x = tf.reshape(x, [-1, 8*8*128])
        x = tf.layers.dropout(inputs=x_flat, rate=0.4, training=training_flag)

    with tf.name_scope("dense2"):
        x = tf.layers.dense(inputs=x, units=256, activation=tf.nn.relu,
                            kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=l2_scale))
        x = tf.layers.dropout(inputs=x, rate=0.4, training=training_flag)

      # Logits Layer
    with tf.name_scope("logit"):
        logits = tf.layers.dense(inputs=x, units=28, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=l2_scale))

    return logits



if __name__ == "__main__":
    pass
    