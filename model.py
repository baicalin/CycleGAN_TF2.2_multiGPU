import tensorflow as tf
from tensorflow import keras
import os
import numpy

class Encoder(keras.Model):

    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = keras.layers.Conv2D(filters=32, kernel_size=7, strides=1,
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv3 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()

    def call(self, inputs, training=True):
        x = tf.pad(inputs, [[0,0], [3,3], [3,3], [0,0]], 'REFLECT')

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)

        return x


class Residual(keras.Model):

    def __init__(self):
        super(Residual, self).__init__()

        self.conv1 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=1,
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.conv2 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=1,
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()

    def call(self, inputs, training=True):
        x = tf.pad(inputs, [[0,0], [1,1], [1,1], [0,0]], 'REFLECT')

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = tf.pad(x, [[0,0], [1,1], [1,1], [0,0]], 'REFLECT')

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = tf.add(x, inputs)

        return x


class Decoder(keras.Model):

    def __init__(self):
        super(Decoder, self).__init__()

        self.conv1 = keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv3 = keras.layers.Conv2D(filters=3, kernel_size=7, strides=1,
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        x = tf.pad(x, [[0,0], [3,3], [3,3], [0,0]], 'REFLECT')

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.tanh(x)

        return x

class Generator(keras.Model):

    def __init__(self, img_size=256, skip=False):

        super(Generator, self).__init__()

        self.img_size = img_size
        self.skip = skip

        self.encoder = Encoder()

        self.res1 = Residual()
        self.res2 = Residual()
        self.res3 = Residual()
        self.res4 = Residual()
        self.res5 = Residual()
        self.res6 = Residual()

        if (img_size > 128):
            self.res7 = Residual()
            self.res8 = Residual()
            self.res9 = Residual()

        self.decoder = Decoder()

    def call(self, inputs, training=True):

        x = self.encoder(inputs, training)
        x = self.res1(x, training)
        x = self.res2(x, training)
        x = self.res3(x, training)
        x = self.res4(x, training)
        x = self.res5(x, training)
        x = self.res6(x, training)

        if (self.img_size > 128):
            x = self.res7(x, training)
            x = self.res8(x, training)
            x = self.res9(x, training)

        x = self.decoder(x, training)

        return x

class Discriminator(keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv3 = keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv4 = keras.layers.Conv2D(filters=512, kernel_size=4, strides=1, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv5 = keras.layers.Conv2D(filters=1, kernel_size=4, strides=1, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.leaky = keras.layers.LeakyReLU(0.2)

        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.leaky(x)

        x = self.conv2(x)
        x = self.bn1(x, training=training)
        x = self.leaky(x)

        x = self.conv3(x)
        x = self.bn2(x, training=training)
        x = self.leaky(x)

        x = self.conv4(x)
        x = self.bn3(x, training=training)
        x = self.leaky(x)

        x = self.conv5(x)

        return x