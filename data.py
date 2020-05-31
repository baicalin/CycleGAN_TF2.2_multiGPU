import tensorflow as tf
import os
import  matplotlib.pyplot as plt

class Image:

    @staticmethod
    def load_image(image_file, img_size=256):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [img_size, img_size])
        image = image * 2 - 1 # [-1, 1]
        return image


    @staticmethod
    def generate_images(images, epoch, global_batch):
        plt.figure(figsize=(30, 15))

        for img_no, imgs in enumerate(images):
            A, B, A2B, B2A, RA, RB = imgs
            A = tf.reshape(A, [256, 256, 3]).numpy()
            B = tf.reshape(B, [256, 256, 3]).numpy()
            B2A = tf.reshape(B2A, [256, 256, 3]).numpy()
            A2B = tf.reshape(A2B, [256, 256, 3]).numpy()
            RA = tf.reshape(RA, [256, 256, 3]).numpy()
            RB = tf.reshape(RB, [256, 256, 3]).numpy()

            display_list = [A, B, A2B, B2A, RA, RB]
            title = ['A', 'B', 'A2B', 'B2A', 'RA', 'RB']

            for i in range(len(title)):
                plt.subplot(2, len(title), img_no * len(title) + i + 1)
                plt.title(title[i])
                # getting the pixel values between [0, 1] to plot it.
                plt.imshow(display_list[i] * 0.5 + 0.5)
                plt.axis('off')

        plt.savefig('images/generated/E%d_B%d.png' % (epoch + 1, global_batch))
        plt.close()


class Pipeline:

    @staticmethod
    def create_pipeline(strategy, pathA, pathB, shuffle=False, epochs=1000, batch_size=1):
        #load file names
        dsA = tf.data.Dataset.list_files(pathA + '/*.jpg', shuffle=shuffle, seed=27)
        dsB = tf.data.Dataset.list_files(pathB + '/*.jpg', shuffle=shuffle, seed=27)

        #load files and cache
        dsA = dsA.map(lambda x: Image.load_image(x)).cache()
        dsB = dsB.map(lambda x: Image.load_image(x)).cache()

        # calculate epoch sample size (we need this because pathA and pathB contain various sample size and we want to use everything we have)
        lenA = len(os.listdir(pathA))
        lenB = len(os.listdir(pathB))
        if lenA > lenB:
            l = lenB - lenB % batch_size
        else:
            l = lenA - lenA % batch_size

        #shuffle
        dsA = dsA.shuffle(buffer_size=lenA, reshuffle_each_iteration=True, seed=27)
        dsB = dsB.shuffle(buffer_size=lenB, reshuffle_each_iteration=True, seed=27)

        #take 'l' elements, batch and prefetch
        dsA = dsA.take(l).batch(batch_size).prefetch(batch_size)
        dsB = dsB.take(l).batch(batch_size).prefetch(batch_size)

        #combine and make distributable
        ds = tf.data.Dataset.zip((dsA, dsB))
        ds = strategy.experimental_distribute_dataset(ds) #placing batch_size / 4 samples per GPU
        return ds

    @staticmethod
    def create_test_pipeline(pathA, pathB):
        # load file names
        dsA = tf.data.Dataset.list_files(pathA + '/*.jpg', shuffle=False).take(2)
        dsB = tf.data.Dataset.list_files(pathB + '/*.jpg', shuffle=False).take(2)

        # load files and cache
        dsA = dsA.map(lambda x: Image.load_image(x)).cache()
        dsB = dsB.map(lambda x: Image.load_image(x)).cache()

        # take 'l' elements, batch and prefetch
        dsA = dsA.batch(2).prefetch(2)
        dsB = dsB.batch(2).prefetch(2)

        # combine and make distributable
        ds = tf.data.Dataset.zip((dsA, dsB))
        return ds