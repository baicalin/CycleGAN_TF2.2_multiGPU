# based on: https://www.tensorflow.org/tutorials/generative/cyclegan
# datasets: https://people.eecs.berkeley.edu/%7Etaesung_park/CycleGAN/datasets/

import tensorflow as tf


from tensorflow_examples.models.pix2pix import pix2pix #pip install -q git+https://github.com/tensorflow/examples.git
import os, os.path
import time
import matplotlib.pyplot as plt

from tensorflow.python.profiler import profiler_v2 as profiler

from colorama import Fore, Style

TEST_SIZE           = 8

AUTOTUNE            = tf.data.experimental.AUTOTUNE

def cprint(s, end='\n', flush=True):
    print('{}{}{}'.format(Fore.GREEN, s, Style.RESET_ALL), end=end, flush=flush)

restored_epoch      = -1
strategy            = tf.distribute.MirroredStrategy()
cprint('Using {} replicas in sync'.format(strategy.num_replicas_in_sync))

BATCH_SIZE          = 1
GLOBAL_BATCH_SIZE   = BATCH_SIZE * strategy.num_replicas_in_sync
IMG_WIDTH           = 256
IMG_HEIGHT          = 256
OUTPUT_CHANNELS     = 3
LAMBDA              = 10


TRAIN_HORSES_PATH   = 'images/trainA'
TRAIN_ZEBRAS_PATH   = 'images/trainB'
TEST_HORSES_PATH    = 'images/testA'
TEST_ZEBRAS_PATH    = 'images/testB'


def calculate_buffer_size():
    a_buffer_size = len([name for name in os.listdir(TRAIN_HORSES_PATH) if os.path.isfile(os.path.join(TRAIN_HORSES_PATH, name))])
    b_buffer_size = len([name for name in os.listdir(TRAIN_ZEBRAS_PATH) if os.path.isfile(os.path.join(TRAIN_ZEBRAS_PATH, name))])

    buffer_size = a_buffer_size
    if b_buffer_size < a_buffer_size: buffer_size = b_buffer_size

    buffer_size = buffer_size - (buffer_size % BATCH_SIZE) 

    return buffer_size, a_buffer_size, b_buffer_size



BUFFER_SIZE, TRAIN_HORSES_BUFFER, TRAIN_ZEBRAS_BUFFER = calculate_buffer_size()



def random_crop(image):
    return tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])


def normalize(image):
    return (tf.cast(image, tf.float32) * 2) - 1


def random_jitter(image):

    image = tf.image.resize(image, size=[286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = random_crop(image)
    return tf.image.random_flip_left_right(image, seed=27)


def preprocess_image_train(image):
    image = random_jitter(image)
    image = normalize(image)
    return image


def preprocess_image_test(image):
    return normalize(image)


def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32) # Images that are represented using floating point values are expected to have values in the range [0,1).
    return image


with strategy.scope():

    train_horses    = tf.data.Dataset.list_files(TRAIN_HORSES_PATH + '/*.jpg').interleave(
        lambda x: tf.data.Dataset.from_tensors(x).map(lambda y: load_image(y), num_parallel_calls=AUTOTUNE),
        num_parallel_calls=AUTOTUNE)

    train_zebras    = tf.data.Dataset.list_files(TRAIN_ZEBRAS_PATH + '/*.jpg').interleave(
        lambda x: tf.data.Dataset.from_tensors(x).map(lambda y: load_image(y), num_parallel_calls=AUTOTUNE),
        num_parallel_calls=AUTOTUNE)

    test_horses     = tf.data.Dataset.list_files(TEST_HORSES_PATH + '/*.jpg').take(TEST_SIZE).map(lambda x: load_image(x), num_parallel_calls=AUTOTUNE).map(preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().batch(TEST_SIZE)
    test_zebras     = tf.data.Dataset.list_files(TEST_ZEBRAS_PATH + '/*.jpg').take(TEST_SIZE).map(lambda x: load_image(x), num_parallel_calls=AUTOTUNE).map(preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().batch(TEST_SIZE)

    train_horses    = train_horses.map(preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(buffer_size=TRAIN_HORSES_BUFFER, reshuffle_each_iteration=True, seed=27).take(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    train_zebras    = train_zebras.map(preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(buffer_size=TRAIN_ZEBRAS_BUFFER, reshuffle_each_iteration=True, seed=27).take(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    train_horses    = strategy.experimental_distribute_dataset(train_horses)
    train_zebras    = strategy.experimental_distribute_dataset(train_zebras)

    for sample_horse in test_horses: break
    for sample_zebra in test_zebras: break


    generator_g         = pix2pix.unet_generator(output_channels=OUTPUT_CHANNELS, norm_type='instancenorm')
    generator_f         = pix2pix.unet_generator(output_channels=OUTPUT_CHANNELS, norm_type='instancenorm')

    discriminator_x     = pix2pix.discriminator(norm_type='instancenorm', target=False)
    discriminator_y     = pix2pix.discriminator(norm_type='instancenorm', target=False)

    generator_g_opt     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_opt     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_x_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_y_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

def discriminator_loss(real, generated):
    real_loss       = loss_obj(tf.ones_like(real), real)
    generated_loss  = loss_obj(tf.zeros_like(generated), generated)
    return tf.nn.compute_average_loss((real_loss + generated_loss) * 0.5, global_batch_size=GLOBAL_BATCH_SIZE)

def generator_loss(generated):
    return tf.nn.compute_average_loss(loss_obj(tf.ones_like(generated), generated), global_batch_size=GLOBAL_BATCH_SIZE)

def calc_cycle_loss(real_image, cycled_image):
    loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return LAMBDA * loss

def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * loss


checkpoint_path = './checkpoints'

ckpt = tf.train.Checkpoint(
    generator_g         = generator_g,
    generator_f         = generator_f,
    discriminator_x     = discriminator_x,
    discriminator_y     = discriminator_y,
    generator_g_opt     = generator_g_opt,
    generator_f_opt     = generator_f_opt,
    discriminator_x_opt = discriminator_x_opt,
    discriminator_y_opt = discriminator_y_opt
)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

with strategy.scope():
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        restored_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1]) * 5
        cprint('Model restored at epoch {}'.format(restored_epoch))


EPOCHS = 40

def add_plot(image, title, index):
    plt.subplot(TEST_SIZE, 4, index)
    plt.title(title)
    plt.imshow(image)
    plt.axis('off')



def get_tensors_from_perreplica(per_replica):
# based on: https://github.com/tensorflow/tensorflow/blob/8d25e4bf616b7ae4ed101c580a23421616bf674c/tensorflow/python/distribute/values.py#L332
    if strategy.num_replicas_in_sync > 1:
        tensors_list = per_replica.values
        y = tf.concat(tensors_list, axis=0)
    else:
        y = per_replica
    
    return y


def generate_image(generator_g, generator_f, sample_horse, sample_zebra, epoch=1):

    plt.figure(figsize=(4 * 4, TEST_SIZE * 4))

    horse       = (sample_horse + 1 ) / 2
    zebra       = (sample_zebra + 1 ) / 2

    if TEST_SIZE == 1:
        sample_horse = tf.expand_dims(sample_horse, 0)
        sample_zebra = tf.expand_dims(sample_zebra, 0)


    horse_gen   = strategy.run(generator_g, args=(sample_horse,))
    zebra_gen   = strategy.run(generator_f, args=(sample_zebra,))

    horse_gen = (get_tensors_from_perreplica(horse_gen) + 1) / 2
    zebra_gen = (get_tensors_from_perreplica(zebra_gen) + 1) / 2

    cprint('horse_gen[0]:')
    print(horse_gen[0])

    for i in range(TEST_SIZE):
        add_plot(horse[i], 'input', 1 + 4 * i)
        add_plot(horse_gen[i], 'generated zebra', 2 + 4 * i)
        add_plot(zebra[i], 'input', 3 + 4 * i)
        add_plot(zebra_gen[i], 'generated horse', 4 + 4 * i)

    plt.savefig('images/generated/E%d.png' % (epoch+1))


@tf.function
def train_step(inputs):
    real_x, real_y = inputs

    with tf.GradientTape(persistent=True) as tape:
        fake_y      = generator_g(real_x, training=True)
        cycled_x    = generator_f(fake_y, training=True)

        fake_x      = generator_f(real_y, training=True)
        cycled_y    = generator_g(fake_x, training=True)
        
        same_x      = generator_f(real_x, training=True)
        same_y      = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        gen_g_loss  = generator_loss(disc_fake_y)
        gen_f_loss  = generator_loss(disc_fake_x)

        total_cycle_loss    = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        total_gen_g_loss    = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss    = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)
        disc_x_loss         = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss         = discriminator_loss(disc_real_y, disc_fake_y)

    generator_g_grads       = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
    generator_f_grads       = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)
    discriminator_x_grads   = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
    discriminator_y_grads   = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)

    generator_g_opt.apply_gradients(zip(generator_g_grads, generator_g.trainable_variables))
    generator_f_opt.apply_gradients(zip(generator_f_grads, generator_f.trainable_variables))
    discriminator_x_opt.apply_gradients(zip(discriminator_x_grads, discriminator_x.trainable_variables))
    discriminator_y_opt.apply_gradients(zip(discriminator_y_grads, discriminator_y.trainable_variables))

    return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss


for epoch in range(restored_epoch + 1, EPOCHS):
    start = time.time()

    n = 0
    for inputs in zip(train_horses, train_zebras):
        strategy.run(train_step, args=(inputs,))

        
        if n % 10 == 0:
            cprint('.', end='')
        n+=1

    cprint('')
    generate_image(generator_g, generator_f, sample_horse, sample_zebra, epoch)

    cprint('Time taken for epoch {} is {} sec'.format(epoch, time.time()-start))
    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        cprint('Saving checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))
