import tensorflow as tf
import os
import model
import data
import time
import save_load

class Training_framework():
    def __init__(self, a_dir, b_dir, beta_1, lr, batch, epochs, lsgan, cyc_lambda, vcpu, n_vcpu, load, load_dir):
        # CPU or virtual CPUs
        if vcpu == True and len(tf.config.experimental.list_physical_devices('GPU')) == 0:
            tf.config.experimental.set_virtual_device_configuration(
                tf.config.experimental.list_physical_devices("CPU")[0], [
                    tf.config.experimental.VirtualDeviceConfiguration() for i in range(n_vcpu)
                ])
            print('Created vCPU(s).')

        #parameters
        self.beta_1 = beta_1
        self.learning_rate = lr
        self.batch_size_per_tesla = batch
        self.epochs = epochs
        self.strategy = tf.distribute.MirroredStrategy()
        self.global_batch_size = batch * self.strategy.num_replicas_in_sync
        self.global_batch_no = 0
        self.epoch = 0
        self.lsgan = lsgan
        self.cyc_lambda = cyc_lambda
        self.models_dir = load_dir

        # paths
        self.trainA_path = a_dir
        self.trainB_path = b_dir

        #create or load models
        models = save_load.LoadSave.load_files(self.strategy, load, load_dir)
        if models:
            self.discA = models[0]
            self.discB = models[1]
            self.genA2B = models[2]
            self.genB2A = models[3]
        else:
            self.discA = model.Discriminator()
            self.discB = model.Discriminator()
            self.genA2B = model.Generator()
            self.genB2A = model.Generator()

        #optimizer
        def lr_schedule():
            if self.epoch < 15:
                return self.learning_rate
            if self.epoch < 35:
                return 1e-1 * self.learning_rate
            if self.epoch < 50:
                return 1e-2 * self.learning_rate
            return 1e-3 * self.learning_rate

        self.discA_opt = tf.keras.optimizers.Adam(lr_schedule, beta_1=beta_1)
        self.discB_opt = tf.keras.optimizers.Adam(lr_schedule, beta_1=beta_1)
        self.genA2B_opt = tf.keras.optimizers.Adam(lr_schedule, beta_1=beta_1)
        self.genB2A_opt = tf.keras.optimizers.Adam(lr_schedule, beta_1=beta_1)

        #pipeline
        self.ds = data.Pipeline.create_pipeline(self.strategy, self.trainA_path, self.trainB_path,
                                                batch_size=self.global_batch_size)
        self.ds_test = data.Pipeline.create_test_pipeline(self.trainA_path, self.trainB_path)



    @tf.function
    def train_step(self, dist_inputs):

        def discriminator_loss(disc_of_real_output, disc_of_gen_output, lsgan=True):
            if lsgan:  # least squares gan
                real_loss = tf.keras.losses.mean_squared_error(disc_of_real_output, tf.ones_like(disc_of_real_output))
                generated_loss = tf.reduce_mean(tf.square(disc_of_gen_output))

                total_disc_loss = (real_loss + generated_loss) / 2

            else:
                raise NotImplementedError

            return total_disc_loss

        def generator_loss(disc_of_gen_output, lsgan=True):
            if lsgan:  # least squares gan
                gen_loss = tf.keras.losses.mean_squared_error(disc_of_gen_output, tf.ones_like(disc_of_gen_output))
            else:
                raise NotImplementedError

            return gen_loss

        def cycle_consistency_loss(data_A, data_B, reconstructed_data_A, reconstructed_data_B, cyc_lambda=10):
            loss = tf.reduce_mean(tf.abs(data_A - reconstructed_data_A) + tf.abs(data_B - reconstructed_data_B))
            return cyc_lambda * loss


        def step_fn(inputs):
            inputA, inputB = inputs

            with tf.GradientTape() as genA2B_tape, tf.GradientTape() as genB2A_tape, \
                    tf.GradientTape() as discA_tape, tf.GradientTape() as discB_tape:
                genA2B_output = self.genA2B(inputA, training=True)
                genB2A_output = self.genB2A(inputB, training=True)

                discA_real_output = self.discA(inputA, training=True)
                discB_real_output = self.discB(inputB, training=True)

                discA_fake_output = self.discA(genB2A_output, training=True)
                discB_fake_output = self.discB(genA2B_output, training=True)

                reconstructedA = self.genB2A(genA2B_output, training=True)
                reconstructedB = self.genA2B(genB2A_output, training=True)

                # generate_images(reconstructedA, reconstructedB)

                # Use history buffer of 50 for disc loss
                discA_loss = discriminator_loss(discA_real_output, discA_fake_output, lsgan=self.lsgan) * \
                             (1.0 / self.global_batch_size)
                discB_loss = discriminator_loss(discB_real_output, discB_fake_output, lsgan=self.lsgan) * \
                             (1.0 / self.global_batch_size)

                genA2B_loss = (generator_loss(discB_fake_output, lsgan=self.lsgan) +
                              cycle_consistency_loss(inputA, inputB, reconstructedA, reconstructedB,
                                                     cyc_lambda=self.cyc_lambda)) * \
                             (1.0 / self.global_batch_size)
                genB2A_loss = (generator_loss(discA_fake_output, lsgan=self.lsgan) +
                              cycle_consistency_loss(inputA, inputB, reconstructedA, reconstructedB,
                                                     cyc_lambda=self.cyc_lambda)) * \
                             (1.0 / self.global_batch_size)

            genA2B_gradients = genA2B_tape.gradient(genA2B_loss, self.genA2B.trainable_variables)
            genB2A_gradients = genB2A_tape.gradient(genB2A_loss, self.genB2A.trainable_variables)

            discA_gradients = discA_tape.gradient(discA_loss, self.discA.trainable_variables)
            discB_gradients = discB_tape.gradient(discB_loss, self.discB.trainable_variables)

            self.genA2B_opt.apply_gradients(zip(genA2B_gradients, self.genA2B.trainable_variables))
            self.genB2A_opt.apply_gradients(zip(genB2A_gradients, self.genB2A.trainable_variables))

            self.discA_opt.apply_gradients(zip(discA_gradients, self.discA.trainable_variables))
            self.discB_opt.apply_gradients(zip(discB_gradients, self.discB.trainable_variables))

            loss = (discA_loss, discB_loss, genA2B_loss, genB2A_loss)

            return loss

        loss = self.strategy.run(step_fn, args=(dist_inputs,))

        mean_loss_discA = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, loss[0], axis=0)
        mean_loss_discB = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, loss[1], axis=0)
        mean_loss_genA2B = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, loss[2], axis=0)
        mean_loss_genB2A = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, loss[3], axis=0)

        return


    # run on CPU
    def inference(self):
        for batch in self.ds_test:
            inputA, inputB = batch

            genA2B_output = self.genA2B(inputA, training=False)
            genB2A_output = self.genB2A(inputB, training=False)

            reconstructedA = self.genB2A(genA2B_output, training=False)
            reconstructedB = self.genA2B(genB2A_output, training=False)

            images = tf.data.Dataset.from_tensor_slices(
                (inputA, inputB, genA2B_output, genB2A_output, reconstructedA, reconstructedB))

            data.Image.generate_images(images, self.epoch, self.global_batch_no)

            break


    def train(self):
        print('Start training on %d device(s)' % self.strategy.num_replicas_in_sync)

        training_start = time.time()

        for self.epoch in range(self.epochs):
            epoch_start = time.time()
            self.global_batch_no = 0
            for distributed_batch in self.ds:
                self.global_batch_no += 1
                batch_start = time.time()
                with self.strategy.scope():
                    self.train_step(distributed_batch)
                print('Time taken for epoch {} and global batch {} is {} sec'.format(self.epoch+1,
                                                                                     self.global_batch_no,
                                                                                     time.time()-batch_start))
                print('Image throughput: %.3f images/s' % ((self.global_batch_size) / (time.time()-batch_start)))

                if self.epoch == 0:
                    save_load.LoadSave.save_models(self.models_dir,
                                                   self.discA,
                                                   self.discB,
                                                   self.genA2B,
                                                   self.genB2A,
                                                   self.epoch,
                                                   self.global_batch_no)
                    if self.global_batch_no % 40 == 0: self.inference()

                elif self.epoch == 1:
                    if self.global_batch_no % 100 == 0: self.inference()

                else:
                    if self.global_batch_no % 300 == 0: self.inference()

            self.inference()

            #save model
            save_load.LoadSave.save_models(self.models_dir,
                                           self.discA,
                                           self.discB,
                                           self.genA2B,
                                           self.genB2A,
                                           self.epoch,
                                           self.global_batch_no)

            print('Time taken for whole epoch {} is {} sec\n'.format(self.epoch + 1, time.time() - epoch_start))

        print('Time taken for whole training is {} sec\n'.format(time.time() - training_start))
