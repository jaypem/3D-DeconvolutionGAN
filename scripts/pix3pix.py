# -*- coding: utf-8 -*-
"""
@author: philipp
"""

import numpy as np
import time
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.layers import Input, Concatenate, BatchNormalization, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv3D, UpSampling3D, ZeroPadding3D, Cropping3D
from keras.models import Model
from keras import optimizers, losses
from keras.callbacks import ModelCheckpoint, TensorBoard, History

from data_loader3D import DataLoader3D
import helper as hp



class Pix3Pix():
    def __init__(self, vol_original, vol_resize, d_name, stack_manipulation, simulation, ganhacks):
        self.ganhacks = ganhacks
        # Configure data loader
        self.dataset_name = d_name
        self.data_loader = DataLoader3D(simulation=simulation,
                                        d_name=self.dataset_name,
                                        manipulation=stack_manipulation,
                                        vol_original=vol_original,
                                        vol_resize=vol_resize)

        # Input shape
        self.channels = 1
        self.vol_shape = self.data_loader.vol_resize+(self.channels,)
        self.vol_rows = self.vol_shape[0]
        self.vol_depth = self.vol_shape[2]

        print('mögliche Fehlerquelle bei Berechnung von patch_depth -> self.disc_patch : ...2**3))#4))')
        # Calculate output shape of D (PatchGAN)
        patch = int(self.vol_rows / 2**3) #4)
        patch_depth = int(np.ceil(self.vol_depth / 2**3))#4))
        self.disc_patch = (patch, patch, patch_depth, 1)
        # self.disc_patch = (patch*patch*patch_depth,)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        loss = losses.kullback_leibler_divergence
        optimizer = optimizers.Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        vol_A = Input(shape=self.vol_shape)
        vol_B = Input(shape=self.vol_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(vol_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, vol_B])

        # TODO: # ACHTUNG:
        # If the model has multiple outputs, you can use a different loss on each output
        # by passing a dictionary or a list of losses.
        # The loss value that will be minimized by the model will then be the sum of all individual losses.
        self.combined = Model(inputs=[vol_A, vol_B], outputs=[valid, fake_A], name='combined')
        self.combined.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=optimizer)
        # self.combined.compile(loss=[loss, 'mae'], loss_weights=[1, 100], optimizer=optimizer)

        # Save the model weights after each epoch if the validation loss decreased
        p = time.strftime("%Y-%m-%d_%H_%M_%S")
        # self.checkpointer = ModelCheckpoint(filepath="logs/{}_CP".format(p), verbose=1,
                                            # save_best_only=True, mode='min')

        # self.tensorboard = TensorBoard(log_dir="logs/{}".format(p), histogram_freq=0, batch_size=1,
        #     write_graph=False, write_grads=True, write_images=False, embeddings_freq=0,
        #     embeddings_layer_names=None, embeddings_metadata=None)
        # self.tensorboard.set_model(self.combined)

        print('finish Pix3Pix __init__')


    def build_generator(self):
        """U-Net Generator"""
        self.stack_downsamling = []

        def conv3d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""

            # # TODO: einziger Weg wie man vor dem conv. das padding selbst bestimmen kann
            # padded_input = tf.pad(input, [[0, 0], [2, 2], [1, 1], [0, 0]], "CONSTANT")
            # output = tf.nn.conv2d(padded_input, filter, strides=[1, 1, 1, 1], padding="VALID")

            d = Conv3D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)

            self.stack_downsamling.append(int(d.shape[3]))
            print('downsampling:\t\t\t', d.shape)
            return d

        def deconv3d(layer_input, skip_input, filters, cnt, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            # TODO: ist das hier genug nur auf den vorletzten zu überprüfen?? generisch schreiben?
            if self.stack_downsamling[cnt-1] == 1:
                u = UpSampling3D(size=(2, 2, 1), data_format="channels_last")(layer_input)
            else:
                u = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(layer_input)

            u = Conv3D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)

            # TODO: hier stimmt die Ausgabe nicht, von data_loader-objekt holen
            print('upsampling:\t\t\t', u.shape)
            u = Concatenate()([u, skip_input])
            return u

        # Image input (distinguish for two potency stack number)
        d0 = Input(shape=self.vol_shape)
        print('generator-model input:\t\t', d0.shape)
        self.stack_downsamling.append(int(d0.shape[3])); c = 0

        # Downsampling
        d1 = conv3d(d0, self.gf, bn=False); c += 1
        d2 = conv3d(d1, self.gf*2); c += 1
        d3 = conv3d(d2, self.gf*4); c += 1
        d4 = conv3d(d3, self.gf*8); c += 1
        d5 = conv3d(d4, self.gf*8); c += 1
        # d6 = conv3d(d5, self.gf*8); c += 1
        # d7 = conv3d(d6, self.gf*8); c += 1

        # Upsampling
        # u1 = deconv3d(d7, d6, self.gf*8, cnt=c); c -= 1
        # u2 = deconv3d(u1, d5, self.gf*8, cnt=c); c -= 1
        # u3 = deconv3d(u2, d4, self.gf*8, cnt=c); c -= 1
        u3 = deconv3d(d5, d4, self.gf*8, cnt=c); c -= 1
        u4 = deconv3d(u3, d3, self.gf*4, cnt=c); c -= 1
        u5 = deconv3d(u4, d2, self.gf*2, cnt=c); c -= 1
        u6 = deconv3d(u5, d1, self.gf, cnt=c); c -= 1

        u7 = UpSampling3D(size=2, data_format="channels_last")(u6)
        output_vol = Conv3D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        print('generator-model output:\t\t', d0.shape, output_vol.shape)
        return Model(d0, output_vol, name='generator')

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv3D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Image input (distinguish for two potency stack number)
        vol_A = Input(shape=self.vol_shape)
        vol_B = Input(shape=self.vol_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_vols = Concatenate(axis=-1)([vol_A, vol_B])

        d1 = d_layer(combined_vols, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        # d4 = d_layer(d3, self.df*8)

        validity = Conv3D(1, kernel_size=4, strides=1, padding='same')(d3) #(d4)
        # validity = Flatten(data_format="channels_last")(validity)
        print('discriminator-model in/output:\t', vol_A.shape, vol_B.shape, '\n\t\t\t\t', validity.shape)
        return Model([vol_A, vol_B], validity, name='discriminator')

    def train(self, epochs, batch_size=1, sample_interval=50, add_noise=False):
        p = time.strftime("%Y-%m-%d_%H_%M_%S")
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths (6 = 1 original volume + 5 noise volumes)
        if add_noise:
            valid = np.ones((5*batch_size,) + self.disc_patch)
            fake = np.zeros((5*batch_size,) + self.disc_patch)
        else:
            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)

        if self.ganhacks:
            fake_label = np.around(np.random.uniform(low=0.0, high=0.3), decimals=1)
            fake = fake + fake_label
            true_label = np.around(np.random.uniform(low=0.7, high=1.2), decimals=1)
            valid = valid * true_label

        for epoch in range(epochs):
            for batch_i, (vols_A, vols_B) in enumerate(self.data_loader.load_batch(batch_size, add_noise)):
                # expand channel dimension/reshape images
                vols_A, vols_B = np.expand_dims(vols_A, axis=4), np.expand_dims(vols_B, axis=4)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                # TODO: hier evtl gpu_options = tf.GPUOptions(allow_growth=True)
                    # session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
                    # GPU gesteuerte Session starten

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(vols_B)
                # print('train-shapes:',vols_A.shape, vols_B.shape, valid.shape, self.disc_patch)
                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([vols_A, vols_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, vols_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Fit the model
                g_loss = self.combined.train_on_batch([vols_A, vols_B], [valid, vols_A])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs-1,
                                                                    batch_i, self.data_loader.n_batches-1, d_loss[0],
                                                                    100*d_loss[1], g_loss[0], elapsed_time))
                # self.write_log(g_loss, batch_i)

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i, p)

        time_elapsed = datetime.datetime.now() - start_time
        print('\nFinish training in (hh:mm:ss.ms) {}'.format(time_elapsed))

    def sample_images(self, epoch, batch_i, p):
        os.makedirs('images/{0}/{0}_{1}_{2}'.format(self.dataset_name, p, self.data_loader.manipulation.name), exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=1)
        imgs_A, imgs_B = np.expand_dims(imgs_A, axis=4), np.expand_dims(imgs_B, axis=4)

        # TODO: methode schreiben, die nur "fake_A", "imgs_A" und "imgs_B" speichert
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_A, imgs_B, fake_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        gen_imgs = np.squeeze(gen_imgs, axis=4)

        titles = ['Original-Volume', 'OTF-Condition', 'Generated-sample']
        # select 'r' (default=3) random adjacent stacks (altogether min. 3 stacks)
        s_i = np.random.randint(low=1, high=gen_imgs.shape[3]-1, size=1)[0]
        s_i = [s_i-1, s_i, s_i+1]

        fig, axs = plt.subplots(nrows=r, ncols=c, figsize=(20,20))
        t = '3 adjacent stacks from {}-dataset\nimage size: {}'.format(self.dataset_name, self.vol_shape)
        plt.suptitle(t, fontsize=25)

        for i in range(r):
            axs[i,0].set_ylabel('stack: {0}'.format(s_i[i]), fontsize=25)
            for j in range(c):
                axs[i,j].imshow(gen_imgs[j,:,:,s_i[i]], cmap='gray')
                axs[i,j].set_xticks([]); axs[i,j].set_yticks([])
                axs[0,j].set_title(titles[j], fontsize=25)

        fig.tight_layout()
        plt.subplots_adjust(left=0.02, wspace=0, top=0.92)
        fig.savefig('images/{0}/{0}_{1}_{4}/{2}_{3}_{4}.png'.format(self.dataset_name, p, epoch, batch_i, self.data_loader.manipulation.name))
        plt.close()

    def write_log(self, logs, batch_no):
        names = ['train_loss', 'discriminator_loss', 'generator_loss']
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.tensorboard.writer.add_summary(summary, batch_no)
            self.tensorboard.writer.flush()


    def resize_stack(self, inputlayer, upsample):
        '''
            CAUTION: - this works just for images with 1 channel,
                       because of the resize interpolation is between z-axes and channel(=1)
                     - batch does not have to be considered
            1. transpose(inputlayer): [batch, height, width, depth, channels]     => [batch, depth, channels, height, width]
            2. reshape(transposed):   [batch, depth, channels, height, width]     => [batch, depth, channels, height*width]
            3. resize(reshaped):      [batch, depth, channels, height*width]      => [batch, depth_new, channels, height*width]
            4. reshaped(resized):     [batch, depth_new, channels, height*width]  => [batch, depth_new, channels, height, width]
            5. transpose(reshaped):   [batch, depth_new, channels, height, width] => [batch, height, width, depth_new, channels]

        '''
        from keras import backend as K
        from keras.backend import tf as ktf
        from keras.layers import Lambda

        if self.vol_depth == 3:
            print('resize_stack: CAUTION - this works just for images with 1 channel')
        if upsample:
            y = self.calculate_stack_manipulation()
        else:
            y = self.calculate_stack_manipulation()*(-1)
        print('resize_stack_0:', inputlayer.shape)
        # transposed = tf.transpose( inputlayer, [0,3,4,1,2] )
        transposed = K.permute_dimensions( inputlayer, [0,3,4,1,2] )
        print('resize_stack_1:', transposed.shape)

        calc_shape = (tf.shape(inputlayer)[0], self.vol_depth, self.channels, self.vol_rows*self.vol_cols)
        # reshaped = tf.reshape( transposed, calc_shape )
        reshaped = K.reshape( transposed, calc_shape )
        print('resize_stack_2:', reshaped.shape)

        new_size = (self.vol_depth + y, 1) #[self.vol_depth + y, 1]
        # resized = tf.image.resize_images( reshaped , new_size, method=tf.image.ResizeMethod.BILINEAR )
        resized = Lambda(lambda image: ktf.image.resize_images( image , new_size, method=tf.image.ResizeMethod.BILINEAR ))(reshaped)
        print('resize_stack_3:', resized.shape)

        calc_shape = (tf.shape(resized)[0], int(resized.shape[1]), self.channels, self.vol_rows, self.vol_cols)
        # reshaped = tf.reshape( resized, calc_shape )
        reshaped = K.reshape( resized, calc_shape )
        print('resize_stack_4:', reshaped.shape)

        # transposed = tf.transpose( reshaped, [0,3,4,1,2] )
        transposed = K.permute_dimensions( reshaped, [0,3,4,1,2] )
        print('resize_stack_5:', transposed.shape)

        return transposed
