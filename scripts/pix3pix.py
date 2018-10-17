# -*- coding: utf-8 -*-
"""
@author: philipp
"""

import numpy as np
import pandas as pd
import time
import datetime
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import filters
import tifffile as tiff
import tensorflow as tf

from keras.layers import Input, Concatenate, BatchNormalization, Dropout, GaussianNoise
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv3D, UpSampling3D, ZeroPadding3D, Cropping3D
from keras.layers.pooling import MaxPooling3D, AveragePooling3D
from keras.models import Model
from keras import optimizers, losses
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard

from data_loader3D import DataLoader3D
import helper as hp


class Pix3Pix():

    def __init__(self, vol_original, grid_search=False, parameter=None):
        # Import settings
        if grid_search:
            self.settings = parameter
        else:
            with open('{}/config.json'.format(os.path.dirname(__file__))) as json_data:
                self.settings = json.load(json_data)['selected']
        # Load OTF/PSF information from json-config
        with open('{}/config.json'.format(os.path.dirname(__file__))) as json_data:
            self.OTF_info = json.load(json_data)['OTF']

        vol_resize = ( self.settings['RESIZE']['width'],
                       self.settings['RESIZE']['height'],
                       self.settings['RESIZE']['depth'] )

        # Configure data loader
        self.__dataset_name = self.settings['DATASET_NAME']
        self.data_loader = DataLoader3D(micro_noise=self.settings['MICRO_NOISE_NPhot'],
                                        d_name=self.__dataset_name,
                                        manipulation=self.settings['MANIPULATION_STACKS'],
                                        vol_original=vol_original,
                                        vol_resize=vol_resize,
                                        otf=self.OTF_info,
                                        augm_factor=self.settings['DATA_AUGMENTATION_FACTOR'])

        # Input shape
        self.__channels = 1
        self.__vol_shape = self.data_loader.vol_resize+(self.__channels,)
        self.__vol_rows = self.settings['RESIZE']['width']
        self.__vol_cols = self.settings['RESIZE']['height']
        self.__vol_depth = self.settings['RESIZE']['depth']

        # Calculate output shape of D (PatchGAN)
        if self.settings['NETWORK_DEPTH'] == 'HIGH':
            network__depth_factor = 4
        elif self.settings['NETWORK_DEPTH'] == 'MEDIUM':
            network__depth_factor = 3
        elif self.settings['NETWORK_DEPTH'] == 'LOW':
            network__depth_factor = 2
        patch = int(self.__vol_rows / 2**network__depth_factor)
        patch_depth = int(np.ceil(self.__vol_depth / 2**network__depth_factor))
        self.__disc_patch = (patch, patch, patch_depth, 1)

        # Number of filters in the first layer of G and D
        self.__gf = 64
        self.__df = 64

        self.__adam = optimizers.Adam(self.settings['ADAM_OPTIMIZER_LEARNRATE'], 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.settings['D_LOSS'], optimizer=self.__adam, metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()
        # self.generator.compile(loss='mae', optimizer=adam)

        # Input images and their conditioning images
        vol_A = Input(shape=self.__vol_shape)
        vol_B = Input(shape=self.__vol_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(vol_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, vol_B])

        # Loss value that will be minimized by the model will be the sum of all individual losses
        self.combined = Model(inputs=[vol_A, vol_B], outputs=[valid, fake_A], name='combined')
        self.combined.compile(loss=self.settings['COMBINED_LOSS'],
                            loss_weights=self.settings['LOSS_WEIGHTS'],
                            optimizer=self.__adam)

        # Save the model weights after each epoch if the validation loss decreased
        p = time.strftime("%Y-%m-%d_%H_%M_%S")
        if self.settings['SAVE_LOGS']:
            # self.__checkpointer = ModelCheckpoint(filepath="logs/{}_CP".format(p), verbose=1,
            #                                     save_best_only=True, mode='min')
            # self.__writer = tf.summary.FileWriter("./logs/{}".format(p))
            self.__tensorboard = TensorBoard(log_dir="./logs/{}".format(p), histogram_freq=2, batch_size=1,
                write_graph=True, write_grads=True, write_images=True, embeddings_freq=0,
                embeddings_layer_names=None, embeddings_metadata=None)
            self.__tensorboard.set_model(self.combined)

        self.train_information = pd.DataFrame(columns=self.save_loss(get_header=True, save_loss=False))

        print('finish Pix3Pix __init__\n')


    def build_generator(self):
        """U-Net Generator"""

        def conv3d(layer_input, filters, f_size=3, bn=True):
            """Layers used during downsampling"""
            if self.settings['POOLING'] == "MAX":
                d = Conv3D(filters, kernel_size=f_size, strides=1, padding='same')(layer_input)
                d = MaxPooling3D(padding='same', data_format="channels_last")(d)
            elif self.settings['POOLING'] == "AVERAGE":
                d = Conv3D(filters, kernel_size=f_size, strides=1, padding='same')(layer_input)
                d = AveragePooling3D(padding='same', data_format="channels_last")(d)
            elif self.settings['POOLING'] == "NONE":
                d = Conv3D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)

            # GANHACKS: add dropout with specified value
            if self.settings['GANHACKS']:
                d = Dropout(rate=self.settings['DROPOUT'])(d)
            if bn and self.settings['BATCH_NORMALIZATION']:
                d = BatchNormalization(momentum=0.8)(d)
            # GANHACKS: adding gaussian noise to every layer of G (Zhao et. al. EBGAN)
            if self.settings['GANHACKS']:
                d = GaussianNoise(stddev=self.settings['GAUSSIAN_NOISE_TO_G'])(d)
            print('downsampling:\t\t\t', d.shape)
            return d

        def deconv3d(layer_input, skip_input, filters, f_size=3):
            """Layers used during upsampling"""
            # padding-and crop-size must have size 2 in sum before and after dimension
            if skip_input.shape[3] == 1:
                u = UpSampling3D(size=(2, 2, 1), data_format="channels_last")(layer_input)
            else:
                u = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(layer_input)
            u = Conv3D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)

            # GANHACKS: add dropout with specified value
            if self.settings['GANHACKS']:
                u = Dropout(rate=self.settings['DROPOUT'])(u)
            if self.settings['BATCH_NORMALIZATION']:
                u = BatchNormalization(momentum=0.8)(u)
            # GANHACKS: adding gaussian noise to every layer of G (Zhao et. al. EBGAN)
            if self.settings['GANHACKS']:
                u = GaussianNoise(stddev=self.settings['GAUSSIAN_NOISE_TO_G'])(u)

            print('upsampling:\t\t\t', u.shape)
            u = Concatenate()([u, skip_input])
            return u

        d0 = Input(shape=self.__vol_shape)
        print('generator-model input:\t\t', d0.shape)

        # Downsampling
        d1 = conv3d(d0, self.__gf, bn=False)
        d2 = conv3d(d1, self.__gf*2)
        d3 = conv3d(d2, self.__gf*4)
        d4 = conv3d(d3, self.__gf*8)
        if self.settings['NETWORK_DEPTH'] == 'MEDIUM':
            d5 = conv3d(d4, self.__gf*8)
        elif self.settings['NETWORK_DEPTH'] == 'HIGH':
            d5 = conv3d(d4, self.__gf*8)
            d6 = conv3d(d5, self.__gf*8)
            # d7 = conv3d(d6, self.__gf*8)

        # Upsampling
        if self.settings['NETWORK_DEPTH'] == 'HIGH':
            # u1 = deconv3d(d7, d6, self.__gf*8)
            # u2 = deconv3d(u1, d5, self.__gf*8)
            u2 = deconv3d(d6, d5, self.__gf*8)
            u3 = deconv3d(u2, d4, self.__gf*8)
            u4 = deconv3d(u3, d3, self.__gf*4)
        elif self.settings['NETWORK_DEPTH'] == 'MEDIUM':
            u3 = deconv3d(d5, d4, self.__gf*8)
            u4 = deconv3d(u3, d3, self.__gf*4)
        elif self.settings['NETWORK_DEPTH'] == 'LOW':
            u4 = deconv3d(d4, d3, self.__gf*4)
        u5 = deconv3d(u4, d2, self.__gf*2)
        u6 = deconv3d(u5, d1, self.__gf)

        u7 = UpSampling3D(size=2, data_format="channels_last")(u6)
        u7 = ZeroPadding3D(padding=(1, 1, 1), data_format="channels_last")(u7)
        output_vol = Conv3D(self.__channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)
        output_vol = Cropping3D(data_format="channels_last")(output_vol)

        print('generator-model output:\t\t', output_vol.shape)
        return Model(d0, output_vol, name='generator')

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=3, bn=True):
            """Discriminator layer"""
            # d = Conv3D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)

            d = Conv3D(filters, kernel_size=f_size, strides=1, padding='same')(layer_input)
            # d = AveragePooling3D(padding='same', data_format="channels_last")(d)
            d = MaxPooling3D(padding='same', data_format="channels_last")(d)

            d = LeakyReLU(alpha=0.2)(d)
            if bn and self.settings['BATCH_NORMALIZATION']:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        vol_A = Input(shape=self.__vol_shape)
        vol_B = Input(shape=self.__vol_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_vols = Concatenate(axis=-1)([vol_A, vol_B])

        d1 = d_layer(combined_vols, self.__df, bn=False)
        d2 = d_layer(d1, self.__df*2)
        d3 = d_layer(d2, self.__df*4)
        d4 = d_layer(d3, self.__df*8)

        if self.settings['NETWORK_DEPTH'] == 'LOW':
            validity = Conv3D(1, kernel_size=3, strides=1, padding='same')(d2)
        elif self.settings['NETWORK_DEPTH'] == 'MEDIUM':
            validity = Conv3D(1, kernel_size=3, strides=1, padding='same')(d3)
        elif self.settings['NETWORK_DEPTH'] == 'HIGH':
            validity = Conv3D(1, kernel_size=3, strides=1, padding='same')(d4)

        print('discriminator-model in/output:\t', vol_A.shape, vol_B.shape, '\n\t\t\t\t', validity.shape)
        return Model([vol_A, vol_B], validity, name='discriminator')

    def train(self, epochs, sample_interval=50):
        batch_size = self.settings['BATCH_SIZE']
        p = time.strftime("%Y-%m-%d_%H_%M_%S")
        start_time = datetime.datetime.now()

        # save GAN config as json
        self.save_config(p)

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.__disc_patch)
        fake = np.zeros((batch_size,) + self.__disc_patch)

        if self.settings['GANHACKS']:
            # GANHACKS: one-sided label smooting
            valid = valid * self.settings['ONE-SIDED-LABEL']

        batch_counter, d_l, g_l, weight_change = 0, 10000., 10000., False
        for epoch in range(epochs):
            for batch_i, (vols_A, vols_B, v_A_aug, v_B_aug) in \
                    enumerate(self.data_loader.load_batch(batch_size)):

                # GANHACKS: flip labels of discriminator randomly
                flip = False
                if self.settings['GANHACKS'] and np.random.rand() < self.settings['FLIP_LABEL_PROB'] and not self.settings['INSTANCE_NOISE']:
                    valid, fake = fake, valid
                    flip = True

                # expand channel dimension/reshape images
                vols_A, vols_B = np.expand_dims(vols_A, axis=4), np.expand_dims(vols_B, axis=4)
                v_A_aug, v_B_aug = np.expand_dims(v_A_aug, axis=4), np.expand_dims(v_B_aug, axis=4)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(vols_B)

                # GANHACKS: add some artificial noise to inputs to D
                if self.settings['INSTANCE_NOISE']:
                    d_loss, g_loss = self.add_artificial_noise_to_D(vols_A.copy(), vols_B.copy(), valid, fake, epochs, epoch, batch_i, start_time)
                    elapsed_time = datetime.datetime.now() - start_time
                    if self.settings['DATA_AUGMENTATION_FACTOR'] > 0:
                        self.train_on_augmentated_data(v_A_aug, v_B_aug, start_time, epoch, epochs, batch_i, flip)
                else:
                    if epoch >= int(self.settings['NUMBER_ONLY_TRAIN_G']):
                        # Train the discriminators (original images = real / generated = Fake)
                        d_loss_real = self.discriminator.train_on_batch([vols_A, vols_B], valid)
                        d_loss_fake = self.discriminator.train_on_batch([fake_A, vols_B], fake)
                        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                        if self.settings['SAVE_LOGS']:
                            self.save_log(['discriminator_loss', 'adverserial_loss'], [d_loss[0], 100*d_loss[1]], batch_i)

                    if self.settings['DATA_AUGMENTATION_FACTOR'] > 0:
                        self.train_on_augmentated_data(v_A_aug, v_B_aug, start_time, epoch, epochs, batch_i, flip)

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Train the generators
                    g_loss = self.combined.train_on_batch([vols_A, vols_B], [valid, vols_A])

                    l_1 = hp.L1_norm(vols_A.squeeze(), fake_A.squeeze())
                    l_2 = hp.L2_norm(vols_A.squeeze(), fake_A.squeeze())
                    elapsed_time = datetime.datetime.now() - start_time
                    if epoch >= int(self.settings['NUMBER_ONLY_TRAIN_G']):
                        loss_msg = "[Epoch %d/%d][Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] t: %s %s" % (epoch, epochs-1,
                                    batch_i, self.data_loader.n_batches-1, d_loss[0], 100*d_loss[1], g_loss[0], elapsed_time, flip)
                        self.save_loss(False, False, epoch, epochs-1, batch_i, self.data_loader.n_batches-1,
                            d_loss[0], 100*d_loss[1], g_loss[0], elapsed_time, flip, l_1, l_2)
                    else:
                        loss_msg = "[Epoch %d/%d][Batch %d/%d] [D loss: ----, acc: ----] [G loss: %f] t: %s %s" % (epoch, epochs-1,
                                    batch_i, self.data_loader.n_batches-1, g_loss[0], elapsed_time, flip)
                        self.save_loss(False, False, epoch, epochs-1, batch_i, self.data_loader.n_batches-1,
                            99.9, 99.9, g_loss[0], elapsed_time, flip, l_1, l_2)
                    print(loss_msg)

                if self.settings['SAVE_LOGS']:
                    self.save_log(['generator_loss'], [g_loss[0]], batch_i)

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    if self.settings['SAVE_TABLE_IMAGES']:
                        self.save_table_images(epoch, batch_i, p)
                    if self.settings['SAVE_VOLUME']:
                        self.save_volume(epoch, batch_i, p)

            # check if loss were improved and save model is actived
            if self.settings['SAVE_MODEL'] and epoch > 7 and d_l > d_loss[0] and g_l > g_loss[0]:
                d_l, g_l = d_loss[0], g_loss[0]
                hp.keras_model_saver(gan=self, p=p, path="./../models/")
            # increase "loss_weights" after specified percentage over all epochs
            if self.settings['INCREASE_LOSS_WEIGHTS'] < (epoch/epochs) and not weight_change:
                self.combined.compile(loss=self.settings['COMBINED_LOSS'],
                                    loss_weights=[10, self.settings['LOSS_WEIGHTS'][1]],
                                    optimizer=self.__adam)
                weight_change = True
                print('change loss weights:', self.combined.loss_weights)
                continue

        time_elapsed = datetime.datetime.now() - start_time

        # save loss and other information
        self.save_loss(get_header=False, save_loss=True, p=p)

        print('\nFinish training in (hh:mm:ss.ms) {}'.format(time_elapsed))

    def RainersReLU(self, x):
        '''Rainers non-exponential liner unit
        # Arguments
            x: Input tensor.
        # Returns
            The non-exponential linear activation: `x` if `x > 0` and
            `(1 / (1-x)) - 1` if `x < 0`.
        # References
            Rainer Heintzmann
        '''
        # return (K.sigmoid(x) * 5) - 1
        if tf.greater_equal(x, tf.constant([0])):
            return keras.activations.linear(x)
        else:
            return (1 / (1-x)) - 1

    def save_table_images(self, epoch, batch_i, p):
        def colorbar(Mappable, Orientation='vertical', Extend='both'):
            Ax = Mappable.axes
            fig = Ax.figure
            divider = make_axes_locatable(Ax)
            Cax = divider.append_axes("right", size="5%", pad=0.08)
            return fig.colorbar(
                mappable=Mappable,
                cax=Cax,
                use_gridspec=True,
                extend=Extend,  # mostra um colorbar full resolution de z
                orientation=Orientation
            )

        directory = 'images/{0}/{0}_{1}'.format(self.__dataset_name, p)
        os.makedirs(directory, exist_ok=True)

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=1)
        imgs_A, imgs_B = np.expand_dims(imgs_A, axis=4), np.expand_dims(imgs_B, axis=4)

        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_A, imgs_B, fake_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        gen_imgs = np.squeeze(gen_imgs, axis=4)

        titles = ['Original-Volume', 'OTF-Condition', 'Generated-sample']
        dims = ['x-dim', 'y-dim', 'z-dim']

        # plot a maximum intensity projection for each dimension
        r, c = 3, 3
        fig, axs = plt.subplots(nrows=r, ncols=c, figsize=(20,20))
        t = 'max projection about x,y,z dimension from {}-dataset\nimage size: {}'.format(self.__dataset_name, self.__vol_shape)
        plt.suptitle(t, fontsize=25)

        for i in range(r):
            axs[i,0].set_ylabel(dims[i], fontsize=25)
            for j in range(c):
                temp = axs[i,j].imshow(np.max(gen_imgs[j,:,:,:], axis=i), cmap='gray')
                colorbar(temp)
                axs[i,j].set_xticks([]); axs[i,j].set_yticks([])
                axs[0,j].set_title(titles[j], fontsize=25)

        fig.tight_layout()
        plt.subplots_adjust(left=0.02, wspace=0.1, top=0.92)
        fig.savefig('{0}/{1}_{2}.png'.format(directory, epoch, batch_i))
        plt.close()

    def save_volume(self, epoch, batch_i, p):
        directory = 'images/{0}/{0}_{1}_VOLUMES'.format(self.__dataset_name, p)
        if not os.path.exists(directory):
            os.makedirs(directory)

        vol_A, vol_B = self.data_loader.load_data(batch_size=1)
        vol_A, vol_B = np.expand_dims(vol_A, axis=4), np.expand_dims(vol_B, axis=4)
        fake_A = self.generator.predict(vol_B)
        vol_A, vol_B, fake_A = vol_A[0,:,:,:,0], vol_B[0,:,:,:,0], fake_A[0,:,:,:,0]
        vol_A, vol_B, fake_A = vol_A.astype(np.uint8), vol_B.astype(np.uint8), fake_A.astype(np.uint8)

        tiff.imsave('{0}/{1}_{2}_Original.tif'.format(directory, epoch, batch_i), vol_A)
        tiff.imsave('{0}/{1}_{2}_OTF.tif'.format(directory, epoch, batch_i), vol_B)
        tiff.imsave('{0}/{1}_{2}_Generated.tif'.format(directory, epoch, batch_i), fake_A)

    # def conv_summary(self, layer):
    #     w     = layer.weights[0]
    #     shape = layer.get_weights()[0].shape
    #     ix    = shape[0]
    #     iy    = shape[1]
    #     cin   = shape[2]
    #     cout  = shape[3]
    #
    #     w = tf.reshape(w,(iy,ix,cin*cout))
    #     ix += 2
    #     iy += 2
    #     w = tf.image.resize_image_with_crop_or_pad(w,iy,ix)
    #     w = tf.reshape(w,(iy,ix,cin,cout))
    #     w = tf.transpose(w, perm=[2,0,3,1]) # ix,iy,cin,cout -> cout,iy,cin,ix
    #     w = tf.reshape(w, (1,iy*cin,ix*cout,1))
    #     return w

    def save_log(self, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.__tensorboard.writer.add_summary(summary, batch_no)
            self.__tensorboard.writer.flush()

            # summary = tf.Summary(value=[
            #     tf.Summary.Value(tag=name, simple_value=value),
            # ])
            # self.__writer.add_summary(summary)

            # tf.summary.image('feature maps', self.conv_summary(self.generator.layers[1]))
            # merged = tf.summary.merge_all()

    def save_config(self, p):
        directory = 'images/{0}/{0}_{1}'.format(self.__dataset_name, p)
        if not os.path.exists(directory):
            os.makedirs(directory)
        file = '{0}/{1}_{2}.json'.format(directory, self.__dataset_name, p)
        with open(file, 'w') as outfile:
            temp = {"hyper_parameter": self.settings, "OTF": self.OTF_info}
            json.dump(temp, outfile, indent=4)

    def save_loss(self, get_header, save_loss, e_c=None, e_a=None, b_c=None, b_a=None,
                d_l=None, a=None, g_l=None, t=None, f_l=None, l_1=None, l_2=None, p=None):
        """ Either
                - commit header of file or
                - save entire DataFrame or
                - append the actual training information
        """

        h = ['epoch_count', 'epoch_all', 'batch_count', 'batch_all', 'D_loss',
            'accuracy', 'G_loss', 'time', 'flip_label', 'L_1', 'L_2' ]

        if get_header:
            return h
        elif save_loss:
            directory = 'images/{0}/{0}_{1}'.format(self.__dataset_name, p)
            os.makedirs(directory, exist_ok=True)

            file = '{0}/{1}.csv'.format(directory, p)
            self.train_information.to_csv(file, sep=',')
        else:
            results = [e_c, e_a, b_c, b_a, d_l, a, g_l, t, f_l, l_1, l_2]
            df = pd.DataFrame(data=[results], columns=h)
            self.train_information = self.train_information.append(df)

    def train_on_augmentated_data(self, vols_A_aug, vols_B_aug, s_time, epoch, epochs, batch_i, flip):
        valid = np.ones((self.settings['BATCH_SIZE']*self.settings['DATA_AUGMENTATION_FACTOR'],) + self.__disc_patch)
        fake = np.zeros((self.settings['BATCH_SIZE']*self.settings['DATA_AUGMENTATION_FACTOR'],) + self.__disc_patch)
        if self.settings['GANHACKS']:
            valid = valid * self.settings['ONE-SIDED-LABEL']

        # Condition on B and generate a translated version
        fake_A_aug = self.generator.predict(vols_B_aug)

        if epoch >= int(self.settings['NUMBER_ONLY_TRAIN_G']):
            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch([vols_A_aug, vols_B_aug], valid)
            d_loss_fake = self.discriminator.train_on_batch([fake_A_aug, vols_B_aug], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            if self.settings['SAVE_LOGS']:
                self.save_log(['discriminator_loss', 'adverserial_loss'], [d_loss[0], 100*d_loss[1]], batch_i)

        g_loss = self.combined.train_on_batch([vols_A_aug, vols_B_aug], [valid, vols_A_aug])
        if self.settings['SAVE_LOGS']:
            self.save_log(['generator_loss'], [g_loss[0]], batch_i)

        elapsed_time = datetime.datetime.now() - s_time
        l_1 = hp.L1_norm(vols_A_aug.squeeze(), fake_A_aug.squeeze())
        l_2 = hp.L2_norm(vols_A_aug.squeeze(), fake_A_aug.squeeze())
        if epoch >= int(self.settings['NUMBER_ONLY_TRAIN_G']):
            loss_msg = "[Epoch %d/%d][Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] t: %s %s A" % (epoch, epochs-1,
                        batch_i, self.data_loader.n_batches-1, d_loss[0], 100*d_loss[1], g_loss[0], elapsed_time, flip)
            self.save_loss(False, False, epoch, epochs-1, batch_i, self.data_loader.n_batches-1,
                d_loss[0], 100*d_loss[1], g_loss[0], elapsed_time, flip, l_1, l_2)
        else:
            loss_msg = "[Epoch %d/%d][Batch %d/%d] [D loss: ----, acc: ----] [G loss: %f] t: %s %s A" % (epoch, epochs-1,
                        batch_i, self.data_loader.n_batches-1, g_loss[0], elapsed_time, flip)
            self.save_loss(False, False, epoch, epochs-1, batch_i, self.data_loader.n_batches-1,
                99.9, 99.9, g_loss[0], elapsed_time, flip, l_1, l_2)
        print(loss_msg)

    def add_artificial_noise_to_D(self, vols_A, vols_B, valid, fake, e_s, e, b_i, s_t):
        '''
            Add some artificial noise to inputs to D, create instance noise
            With 50% probability it is guassian/poisson noise
            After putting noise to pixels, convolve images
            image values are between [-1,1]
        '''

        # Condition on B and generate a translated version
        fake_A = self.generator.predict(vols_B)
        # vols_A = vols_A[:,:,:,:,0]
        # put noise to real and fake images, which D sees: 50% gaussian/poisson
        if np.random.rand() < 0.5:
            vols_A = np.random.normal(vols_A)
            vols_B = np.random.normal(vols_B)
            fake_A = np.random.normal(fake_A)
        else:
            vols_A = vols_A + np.random.poisson(lam=0.2, size=vols_A.shape)
            vols_B = vols_B + np.random.poisson(lam=0.2, size=vols_B.shape)
            fake_A = fake_A + np.random.poisson(lam=0.2, size=vols_B.shape)

        vols_A[vols_A < -1] = -1; vols_A[vols_A > 1] = 1
        vols_B[vols_B < -1] = -1; vols_B[vols_B > 1] = 1
        fake_A[fake_A < -1] = -1; fake_A[fake_A > 1] = 1

        # filter each batch
        sig = 1
        for i in range(vols_A.shape[0]):
            v_A = vols_A[i,:,:,:,0]; v_B = vols_B[i,:,:,:,0]; f_A = fake_A[i,:,:,:,0]
            vols_A[i,:,:,:,0] = filters.gaussian_filter(v_A, sigma=(sig, sig, sig), order=0)
            vols_B[i,:,:,:,0] = filters.gaussian_filter(v_B, sigma=(sig, sig, sig), order=0)
            fake_A[i,:,:,:,0] = filters.gaussian_filter(f_A, sigma=(sig, sig, sig), order=0)

        if epoch >= int(self.settings['NUMBER_ONLY_TRAIN_G']):
            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch([vols_A, vols_B], valid)
            d_loss_fake = self.discriminator.train_on_batch([vols_A, vols_B], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # Train Generator
        g_loss = self.combined.train_on_batch([vols_A, vols_B], [valid, vols_A])

        elapsed_time = datetime.datetime.now() - s_t
        # Plot the progress
        loss_msg = "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] t: %s [IN] " % (e, e_s-1,
                        b_i, self.data_loader.n_batches-1, d_loss[0], 100*d_loss[1], g_loss[0], elapsed_time)
        print(loss_msg)
        return d_loss, g_loss

    # def resize_like(input_tensor, ref_tensor): # resizes input tensor wrt. ref_tensor
    #     H, W = ref_tensor.get_shape()[1], ref.get_shape()[2]
    #     return tf.image.resize_nearest_neighbor(inputs, [H.value, W.value])

    def upsample3D(self, input_tensor):
        H, W, D = input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]
        return tf.image.resize_nearest_neighbor(inputs, [H.value, W.value])
# resized_tensor = Lambda(resize_like, arguments={'ref_tensor':ref_tensor})(input_tensor)
