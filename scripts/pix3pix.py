# -*- coding: utf-8 -*-
"""
@author: philipp
"""

import numpy as np
import time
import datetime
import os
import json
import csv
import matplotlib.pyplot as plt
import tifffile as tiff
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
    def req_key(self, key):
        return list(self.settings[key].keys())[0]

    def __init__(self, vol_original):
        # Import settings
        with open('{}/config.json'.format(os.path.dirname(__file__))) as json_data:
            self.settings = json.load(json_data)['selected']

        vol_resize = ( self.settings['RESIZE']['width'],
                       self.settings['RESIZE']['height'],
                       self.settings['RESIZE']['depth'] )

        # Configure data loader
        self.dataset_name = self.settings['DATASET_NAME']
        self.data_loader = DataLoader3D(micro_noise=self.settings['ADD_MICRO_NOISE'],
                                        d_name=self.settings['DATASET_NAME'],
                                        manipulation=self.settings['MANIPULATION_STACKS'],
                                        vol_original=vol_original,
                                        vol_resize=vol_resize,
                                        norm=self.settings["PSF_OTF_NORM"])

        # Input shape
        self.channels = 1
        self.vol_shape = self.data_loader.vol_resize+(self.channels,)
        self.vol_rows = self.settings['RESIZE']['width']
        self.vol_cols = self.settings['RESIZE']['height']
        self.vol_depth = self.vol_shape[2]

        # Calculate output shape of D (PatchGAN)
        if self.settings['NETWORK_DEPTH'] == 'HIGH':
            network__depth_factor = 4
        elif self.settings['NETWORK_DEPTH'] == 'MEDIUM':
            network__depth_factor = 3
        elif self.settings['NETWORK_DEPTH'] == 'LOW':
            network__depth_factor = 2
        patch = int(self.vol_rows / 2**network__depth_factor)
        patch_depth = int(np.ceil(self.vol_depth / 2**network__depth_factor))
        self.disc_patch = (patch, patch, patch_depth, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        # GANHACKS: train with improve-techniques for GANs
        self.ganhacks = self.settings['GANHACKS']
        if self.ganhacks:
            self.true_label = self.settings['ONE-SIDED-LABEL']
            # alternative: = np.around(np.random.uniform(low=.7, high=1.2, decimals=1)

        loss = losses.kullback_leibler_divergence
        adam = optimizers.Adam(0.0001, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse', optimizer=adam, metrics=['accuracy'])

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
        # If the model has multiple outputs, you can use a different loss on each output by passing a dictionary or a list of losses.
        # loss value that will be minimized by the model will be the sum of all individual losses
        self.combined = Model(inputs=[vol_A, vol_B], outputs=[valid, fake_A], name='combined')
        # self.combined.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=adam)
        self.combined.compile(loss=['kullback_leibler_divergence', 'mae'], loss_weights=[0, 10], optimizer=adam)

        # Save the model weights after each epoch if the validation loss decreased
        p = time.strftime("%Y-%m-%d_%H_%M_%S")
        if self.settings['SAVE_LOGS']:
            self.checkpointer = ModelCheckpoint(filepath="logs/{}_CP".format(p), verbose=1,
                                                save_best_only=True, mode='min')

            self.tensorboard = TensorBoard(log_dir="logs/{}".format(p), histogram_freq=0, batch_size=1,
                write_graph=True, write_grads=True, write_images=False, embeddings_freq=0,
                embeddings_layer_names=None, embeddings_metadata=None)
            self.tensorboard.set_model(self.combined)

        print('finish Pix3Pix __init__')


    def build_generator(self):
        """U-Net Generator"""

        # TODO: mit filer size 3 filtern und testen
        # def conv3d(layer_input, filters, f_size=3, bn=True, dropout_prob=0.4)
        def conv3d(layer_input, filters, f_size=4, bn=True, dropout_prob=0.4):
            """Layers used during downsampling"""
            d = Conv3D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            # GANHACKS: add dropout with 'dropout_prob'%
            if self.ganhacks and (np.random.rand() < dropout_prob):
                d = Dropout(rate=self.settings['DROPOUT'])(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)

            print('downsampling:\t\t\t', d.shape)
            return d

        def deconv3d(layer_input, skip_input, filters, f_size=4, dropout_prob=0.4):
            """Layers used during upsampling"""
            if skip_input.shape[3] == 1:
                u = UpSampling3D(size=(2, 2, 1), data_format="channels_last")(layer_input)
            else:
                u = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(layer_input)

            u = Conv3D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            # GANHACKS: add dropout with 'dropout_prob'%
            if self.ganhacks and (np.random.rand() < dropout_prob):
                u = Dropout(rate=self.settings['DROPOUT'])(u)
            u = BatchNormalization(momentum=0.8)(u)

            print('upsampling:\t\t\t', u.shape)
            u = Concatenate()([u, skip_input])
            return u

        d0 = Input(shape=self.vol_shape)
        print('generator-model input:\t\t', d0.shape)

        # Downsampling
        d1 = conv3d(d0, self.gf, bn=False)
        d2 = conv3d(d1, self.gf*2)
        d3 = conv3d(d2, self.gf*4)
        d4 = conv3d(d3, self.gf*8)
        if self.settings['NETWORK_DEPTH'] == 'MEDIUM':
            d5 = conv3d(d4, self.gf*8)
        elif self.settings['NETWORK_DEPTH'] == 'HIGH':
            d5 = conv3d(d4, self.gf*8)
            d6 = conv3d(d5, self.gf*8)
            d7 = conv3d(d6, self.gf*8)

        # Upsampling
        if self.settings['NETWORK_DEPTH'] == 'HIGH':
            u1 = deconv3d(d7, d6, self.gf*8)
            u2 = deconv3d(u1, d5, self.gf*8)
            u3 = deconv3d(u2, d4, self.gf*8)
            u4 = deconv3d(u3, d3, self.gf*4)
        elif self.settings['NETWORK_DEPTH'] == 'MEDIUM':
            u3 = deconv3d(d5, d4, self.gf*8)
            u4 = deconv3d(u3, d3, self.gf*4)
        elif self.settings['NETWORK_DEPTH'] == 'LOW':
            u4 = deconv3d(d4, d3, self.gf*4)
        u5 = deconv3d(u4, d2, self.gf*2)
        u6 = deconv3d(u5, d1, self.gf)

        u7 = UpSampling3D(size=2, data_format="channels_last")(u6)
        output_vol = Conv3D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        print('generator-model output:\t\t', output_vol.shape)
        return Model(d0, output_vol, name='generator')

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv3D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        vol_A = Input(shape=self.vol_shape)
        vol_B = Input(shape=self.vol_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_vols = Concatenate(axis=-1)([vol_A, vol_B])

        d1 = d_layer(combined_vols, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        if self.settings['NETWORK_DEPTH'] == 'LOW':
            validity = Conv3D(1, kernel_size=4, strides=1, padding='same')(d2)
        elif self.settings['NETWORK_DEPTH'] == 'MEDIUM':
            validity = Conv3D(1, kernel_size=4, strides=1, padding='same')(d3)
        elif self.settings['NETWORK_DEPTH'] == 'HIGH':
            validity = Conv3D(1, kernel_size=4, strides=1, padding='same')(d4)

        print('discriminator-model in/output:\t', vol_A.shape, vol_B.shape, '\n\t\t\t\t', validity.shape)
        return Model([vol_A, vol_B], validity, name='discriminator')

    def train(self, epochs, batch_size=1, sample_interval=50):
        p = time.strftime("%Y-%m-%d_%H_%M_%S")
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths (6 = 1 original volume + 5 noise volumes)
        if (0): #self.ganhacks:
            valid = np.ones((5*batch_size,) + self.disc_patch)
            fake = np.zeros((5*batch_size,) + self.disc_patch)
        else:
            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)

        if self.ganhacks:
            # GANHACKS: one-sided label smooting
            valid = valid * self.true_label
            # GANHACKS: flip labels of discriminator randomly
            if np.random.rand() < self.settings['FLIP_LABEL_PROB']:
                valid, fake = fake, valid

        for epoch in range(epochs):
            for batch_i, (vols_A, vols_B) in \
                    enumerate(self.data_loader.load_batch(batch_size, self.settings['ADD_MICRO_NOISE'])):
                # expand channel dimension/reshape images
                vols_A, vols_B = np.expand_dims(vols_A, axis=4), np.expand_dims(vols_B, axis=4)

                # return 0

                # ---------------------
                #  Train Discriminator
                # ---------------------
                # TODO: hier evtl gpu_options = tf.GPUOptions(allow_growth=True)
                    # session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
                    # GPU gesteuerte Session starten

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(vols_B)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([vols_A, vols_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, vols_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                 # Train the generators
                g_loss = self.combined.train_on_batch([vols_A, vols_B], [valid, vols_A])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                loss_msg = "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs-1,
                                batch_i, self.data_loader.n_batches-1, d_loss[0], 100*d_loss[1], g_loss[0], elapsed_time)
                print (loss_msg)

                if self.settings['SAVE_LOSS']:
                    self.save_loss(loss_msg, p)

                if self.settings['SAVE_LOGS']:
                    self.save_log(g_loss, batch_i)

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    if self.settings['SAVE_TABLE_IMAGES']:
                        self.save_table_images(epoch, batch_i, p)
                    if self.settings['SAVE_VOLUME']:
                        self.save_volume(epoch, batch_i, p)

        time_elapsed = datetime.datetime.now() - start_time

        # If specified => save GAN config file as json
        if self.settings['SAVE_CONFIG']:
            self.save_config(p)

        print('\nFinish training in (hh:mm:ss.ms) {}'.format(time_elapsed))

    def save_table_images(self, epoch, batch_i, p):
        directory = 'images/{0}/{0}_{1}'.format(self.dataset_name, p)
        os.makedirs(directory, exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=1)
        imgs_A, imgs_B = np.expand_dims(imgs_A, axis=4), np.expand_dims(imgs_B, axis=4)

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
        fig.savefig('{0}/{1}_{2}.png'.format(directory, epoch, batch_i))
        plt.close()

    def save_volume(self, epoch, batch_i, p):
        directory = 'images/{0}/{0}_{1}_VOLUMES'.format(self.dataset_name, p)
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

    def save_log(self, logs, batch_no):
        names = ['train_loss', 'discriminator_loss', 'generator_loss']
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.tensorboard.writer.add_summary(summary, batch_no)
            self.tensorboard.writer.flush()

    def save_config(self, p):
        directory = 'images/{0}/{0}_{1}'.format(self.dataset_name, p)
        if not os.path.exists(directory):
            os.makedirs(directory)
        file = '{0}/{1}_{2}.json'.format(directory, self.dataset_name, p)
        with open(file, 'w') as outfile:
            json.dump(self.settings, outfile)

    def save_loss(self, msg, p):
        directory = 'images/{0}/{0}_{1}_{2}'.format(self.dataset_name,
            p, self.settings['MANIPULATION_STACKS'])
        if not os.path.exists(directory):
            os.makedirs(directory)
        file = '{0}/{1}_{2}.csv'.format(directory, self.dataset_name, p)

        arr = msg.replace('[', '').replace(']', '').replace('%', '').replace('/', ' ')
        arr = np.array(arr.split(' ')).take(indices=[1, 2, 4, 5, 8, 11, 14, 16])

        # hd = ['Epoch val', 'Epoch of', 'Batch val', 'Batch of', 'D_loss', 'acc', 'G_loss', 'time']
        with open(file,'a') as f1:
            writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
            writer.writerow(arr)

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
