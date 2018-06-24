import numpy as np
import time
import datetime
from enum import Flag, auto
import tensorflow as tf

from keras.layers import Input, Concatenate, BatchNormalization, Dropout #Dense, Reshape, Flatten,Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv3D, UpSampling3D, ZeroPadding3D, Cropping3D
from keras.models import Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard

from data_loader3D import DataLoader
import helper as hp

class Enhancement(Flag):
    SPATIAL_UP = auto()
    SPATIAL_DOWN = auto()   # noch nicht erledigt
    FREQUENCY_UP = auto()   # noch nicht erledigt
    FREQUENCY_MIN = auto()  # noch nicht erledigt
    GENERATIVE_SPATIAL = auto() # muss noch genauer überlegt werden     # noch nicht erledigt
    GENERATIVE_FREQUENCY = auto() # muss noch genauer überlegt werden   # noch nicht erledigt


class Pix3Pix():
    def __init__(self, vol_size, vol_depth, d_name, enhancement):
        # Input shape
        self.vol_rows = vol_size
        self.vol_cols = vol_size
        self.vol_depth = vol_depth
        self.channels = 1
        self.vol_shape = (self.vol_rows, self.vol_cols, self.vol_depth, self.channels)
        self.input_shape = [None,self.vol_rows,self.vol_cols,self.vol_depth,self.channels]

        # Distinguish for stack manipulation
        self.depth_two_potency = hp.check_for_two_potency(self.vol_depth)
        self.enhancement = enhancement
        if self.depth_two_potency:
            self.e_v = 0
        else:
            self.e_v = self.calculate_stack_manipulation()

        # Configure data loader
        self.dataset_name = d_name
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      vol_size=self.vol_shape)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.vol_rows / 2**4)
        self.disc_patch = (patch, patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

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

        # Save the model weights after each epoch if the validation loss decreased
        p = time.strftime("%Y-%m-%d_%H_%M_%S")
        self.checkpointer = ModelCheckpoint(filepath="logs/{}_CP".format(p), verbose=1, #filepath="logs/{}.base".format(p), verbose=1,
                                            save_best_only=True, mode='min')

        self.tensorboard = TensorBoard(log_dir="logs/{}".format(p), histogram_freq=0, batch_size=8,
            write_graph=False, write_grads=True, write_images=False, embeddings_freq=0,
            embeddings_layer_names=None, embeddings_metadata=None)

        self.combined = Model(inputs=[vol_A, vol_B], outputs=[valid, fake_A], name='combined')
        self.combined.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=optimizer)
        print(self.combined.summary())
        print('finish Pix3Pix __init__')

    def build_generator(self):
        """U-Net Generator"""
        self.stack_downsamling = []

        def conv3d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
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

            print('upsampling:  ', cnt, '\t\t', u.shape)
            u = Concatenate()([u, skip_input])
            return u

        # Image input (distinguish for two potency stack number)
        d0 = Input(shape=self.vol_shape)
        print('model input:\t\t\t', d0.shape)
        if not self.depth_two_potency:
            d0_m = self.manipulate_input_stack(d0)
        # print('d0_m shape ', d0_m.shape)

        self.stack_downsamling.append(int(d0.shape[3]))
        c = 0

        # Downsampling
        if not self.depth_two_potency:
            d1 = conv3d(d0_m, self.gf, bn=False); c += 1
        else:
            d1 = conv3d(d0, self.gf, bn=False); c += 1
        d2 = conv3d(d1, self.gf*2); c += 1
        d3 = conv3d(d2, self.gf*4); c += 1
        d4 = conv3d(d3, self.gf*8); c += 1
        d5 = conv3d(d4, self.gf*8); c += 1
        d6 = conv3d(d5, self.gf*8); c += 1
        d7 = conv3d(d6, self.gf*8); c += 1

        print('manipulation of stack axes:', self.stack_downsamling, c)
        # Upsampling
        u1 = deconv3d(d7, d6, self.gf*8, cnt=c); c -= 1
        u2 = deconv3d(u1, d5, self.gf*8, cnt=c); c -= 1
        u3 = deconv3d(u2, d4, self.gf*8, cnt=c); c -= 1
        u4 = deconv3d(u3, d3, self.gf*4, cnt=c); c -= 1
        u5 = deconv3d(u4, d2, self.gf*2, cnt=c); c -= 1
        u6 = deconv3d(u5, d1, self.gf, cnt=c); c -= 1

        u7 = UpSampling3D(size=2, data_format="channels_last")(u6)
        output_vol = Conv3D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        # model output (distinguish for two potency stack number)
        if not self.depth_two_potency:
            output_vol = self.manipulate_output_stack(output_vol)
        print('model output:', c, '\t\t', output_vol.shape)
        return Model(d0, output_vol)

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

        validity = Conv3D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([vol_A, vol_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        # # TODO: ones und zeros vertauschen?!? für min.
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (vols_A, vols_B) in enumerate(self.data_loader.load_batch(batch_size, k_size=9)):
                # reshape images
                print('test0', vols_A[0].shape, vols_B[0].shape)
                vols_A, vols_B = np.expand_dims(vols_A, axis=4), np.expand_dims(vols_B, axis=4)

                print('test1', vols_A[0].shape, vols_B[0].shape)
                #print('eine epoche', batch_i, vols_A.shape)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(vols_B)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([vols_A, vols_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, vols_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Fit the model
                g_loss = self.combined.fit([vols_A, vols_B], [valid, vols_A],
                                            validation_split=0.1,
                                            verbose=0,
                                            epochs=1,
                                            batch_size=batch_size,
                                            callbacks=[self.tensorboard, self.checkpointer])
                g_loss = g_loss.history['loss']

                # Train the generators (alternative way for train the combined model)
                # g_loss = self.combined.train_on_batch([vols_A, vols_B], [valid, vols_A])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs-1,
                                                                    batch_i, self.data_loader.n_batches,
                                                                    d_loss[0], 100*d_loss[1],
                                                                    g_loss[0],
                                                                    elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

        time_elapsed = datetime.datetime.now() - start_time
        print('\nFinish training in (hh:mm:ss.ms) {}'.format(time_elapsed))
        return self.discriminator, self.generator, self.combined

    def calculate_stack_manipulation(self):
        if self.enhancement == Enhancement.SPATIAL_UP:
            stack_diff = hp.calculate_stack_resize(self.vol_depth, 'up')
        elif self.enhancement == Enhancement.FREQUENCY_UP:
            stack_diff = hp.calculate_stack_resize(self.vol_depth, 'up')
        elif self.enhancement == Enhancement.FREQUENCY_MIN:
            stack_diff = hp.calculate_stack_resize(self.vol_depth, 'min')
        elif self.enhancement == Enhancement.GENERATIVE_SPATIAL:
            stack_diff = hp.calculate_stack_resize(self.vol_depth, 'down')
        elif self.enhancement == Enhancement.GENERATIVE_FREQUENCY:
            stack_diff = hp.calculate_stack_resize(self.vol_depth, 'down')
        return stack_diff

    def manipulate_input_stack(self, inputlayer):
        pad_crop = ( (0, 0), (0, 0), hp.calculate_padding_value(self.e_v) )
        if self.enhancement == Enhancement.SPATIAL_UP:
            output = ZeroPadding3D(padding=pad_crop, data_format="channels_last")(inputlayer)
        elif self.enhancement == Enhancement.FREQUENCY_UP:
            print(inputlayer)
            input = tf.placeholder(tf.complex64)
            sess = tf.InteractiveSession()
            sess.run()

            inputlayer = tf.cast(inputlayer, tf.complex64)
            print(inputlayer)
            vol_fft = tf.spectral.fft3d(inputlayer)

            vol_fftshift = vol_fft.eval()
            vol_fftshift = np.fft.fftshift(vol_fftshift)
            vol_fft_crop = tf.Variable(vol_fftshift)
            vol_fft_crop = ZeroPadding3D(cropping=pad_crop, data_format="channels_last")(inputlayer)
            vol_fftshift = vol_fft_crop.eval()
            vol_fftshift = np.fft.ifftshift(vol_fftshift)
            vol_fft = tf.Variable(vol_fftshift)
            output = tf.ifft3d(vol_fft)
            output = tf.abs(output)
            output = tf.round(output)
            sess.close()

        return output

    def manipulate_output_stack(self, inputlayer):
        pad_crop = ( (0, 0), (0, 0), hp.calculate_padding_value(self.e_v) )
        if self.enhancement == Enhancement.SPATIAL_UP:
            output = Cropping3D(cropping=pad_crop, data_format="channels_last")(inputlayer)
        elif self.enhancement == Enhancement.FREQUENCY_UP:
            vol_fft = tf.spectral.fft3d(inputlayer)
            sess = tf.InteractiveSession()
            vol_fftshift = vol_fft.eval()
            vol_fftshift = np.fft.fftshift(vol_fftshift)
            vol_fft_crop = tf.Variable(vol_fftshift)
            vol_fft_crop = Cropping3D(cropping=pad_crop, data_format="channels_last")(inputlayer)
            vol_fftshift = vol_fft_crop.eval()
            vol_fftshift = np.fft.ifftshift(vol_fftshift)
            vol_fft = tf.Variable(vol_fftshift)
            output = tf.ifft3d(vol_fft)
            output = tf.abs(output)
            output = tf.round(output)
            sess.close()

        return output


    # def downsample(d, fourier_transform=False):
    #     '''
    #         remove padding of the stack depth from 4 to 3
    #     ''
    #     vol_fft = vol_fft[:,:,:3]
    #     back = tf.spectral.irfft3d(vol_fft)
    #     back = tf.abs(back)
    #     back = tf.round(back)
    #
    #     return back
