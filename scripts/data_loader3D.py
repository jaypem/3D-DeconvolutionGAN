# -*- coding: utf-8 -*-

from glob import glob
import numpy as np
from skimage import io
from skimage.transform import resize
import sys
import tensorflow as tf

import helper as hp
import deconvolution as deconv

import scipy.io

def print_volume_dimension(path, max_print=5):
    print('dimensions of volumes:')
    for i, p in enumerate(path):
        if i >= max_print:
            break
        try:
            vol = io.imread(p)
            print(i,vol.shape)
        except:
            print('skip file:', p)


class DataLoader3D():
    def __init__(self, micro_noise, d_name, manipulation, vol_original, vol_resize, norm="OTF"):
        self.add_microscopy_noise = micro_noise
        self.dataset_name = d_name
        self.vol_original = vol_original
        self.vol_resize = vol_resize[:3]
        self.manipulation = manipulation
        if hp.check_for_two_potency(self.vol_original[2]):
            self.e_v = 0
        else:
            self.e_v = hp.calculate_stack_manipulation(self.manipulation, self.vol_original[2], self.vol_resize[2])
        if not (self.manipulation == 'SPATIAL_RESIZE' or self.manipulation == 'NONE'):
            self.vol_resize = (self.vol_original[0], self.vol_original[1], self.vol_original[2]+self.e_v)
        path = glob('../data/3D/%s/*' % (self.dataset_name))
        self.path = [item for item in path if item.endswith('.tiff') or item.endswith('.tif')]
        self.vol_n = 0

        # TODO: hier relativ den Pfad einbauen von "data_loader3D"
        sys.path.insert(0, '../scripts/NanoImagingPack')
        from microscopy import PSF3D
        from transformations import ft

        if norm == "PSF":
            psf_temp = PSF3D(im=self.vol_original, ret_val = 'PSF')
            # psf_temp = PSF3D(im=self.vol_resize, ret_val = 'PSF')
            psf_temp = psf_temp/np.sum(psf_temp)
            # otf_norm = np.fft.fftn(psf_temp)
            otf_norm = ft(psf_temp, norm=None, force_full_fft=True)
            self.otf = otf_norm
            # print(np.sum(psf_temp))
        else:
            otf = PSF3D(im=self.vol_original, ret_val = 'OTF')
            # otf = PSF3D(im=self.vol_resize, ret_val = 'OTF')
            self.otf = otf/np.max(np.abs(otf))

            # print(np.max(np.abs(self.otf)))
            #otf_norm = np.fft.fftshift(otf_norm)
            # arr = {'psf_temp': psf_temp, 'otf_norm': otf_norm}
            # scipy.io.savemat('psf.mat', arr)
            #otf_norm = np.fft.ifftn(psf_norm)


    def load_data(self, batch_size=1, add_noise=False):
        batch_images = np.random.choice(self.path, size=batch_size)

        vols_A = []
        vols_B = []
        for vol in batch_images:
            vol_A = self.imread(vol)
            # vol_B = deconv.conv3d_fft(vol_A, self.otf)

            sigma = 2
            vol_B = scipy.ndimage.filters.gaussian_filter(vol_A, sigma=(sigma, sigma, sigma), order=0)

            vol_A = self.cut_volume(vol_A, self.vol_resize, centered=True)
            vol_B = self.cut_volume(vol_B, self.vol_resize, centered=True)

            if not self.manipulation == 'NONE':
                vol_A = self.manipulate_stack(vol_A)
                vol_B = self.manipulate_stack(vol_B)
            vols_A.extend([vol_A])

            # add poisson and gaussian noise to measurement
            if self.add_microscopy_noise:
                vol_B = deconv.add_poisson(vol_B) #+ deconv.create_gaussian_noise(vol_B)
            vols_B.extend([vol_B])

            if (0): #add_noise:
                # manipulate original image and convolve the manipulated images after that
                flip_A = deconv.flip_vol(vol_A)
                roll_A = deconv.roll_vol(vol_A, fraction=.1)
                shift_A = deconv.add_affineTransformation(vol_A)
                log_intensity_A = deconv.add_logIntensityTransformation(vol_A)

                flip_B = deconv.conv3d_fft(flip_A, self.otf)
                roll_B = deconv.conv3d_fft(roll_A, self.otf)
                shift_B = deconv.conv3d_fft(shift_A, self.otf)
                log_intensity_B = deconv.conv3d_fft(log_intensity_A, self.otf)
                if self.add_microscopy_noise:
                    flip_B = deconv.add_poisson(flip_B) + deconv.create_gaussian_noise(flip_B)
                    roll_B = deconv.add_poisson(roll_B) + deconv.create_gaussian_noise(roll_B)
                    shift_B = deconv.add_poisson(shift_B) + deconv.create_gaussian_noise(shift_B)
                    log_intensity_B = deconv.add_poisson(log_intensity_B) + deconv.create_gaussian_noise(log_intensity_B)

                vols_A.extend([flip_A, roll_A, shift_A, log_intensity_A])
                vols_B.extend([flip_B, roll_B, shift_B, log_intensity_B])

        vols_A = np.array(vols_A)/127.5 - 1.
        vols_B = np.array(vols_B)/127.5 - 1.

        return vols_A, vols_B

    def load_batch(self, batch_size, add_noise):
        self.n_batches = int(len(self.path) / batch_size)

        if self.n_batches == 1:
            print('CAUTION: n_batches = 1, data will not be loaded, length path:', len(self.path))
        elif self.n_batches-1 == -1 or self.n_batches-1 == 0:
            print('CAUTION: n_batches = 0 or n_batches = -1, data will not be loaded, check dataset name')

        for i in range(self.n_batches-1):
            batch = self.path[i*batch_size:(i+1)*batch_size]
            vols_A, vols_B = [], []

            for vol in batch:
                vol_A = self.imread(vol)
                # vol_B = deconv.conv3d_fft(vol_A, self.otf)

                sigma = 1 #2
                vol_B = scipy.ndimage.filters.gaussian_filter(vol_A, sigma=(sigma, sigma, sigma), order=0)

                # arr = {'vol_B': vol_B, 'vol_A': vol_A}
                # scipy.io.savemat('test.mat', arr)

                vol_A = self.cut_volume(vol_A, self.vol_resize, centered=True)
                vol_B = self.cut_volume(vol_B, self.vol_resize, centered=True)

                if not self.manipulation == 'NONE':
                    vol_A = self.manipulate_stack(vol_A)
                    vol_B = self.manipulate_stack(vol_B)

                vols_A.extend([vol_A])

                # add poisson and gaussian noise to measurement
                if self.add_microscopy_noise:
                    vol_B = deconv.add_poisson(vol_B) #deconv.create_gaussian_noise(vol_B) #+deconv.add_poisson(vol_B)
                vols_B.extend([vol_B])

                if (0): #add_noise:
                    # manipulate original image and convolve the manipulated images after that
                    flip_A = deconv.flip_vol(vol_A)
                    roll_A = deconv.roll_vol(vol_A, fraction=.1)
                    shift_A = deconv.add_affineTransformation(vol_A)
                    log_intensity_A = deconv.add_logIntensityTransformation(vol_A)

                    flip_B = deconv.conv3d_fft(flip_A, self.otf)
                    roll_B = deconv.conv3d_fft(roll_A, self.otf)
                    shift_B = deconv.conv3d_fft(shift_A, self.otf)
                    log_intensity_B = deconv.conv3d_fft(log_intensity_A, self.otf)
                    # https://arxiv.org/pdf/1711.04340.pdf Introduction
                    # random translations,rotations and flips as well as addition of Gaussian noise
                    if self.add_microscopy_noise:
                        flip_B = deconv.add_poisson(flip_B) + deconv.create_gaussian_noise(flip_B)
                        roll_B = deconv.add_poisson(roll_B) + deconv.create_gaussian_noise(roll_B)
                        shift_B = deconv.add_poisson(shift_B) + deconv.create_gaussian_noise(shift_B)
                        log_intensity_B = deconv.add_poisson(log_intensity_B) + deconv.create_gaussian_noise(log_intensity_B)

                    vols_A.extend([flip_A, roll_A, shift_A, log_intensity_A])
                    vols_B.extend([flip_B, roll_B, shift_B, log_intensity_B])

            vols_A = np.array(vols_A)/127.5 - 1.
            vols_B = np.array(vols_B)/127.5 - 1.

            yield vols_A, vols_B

# ****************************************************************************
# *                              Volume operations                           *
# ****************************************************************************

    def imread(self, path, swap_vol=False, colormode='L'):
        vol = io.imread(path)
        if swap_vol:
            return hp.swapAxes(vol, swap=True)
        elif self.vol_original != vol.shape:
            return hp.swapAxes(vol, swap=True)
        else:
            return vol

    def cut_volume(self, vol, resize, centered=True):
        rows, cols, depth = vol.shape
        if len(resize) == 2:
            r, c = resize[:2]
            r2, c2 = int(r/2), int(c/2)
        else:
            r, c, s = resize
            r2, c2, s2 = int(r/2), int(c/2), int(s/2)

        if centered:
            crow, ccol = int(rows/2), int(cols/2)
            if len(resize) == 3:
                cstack = int(depth/2)
        else:
            crow = np.random.randint(low=r2, high=rows-r2, size=1)[0]
            ccol = np.random.randint(low=c2, high=cols-c2, size=1)[0]
            if len(resize) == 3:
                cstack = np.random.randint(low=s2, high=depth-s2, size=1)[0]

        try:
            if len(resize) == 2:
                return vol[(crow-r2):(crow+r2), (ccol-c2):(ccol+c2), :]
            else:
                return vol[(crow-r2):(crow+r2), (ccol-c2):(ccol+c2), (cstack-s2):(cstack+s2)]
        except:
            print('ERROR by method: DataLoader3D.cut_volume, resize volume')
            return resize(vol_A, self.vol_size)

    def manipulate_stack(self, vol, pad_mode='linear_ramp'):
        pad = ((0,0), (0,0), hp.calculate_pad_crop_value(self.e_v))
        if self.manipulation == 'SPATIAL_UP':
            return np.pad(vol, pad_width=pad, mode=pad_mode)
        elif self.manipulation == 'SPATIAL_DOWN':
            return vol[:,:,:self.vol_resize[2]]
        elif self.manipulation == 'SPATIAL_MIN':
            x = hp.calculate_stack_resize(self.vol_original[2], 'min')[0]
            if 2**x < self.vol_original[2]:
                return vol[:,:,:self.vol_resize[2]]
            else:
                return np.pad(vol, pad_width=pad, mode=pad_mode)
        elif self.manipulation == 'SPATIAL_RESIZE':
            # interpolation: default bi-linear
            return resize(vol, self.vol_resize)#, anti_aliasing=True)
        elif self.manipulation == 'FREQUENCY_UP':
            vol_fft = np.fft.fftn(vol)
            vol_fftshift = np.fft.fftshift(vol_fft)
            vol_fftshift = np.pad(vol_fftshift, pad_width=pad, mode='constant')
            vol_fftshift = np.fft.ifftshift(vol_fftshift)
            vol_fft = np.fft.ifftn(vol_fftshift)
            return np.real(vol_fft*np.conj(vol_fft))
        elif self.manipulation == 'FREQUENCY_DOWN':
            vol_fft = np.fft.fftn(vol)
            vol_fftshift = np.fft.fftshift(vol_fft)
            vol_fftshift = vol_fftshift[:,:,:self.vol_resize[2]]
            vol_fftshift = np.fft.ifftshift(vol_fftshift)
            vol_fft = np.fft.ifftn(vol_fftshift)
            return np.real(vol_fft*np.conj(vol_fft))
        elif self.manipulation == 'FREQUENCY_MIN':
            x = hp.calculate_stack_resize(self.vol_original[2], 'constant')[0]
            vol_fft = np.fft.fftn(vol)
            vol_fftshift = np.fft.fftshift(vol_fft)
            if 2**x < self.vol_original[2]:
                vol_fftshift = vol_fftshift[:,:,:self.vol_resize[2]]
            else:
                vol_fftshift = np.pad(vol_fftshift, pad_width=pad, mode='edge')
            vol_fftshift = np.fft.ifftshift(vol_fftshift)
            vol_fft = np.fft.ifftn(vol_fftshift)
            return np.real(vol_fft*np.conj(vol_fft))

# ****************************************************************************
# *                             TFRecords functions                          *
# ****************************************************************************

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def writeTFRecord(self, filename, data='train', labels_given=False):
        '''
            create a TFRecord file an write images with corresponding informations

            filename:               name of the TFRecord file
            data:                   distinguish for data {train, test, validation}
            labels_given:           specified if labels are given (# TODO: implemnt label extraction)
        '''
        import time
        start_otf = time.time()
        # open the TFRecords file
        writer = tf.python_io.TFRecordWriter(filename)

        for i in range(len(self.path)):
            # print how many images are saved every 500 images
            if not i % 500:
                print('{} data: {}/{}'.format(data, i, len(self.path)))
                sys.stdout.flush()
            # Load the image
            img = self.imread(self.path[i]) #.astype(np.float32)

            if labels_given:
                print('todo: implement this part (not used for GANs)')#label = labels[i]

            # Create a feature
            if labels_given:
                print('todo: implement this part (not used for GANs)')
                # feature = {'{}/label'.format(data): self._int64_feature(label),
                #            '{}/image'.format(data): self._bytes_feature(tf.compat.as_bytes(img.tostring()))}
            else:
                feature = {'{}/image'.format(data): self._bytes_feature(img.tostring()), #tf.compat.as_bytes(img.tostring())),
                           '{}/height'.format(data): self._int64_feature(self.vol_original_size[0]),
                           '{}/width'.format(data): self._int64_feature(self.vol_original_size[1]),
                           '{}/stack'.format(data): self._int64_feature(self.vol_original_size[2])}

            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

        writer.close()
        sys.stdout.flush()
        print('Time for writeTFRecord :\t\t', time.time() - start_otf, 's')

    def input_fn(self, batch_size):
        files = tf.data.Dataset.list_files(FLAGS.data_dir)
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=4)
        dataset = dataset.shuffle(2048) # Sliding window of 2048 records
        dataset = dataset.repeat(NUM_EPOCHS)
        dataset = dataset.map(parser_fn, num_parallel_calls=64)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(2)
        return dataset

    # TODO: die variante verwerfen und read_and_decode testen/fixen/verwenden
    def readTFRecords(self, filenames, epochs, batches=1, labels_given=False):
        '''
            read a TFRecord file encode strings to extract images with corresponding informations

            filename:               name of the TFRecord file
            epochs:                 number of epochs
            data:                   distinguish for data [train, test, validation]
            labels_given:           specified if labels are given (# TODO: implemnt label extraction)
        '''
        data = ['train', 'test', 'validation']
        with tf.Session() as sess:
            if labels_given:
                feature = {'{}/image'.format(data[0]): tf.FixedLenFeature([], tf.string),
                           '{}/label'.format(data[0]): tf.FixedLenFeature([], tf.int64)}
            else:
                feature = {'{}/image'.format(data[0]): tf.FixedLenFeature([], tf.string)}

            # Create a list of filenames and pass it to a queue
            filename_queue = tf.train.string_input_producer(filenames, num_epochs=1)

            # Define a reader and read the next record
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)

            # Decode the record read by the reader
            features = tf.parse_single_example(serialized_example, features=feature)

            # Convert the image data from string back to the numbers
            image = tf.decode_raw(features['{}/image'.format(data[0])], tf.float32)

            # Cast label data into int32
            if labels_given:
                label = tf.cast(features['{}/label'.format(data[0])], tf.int32)

            # Reshape image data into the original shape
            image = tf.reshape(image, [self.vol_original_size[0], self.vol_original_size[1], self.vol_original_size[2]])
            print(image.shape)

            # TODO: Any preprocessing here ...

            # Creates batches by randomly shuffling tensors
            if labels_given:
                images, labels = tf.train.shuffle_batch([image, label], batch_size=batches, capacity=30, num_threads=1, min_after_dequeue=10)
            else:
                images = tf.train.shuffle_batch([image], batch_size=batches, capacity=30, num_threads=1, min_after_dequeue=10)

            # Initialize all global and local variables
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

            # Create a coordinator and run all QueueRunner objects
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # Stop the threads
            coord.request_stop()

            # Wait for threads to stop
            coord.join(threads)
            sess.close()

            return images

    def readTFRecords2(self, filenames, epochs, batches=1, data='train', labels_given=False):
        '''
            read a TFRecord file encode strings to extract images with corresponding informations

            filename:               name of the TFRecord file
            epochs:                 number of epochs
            data:                   distinguish for data [train, test, validation]
            labels_given:           specified if labels are given (# TODO: implemnt label extraction)
        '''
        record_iterator = tf.python_io.tf_record_iterator(path=filenames[0])

        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)

            height = int(example.features.feature['{}/height'.format(data)].int64_list.value[0])
            width = int(example.features.feature['{}/width'.format(data)].int64_list.value[0])
            stack = int(example.features.feature['{}/stack'.format(data)].int64_list.value[0])
            img_string = (example.features.feature['{}/image'.format(data)].bytes_list.value[0])

            img_1d = np.fromstring(img_string, dtype=np.uint8)
            return img_1d.reshape((height, width, stack))

    def read_and_decode(self, filenames, data='train', epochs=10, batches=1, labels_given=False):
        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=epochs)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        if labels_given:
            feature={
              '{}/height'.format(data): tf.FixedLenFeature([], tf.int64),
              '{}/width'.format(data): tf.FixedLenFeature([], tf.int64),
              '{}/stack'.format(data): tf.FixedLenFeature([], tf.int64),
              '{}/image'.format(data): tf.FixedLenFeature([], tf.string),
              '{}/label'.format(data): tf.FixedLenFeature([], tf.string)
              }
        else:
            feature={
              '{}/height'.format(data): tf.FixedLenFeature([], tf.int64),
              '{}/width'.format(data): tf.FixedLenFeature([], tf.int64),
              '{}/stack'.format(data): tf.FixedLenFeature([], tf.int64),
              '{}/image'.format(data): tf.FixedLenFeature([], tf.string)
              }

        # Decode the record read by the reader
        features = tf.parse_single_example( serialized_example, features=feature)

        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features['{}/image'.format(data)], tf.uint8)
        if labels_given:
            label = tf.decode_raw(features['{}/label'.format(data)], tf.uint8)

        height = tf.cast(features['{}/height'.format(data)], tf.int32)
        width = tf.cast(features['{}/width'.format(data)], tf.int32)
        stack = tf.cast(features['{}/stack'.format(data)], tf.int32)

        image = tf.reshape(image, self.vol_original_size)
        if labels_given:
            label = tf.reshape(annotation, annotation_shape)

        # image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.int32)
        # annotation_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=tf.int32)

        # Random transformations can be put here: right before you crop images
        # to predefined size. # TODO: Any preprocessing here ...

        # resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
        #                                        target_height=IMAGE_HEIGHT,
        #                                        target_width=IMAGE_WIDTH)
        #
        # resized_annotation = tf.image.resize_image_with_crop_or_pad(image=annotation,
        #                                        target_height=IMAGE_HEIGHT,
        #                                        target_width=IMAGE_WIDTH)
        # print(image)
        images = tf.train.shuffle_batch([image],
            batch_size=batches, capacity=30, num_threads=2, min_after_dequeue=10)

        # images, annotations = tf.train.shuffle_batch( [resized_image, resized_annotation],
        #                                              batch_size=2,
        #                                              capacity=30,
        #                                              num_threads=2,
        #                                              min_after_dequeue=10)

        return images
