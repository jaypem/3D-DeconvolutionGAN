import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
import cv2
import sys
import tensorflow as tf

import helper as hp
import deconvolution as deconv


def print_volume_dimension(path):
    print('dimensions of volumes:')
    for p in path:
        vol = io.imread(p)
        print(vol.shape)


class DataLoader3D():
    def __init__(self, dataset_name, vol_resize):
        self.dataset_name = dataset_name
        self.vol_size = vol_resize[:3]
        # self.vol_original_size = vol_original_size
        path = glob('../data/3D/%s/*' % (self.dataset_name))
        self.path = [item for item in path if not item.endswith('.txt')]
        self.vol_original_size = self.imread(np.random.choice(self.path, size=1)[0]).shape

        sys.path.insert(0, '../scripts/NanoImagingPack')
        from microscopy import PSF3D
        self.otf = PSF3D(im=self.vol_size, ret_val = 'OTF')


    def load_data(self, batch_size=1, add_noise=False):
        batch_images = np.random.choice(self.path, size=batch_size)

        vols_A = []
        vols_B = []
        for vol_path in batch_images:
            vol_A = self.imread(vol_path)
            # interpolation: default bi-linear
            vol_A = resize(vol_A, self.vol_size)

            vol_B = deconv.conv3d_fft(vol_A, self.otf)
            vol_B = resize(vol_B, self.vol_size)#, anti_aliasing=True)

            vols_A.extend([vol_A])
            vols_B.extend([vol_B])

            if add_noise:
                poisson_A, poisson_B = deconv.add_poisson(vol_A), deconv.add_poisson(vol_B)
                gauss_A, gauss_B = deconv.add_gaussian(vol_A), deconv.add_gaussian(vol_B)
                flip_A, flip_B = np.fliplr(vol_A), np.fliplr(vol_B)
                roll_A, roll_B = np.roll(vol_A, int(vol_A.shape[0]*.1)), np.roll(vol_B, int(vol_B.shape[0]*.1))
                shift_A, shift_B = deconv.add_shift(vol_A), deconv.add_shift(vol_B)
                vols_A.extend([poisson_A, gauss_A, flip_A, roll_A, shift_A])
                vols_B.extend([poisson_B, gauss_B, flip_B, roll_B, shift_B])

        vols_A = np.array(vols_A)/127.5 - 1.
        vols_B = np.array(vols_B)/127.5 - 1.

        return vols_A, vols_B

    def load_batch(self, batch_size=1, add_noise=False):
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
                # interpolation: default bi-linear
                vol_A = resize(vol_A, self.vol_size)#, anti_aliasing=True)
                vols_A.extend([vol_A])

                vol_B = deconv.conv3d_fft(vol_A, self.otf)
                vol_B = resize(vol_B, self.vol_size)#, anti_aliasing=True)
                vols_B.extend([vol_B])

                if add_noise:
                    # put noise on original and convolved image
                    poisson_A, poisson_B = deconv.add_poisson(vol_A), deconv.add_poisson(vol_B)
                    gauss_A, gauss_B = deconv.add_gaussian(vol_A), deconv.add_gaussian(vol_B)

                    # manipulate original image and convolve the manipulated images after that
                    flip_A = np.fliplr(vol_A)
                    roll_A = np.roll(vol_A, int(vol_A.shape[0]*.1))
                    shift_A = deconv.add_shift(vol_A)

                    flip_B = deconv.conv3d_fft(flip_A, self.otf)
                    roll_B = deconv.conv3d_fft(roll_A, self.otf)
                    shift_B = deconv.conv3d_fft(shift_A, self.otf)

                    vols_A.extend([poisson_A, gauss_A, flip_A, roll_A, shift_A])
                    vols_B.extend([poisson_B, gauss_B, flip_B, roll_B, shift_B])

            vols_A = np.array(vols_A)/127.5 - 1.
            vols_B = np.array(vols_B)/127.5 - 1.

            yield vols_A, vols_B

    def imread(self, path, colormode='L'):
        vol = io.imread(path)
        return hp.swapAxes(vol, swap=True)


### TFRecords functions ###

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
