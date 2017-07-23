import tensorflow as tf
import Utils.Eva_config_consts as config
import numpy as np
from enum import Enum
import sys


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class TFStorageOpenOptions(Enum):
    READ = 1
    WRITE = 2


def get_total_number_of_rows(path):
    """
    Counts total number of examples in a .tfrecords file

    :param path: path to a .tfrecords file
    :return: total number of examples
    """

    total_number = 0
    for record in tf.python_io.tf_record_iterator(path):
        total_number += 1
        if total_number % 10000 == 0:
            print total_number
    return total_number


class TFStorage(object):
    """
    The Storage based on TFRecords
    """

    def __init__(self, path, openOption, storageVolume=None):
        """
        Initializes the storage
        :param path: A path to a storage file (should have .tfrecords extension)
        :param openOption: Specifies if the storage object should be created for reading or writing
        :param storageVolume: Maximum number of rows
        """

        if not tf.gfile.Exists(path):
            raise ValueError('Failed to find file: ' + path)

        self.path = path
        self.openOption = openOption
        if openOption == TFStorageOpenOptions.WRITE:
            self.maximumNumberOfRows = storageVolume
            self.currentNumberOfRows = 0

    def __enter__(self):
        if self.openOption == TFStorageOpenOptions.WRITE:
            self.writer = tf.python_io.TFRecordWriter(self.path)
        else:
            self.reader = tf.TFRecordReader()
            self.filename_queue = tf.train.string_input_producer([self.path])
        return self

    def __exit__(self, *err):
        if self.openOption == TFStorageOpenOptions.WRITE:
            self.writer.close()

    def insert_row(self, item):
        """
        Inserts a row into the storage.

        :param item: a list of length 3 where:
            item[0] - spectrum
            item[1] - phoneme
            item[2] - speaker
        :return:
        """
        features = item[0]
        phoneme = item[1]
        speaker = item[2]

        features_raw = features.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(config.NUM_MEL_FREQ_COMPONENTS),
            'width': _int64_feature(config.FEATURES_CHUNK_LENGTH),
            'depth': _int64_feature(config.MFCC_DEPTH),
            'phoneme': _int64_feature(phoneme),
            'speaker': _int64_feature(speaker),
            'spectrum_raw': _bytes_feature(features_raw)}))

        self.writer.write(example.SerializeToString())

    def cut_phoneme_into_chunks_and_save(self, phoneme_features, chunkLength, phoneme, speaker):
        """
        Accepts a mfcc or spectrogram of arbitrary size of one concrete phoneme. Cuts chunks of size chunkLength which
        will be input for the neural network. This gives the opportunity to deal with different phoneme length.
        To create as many features for the specified phoneme the shift of size 1 is used.
        Finally, a cut chunk is saved to the storage.

        :param phoneme_features: phoneme feature in this particular case mel frequencies
        :param chunkLength: spectrogram chunk length which defines how many spectrums are considered
        around the middle one. The middle one defines the phoneme and speaker.
        :param phoneme: Phoneme string value
        :param speaker: Speaker string value
        :return:
        """

        totalNumberOfSpectrums = phoneme_features.shape[0]
        # The stepLength is 1 therefore the number of chunks is calculated as follows
        numChunks = totalNumberOfSpectrums - chunkLength + 1
        phoneme_index = config.TOTAL_TIMIT_PHONEME_LIST.index(phoneme)
        speaker_index = config.TOTAL_SPEAKERS_LIST.index(speaker)

        for i in range(numChunks):
            chunk = phoneme_features[i:i + chunkLength, :]
            # shape check
            if np.shape(chunk) != (config.FEATURES_CHUNK_LENGTH, config.NUM_MEL_FREQ_COMPONENTS):
                raise ValueError('The chunk has incorrect shape' + str(np.shape(chunk)) + ' where expected' + '('
                                 + str(config.FEATURES_CHUNK_LENGTH) + ',' + str(config.NUM_MEL_FREQ_COMPONENTS) + ')')
            if not isinstance(chunk[0][0], np.float32):
                raise ValueError('The chunk values has incorrect type: ' + str(type(chunk[0][0])) + ' where expected: '
                                 + str(np.float32))

            row = (chunk, phoneme_index, speaker_index)
            self.insert_row(row)
            self.currentNumberOfRows += 1
            if self.maximumNumberOfRows is not None and self.maximumNumberOfRows == self.currentNumberOfRows:
                print "Added: " + str(self.currentNumberOfRows) + " rows"
                sys.exit(0)

    # Reader
    def _read_one_example(self):
        """
        Reads one example (one row) from a storage.

        :return:
            features vector: a vector of features of size config.CHUNK_VECTOR_SIZE. Features are mfccs
            phoneme: phoneme
            speaker: speaker
        """
        _, serialized_example = self.reader.read(self.filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'phoneme': tf.FixedLenFeature([], tf.int64),
                'speaker': tf.FixedLenFeature([], tf.int64),
                'spectrum_raw': tf.FixedLenFeature([], tf.string),
            })

        features_raw = tf.decode_raw(features['spectrum_raw'], tf.float32)
        features_raw.set_shape([config.CHUNK_VECTOR_SIZE])
        phoneme = tf.cast(features['phoneme'], tf.int64)
        speaker = tf.cast(features['speaker'], tf.int64)

        return features_raw, phoneme, speaker

    def inputs(self, batch_size, shuffle=True):
        """
        Construct input for EVA evaluation using the Reader ops.

        :param batch_size: Number of examples per batch.
        :return:
            features_batch: Sound features. 4D tensor of [batch_size, config.CHUNK_SHAPE] size.
            phoneme_batch: A batch of phonemes. 1D tensor of [batch_size] size.
            speaker_batch: A batch of speakers. 1D tensor of [batch_size] size.
        """
        features, phoneme, speaker = self._read_one_example()
        # Set the shapes of tensors.
        features = tf.reshape(features, config.CHUNK_SHAPE)

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.1
        min_queue_examples = int(config.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

        # Generate a batch of images and labels by building up a queue of examples.
        return self._generate_image_and_label_batch(features, phoneme, speaker,
                                                    min_queue_examples, batch_size,
                                                    shuffle=shuffle)

    def _generate_image_and_label_batch(self, features, phoneme, speaker, min_queue_examples, batch_size, shuffle):
        """Construct a queued batch of features and labels.

        :param features: 4D Tensor of [batch_size, config.CHUNK_SHAPE] of type.float32.
        :param phoneme: 1D Tensor of type.int32 of phonemes
        :param speaker: 1D Tensor of type.int32 of speakers
        :param min_queue_examples: int32, minimum number of samples to retain
            in the queue that provides of batches of examples.
        :param batch_size: Number of examples per batch.
        :param shuffle: boolean indicating whether to use a shuffling queue.
        :return:
            features_batch: Sound features. 4D tensor of [batch_size, config.CHUNK_SHAPE] size.
            phoneme_batch: A batch of phonemes. 1D tensor of [batch_size] size.
            speaker_batch: A batch of speakers. 1D tensor of [batch_size] size.
        """

        # Create a queue that shuffles the examples, and then
        # read 'batch_size' features, phoneme and speaker from the example queue.
        if shuffle:
            features_batch, phoneme_batch, speaker_batch = tf.train.shuffle_batch(
                [features, phoneme, speaker],
                batch_size=batch_size,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=int(min_queue_examples * 0.1))
        else:
            features_batch, phoneme_batch, speaker_batch = tf.train.batch(
                [features, phoneme, speaker],
                batch_size=batch_size,
                capacity=min_queue_examples + 3 * batch_size)

        # Display the training images in the visualizer.
        # tf.summary.image('spectrograms', spectrograms)

        return features_batch, tf.reshape(phoneme_batch, [batch_size]), tf.reshape(speaker_batch, [batch_size])
