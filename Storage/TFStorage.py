import tensorflow as tf
import Utils.Eva_config_consts as config
from enum import Enum


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class TFStorageOpenOptions(Enum):
    READ = 1
    WRITE = 2


class TFStorageLabelOption(Enum):
    PHONEME = 1
    SPEAKER = 2


class TFStorage(object):
    """
    The Storage based on TFRecords
    """

    def __init__(self, path, openOption):
        """
        Initializes the storage

        Inputs:
        - path: a path to a storage file (should have .tfrecords extension)
        - openOption: Specifies if the storage object should be created for reading or writing.
        """
        if not tf.gfile.Exists(path):
            raise ValueError('Failed to find file: ' + path)

        self.path = path
        self.openOption = openOption

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
        spectrum = item[0]
        phoneme = item[1]
        speaker = item[2]

        spectrum_raw = spectrum.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(config.SPECTROGRAM_HEIGHT),
            'width': _int64_feature(config.SPECTROGRAM_CHUNK_LENGTH),
            'depth': _int64_feature(config.SPECTROGRAM_DEPTH),
            'phoneme': _bytes_feature(phoneme),
            'speaker': _bytes_feature(speaker),
            'spectrum_raw': _bytes_feature(spectrum_raw)}))

        self.writer.write(example.SerializeToString())

    # Reader
    def _read_one_example(self):
        """
        Reads one example (one row) from a storage.

        :return: spectrum vector, phoneme and speaker
        """
        _, serialized_example = self.reader.read(self.filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'phoneme': tf.FixedLenFeature([], tf.string),
                'speaker': tf.FixedLenFeature([], tf.string),
                'spectrum_raw': tf.FixedLenFeature([], tf.string),
            })

        # Convert from a spectrum vector of size config.CHUNK_VECTOR_SIZE to
        # a tensor of shape config.CHUNK_SHAPE
        spectrum = tf.decode_raw(features['spectrum_raw'], tf.float64)
        spectrum.set_shape([config.CHUNK_VECTOR_SIZE])

        # Convert phoneme and speaker bytes(uint8) to string.
        phoneme = tf.cast(features['phoneme'], tf.string)
        speaker = tf.cast(features['speaker'], tf.string)
        tf.reshape(spectrum, config.CHUNK_SHAPE)

        return spectrum, phoneme, speaker

    def inputs(self, labelOption, batch_size):
        """
        Construct input for EVA evaluation using the Reader ops.

        :param labelOption: Specifies which labels to use either phoneme or speaker
        :param batch_size: Number of examples per batch.
        :return:
            spectrograms: Chunk spectrograms. 4D tensor of [batch_size, SPECTROGRAM_HEIGHT,
                SPECTROGRAM_CHUNK_LENGTH, SPECTROGRAM_DEPTH)] size.
            labels: Labels. 1D tensor of [batch_size] size.
        """
        spectrogram, phoneme, speaker = self._read_one_example()
        # Set the shapes of tensors.
        spectrogram = tf.reshape(spectrogram, config.CHUNK_SHAPE)
        # phoneme.set_shape([1])
        # speaker.set_shape([1])

        if (labelOption == TFStorageLabelOption.PHONEME):
            label = phoneme
        else:
            label = speaker

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(config.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                 min_fraction_of_examples_in_queue)

        # Generate a batch of images and labels by building up a queue of examples.
        return self._generate_image_and_label_batch(spectrogram, label,
                                                    min_queue_examples, batch_size,
                                                    shuffle=False)

    def _generate_image_and_label_batch(self, spectrogram, label, min_queue_examples, batch_size, shuffle):
        """Construct a queued batch of spectrograms and labels.

        :param spectrogram: 3-D Tensor of config.CHUNK_SHAPE of type.float32.
        :param label: 1-D Tensor of type.int32 either of phonemes or speakers
        :param min_queue_examples: int32, minimum number of samples to retain
            in the queue that provides of batches of examples.
        :param batch_size: Number of examples per batch.
        :param shuffle: boolean indicating whether to use a shuffling queue.
        :return:
            spectrograms: Chunk spectrograms. 4D tensor of [batch_size, SPECTROGRAM_HEIGHT,
                SPECTROGRAM_CHUNK_LENGTH, SPECTROGRAM_DEPTH)] size.
            labels: Labels. 1D tensor of [batch_size] size.
        """

        # Create a queue that shuffles the examples, and then
        # read 'batch_size' spectrograms + labels from the example queue.
        num_preprocess_threads = 16
        if shuffle:
            spectrograms, label_batch = tf.train.shuffle_batch(
                [spectrogram, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples)
        else:
            spectrograms, label_batch = tf.train.batch(
                [spectrogram, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size)

        # Display the training images in the visualizer.
        # tf.contrib.deprecated.image_summary('spectrograms', spectrograms)

        return spectrograms, tf.reshape(label_batch, [batch_size])