import tensorflow as tf
import Utils.Eva_config_consts as config


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class TFStorage(object):
    """
    The Storage based on TFRecords
    """

    def __init__(self, path):
        """
        Initializes the storage

        Inputs:
        - path: a path to a storage file (should have .tfrecords extention)
        """
        self.path = path

    def CreateWriter(self):
        self.writer = tf.python_io.TFRecordWriter(self.path)

    def InsertRow(self, item):
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

    def StopWriting(self):
        self.writer.close()

    def NextBatch(self):
        raise NotImplementedError("Should have implemented this")
