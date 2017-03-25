from __future__ import absolute_import

from Storage.TFStorage import TFStorage, TFStorageOpenOptions
import Utils.Eva_config_consts as config
import random
import numpy as np
import tensorflow as tf


class EvaTest(tf.test.TestCase):
    def _generate_row(self):
        phoneme = random.choice(config.TOTAL_TIMIT_PHONEME_LIST)
        speaker = "speaker" + str(random.randint(0, 9))
        spectrogram_chunk = np.random.rand(config.SPECTROGRAM_HEIGHT, config.SPECTROGRAM_CHUNK_LENGTH,
                                           config.SPECTROGRAM_DEPTH)
        row = (spectrogram_chunk, phoneme, speaker)
        return row

    def testInsertionAndRetrievingDataFromStorage(self):
        row = self._generate_row()

        filename = config.TEST_DATESET_FILE_PATH()

        # write a row
        with TFStorage(filename, TFStorageOpenOptions.WRITE) as storage:
            storage.insert_row(row)

        # read a row
        with self.test_session() as sess, TFStorage(filename, TFStorageOpenOptions.READ) as storage:
            spectrum, phoneme, speaker = storage._read_one_example()
            spectrum = tf.reshape(spectrum, config.CHUNK_SHAPE)
            # Initializing all variables
            init = tf.global_variables_initializer()
            sess.run(init)
            # Starting threads that run the input pipeline, filling the example queue
            # so that the dequeue to get the examples will succeed
            tf.train.start_queue_runners(sess=sess)

            retrieved_spectrogram, retrieved_phoneme, retrieved_speaker = sess.run([spectrum, phoneme, speaker])
            self.assertAllEqual(row[0], retrieved_spectrogram)
            self.assertEqual(row[1], retrieved_phoneme)
            self.assertEqual(row[2], retrieved_speaker)


if __name__ == '__main__':
    tf.test.main()
