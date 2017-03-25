import os

# storage
from Storage.TFStorage import *

# Config
import Utils.Eva_config_consts as config

# Utilities
import Utils.folder_utils as folder_utils
import Utils.TIMIT_utils as TIMIT_utils

# Sound utils
from Utils.nist_reader import NistReader
import Utils.sound_utils as sound_utils

# Spectrograms
from Utils.SpectrogramFactory import SpectrogramFactory

import numpy as np

GLOBAL_EXAMPLES_COUNTER = 0


def CutPhonemeIntoChunksAndSave(storage, phoneme_spectrums, chunkLength, phoneme, speaker):
    """
    Accepts a spectrogram of arbitrary size of one concrete phoneme. Cuts chunks of size chunkLength which
    will be input for the neural network. This gives the opportunity to deal with different phoneme length.
    To create as many and as variable chunk spectrograms for the specified phoneme the shift of size 1 is used.
    Finally, a cut chunk is saved to the storage.

    :param storage: a TFStorage storage
    :param phoneme_spectrums: Cut phoneme spectrogram
    :param chunkLength: spectrogram chunk length which defines how many spectrums are considered
    around the middle one. The middle one defines the phoneme and speaker.
    :param phoneme: Phoneme string value
    :param speaker: Speaker string value
    :return:
    """

    global GLOBAL_EXAMPLES_COUNTER
    totalNumberOfSpectrums = phoneme_spectrums.shape[1]
    # The stepLength is 1 therefore the number of chunks is calculated as follows
    numChunks = totalNumberOfSpectrums - chunkLength + 1
    phoneme_index = config.TOTAL_TIMIT_PHONEME_LIST.index(phoneme)

    for i in range(numChunks):
        chunk = phoneme_spectrums[:, i:i + chunkLength]
        real = np.real(chunk)
        imag = np.imag(chunk)
        phone_item = np.stack((real, imag), axis=-1)
        row = (phone_item, phoneme_index, speaker)
        storage.insert_row(row)
        GLOBAL_EXAMPLES_COUNTER += 1


def create_dataset(path_to_TIMIT, storage_path, number_of_examples):
    global GLOBAL_EXAMPLES_COUNTER
    nistReader = NistReader()
    spectrogramFactory = SpectrogramFactory(window_size=config.WINDOW_SIZE, window_step=config.WINDOW_STEP)

    # create a list of paths to WAV files inside path_to_TIMIT
    paths = folder_utils.reverse_folder(path_to_TIMIT, ".WAV")

    with TFStorage(storage_path, TFStorageOpenOptions.WRITE) as storage:
        for path in paths:
            print path
            phonemes = TIMIT_utils.parse_phoneme_file(path)
            speaker = folder_utils.get_speaker_name(path)

            # temp_speaker_folder is used for storing converted to wav audio files.
            temp_speaker_folder = os.path.join(config.TEMP_DATA_FOLDER_PATH, speaker)
            if not os.path.exists(temp_speaker_folder):
                os.makedirs(temp_speaker_folder)

            # convert a nist file to a wav file
            wav_file = nistReader.Nist2Wav(path, temp_speaker_folder)
            for i in range(len(phonemes)):
                phoneme = phonemes[i]
                # Cutting one phoneme
                if i == 0 or i == len(phonemes):
                    start = int(phoneme[0])
                    end = int(phoneme[1])
                else:
                    start = int(phoneme[0]) - config.PHONEME_OFFSET
                    end = int(phoneme[1]) + config.PHONEME_OFFSET

                phone_file = sound_utils.cutPhonemeChunk(wav_file, config.TEMP_PHONEME_FOLDER_PATH, start, end,
                                                         phoneme[2])
                phoneme_spectrogram = spectrogramFactory.create_spectrogram(phone_file)
                CutPhonemeIntoChunksAndSave(storage, phoneme_spectrogram.spectrogram_values,
                                            config.SPECTROGRAM_CHUNK_LENGTH, phoneme[2], speaker)
                if number_of_examples == GLOBAL_EXAMPLES_COUNTER:
                    print "Created: " + str(GLOBAL_EXAMPLES_COUNTER) + " examples"
                    return


if __name__ == '__main__':
    # full data set
    # path_to_TIMIT_subset = config.PATH_TO_TIMIT_TRAIN

    # path to one dr1 folder
    path_to_TIMIT_subset = os.path.join(config.PATH_TO_TIMIT_TRAIN, "dr1")
    storage_path = config.DATESET_FILE_PATH()
    number_of_examples = config.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    create_dataset(path_to_TIMIT_subset, storage_path, number_of_examples)
    print GLOBAL_EXAMPLES_COUNTER
