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
from Utils.MFCC import *


def create_dataset(path_to_TIMIT, storage_path, number_of_examples=None):
    nistReader = NistReader()
    mel_filter, mel_inversion_filter = create_mel_filter(fft_size=config.WINDOW_SIZE,
                                                         n_freq_components=config.NUM_MEL_FREQ_COMPONENTS,
                                                         start_freq=config.START_FREQ,
                                                         end_freq=config.END_FREQ,
                                                         samplerate=config.FRAME_RATE)

    # create a list of paths to WAV files inside path_to_TIMIT
    paths = folder_utils.reverse_folder(path_to_TIMIT, ".WAV")

    with TFStorage(storage_path, TFStorageOpenOptions.WRITE, number_of_examples) as storage:
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
                    if end - start < (config.WINDOW_SIZE + 2 * config.PHONEME_OFFSET):
                        continue
                else:
                    # in case is a phone lasts less than window_size the window size is used.
                    if int(phoneme[1]) - int(phoneme[0]) < config.WINDOW_SIZE:
                        start = int(phoneme[0]) - config.PHONEME_OFFSET
                        end = int(phoneme[0]) + config.WINDOW_SIZE + config.PHONEME_OFFSET
                    else:
                        start = int(phoneme[0]) - config.PHONEME_OFFSET
                        end = int(phoneme[1]) + config.PHONEME_OFFSET

                phone_file = sound_utils.cutPhonemeChunk(wav_file, config.TEMP_PHONEME_FOLDER_PATH, start, end,
                                                         phoneme[2])
                phone_wave_form, frame_rate = sound_utils.get_wav_info(phone_file)

                # create spectrogram of a phone chunk
                phoneme_spectrogram = pretty_spectrogram(phone_wave_form, fft_size=config.WINDOW_SIZE,
                                                         step_size=config.WINDOW_STEP, log=True,
                                                         thresh=config.SPEC_THRESH)
                # create mels out of spectrogram
                phone_mfcc = make_mel(phoneme_spectrogram, mel_filter, shorten_factor=1)
                storage.cut_phoneme_into_chunks_and_save(phone_mfcc, config.FEATURES_CHUNK_LENGTH,
                                                         phoneme[2], speaker)

        print "Added: " + str(storage.currentNumberOfRows) + " rows"


if __name__ == '__main__':
    # full data set
    # path_to_TIMIT_subset = config.PATH_TO_TIMIT_TRAIN
    train = True
    if train:
        # path to one dr1 folder
        # path_to_TIMIT_subset = os.path.join(config.PATH_TO_TIMIT_TRAIN, "dr2")
        path_to_TIMIT_subset = config.PATH_TO_TIMIT_TRAIN
        storage_path = config.DATESET_FILE_PATH()
        number_of_examples = config.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        # path_to_TIMIT_subset = os.path.join(config.PATH_TO_TIMIT_TEST, "dr1")
        path_to_TIMIT_subset = config.PATH_TO_TIMIT_TEST
        storage_path = config.TEST_DATESET_FILE_PATH()
        number_of_examples = config.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    create_dataset(path_to_TIMIT_subset, storage_path, number_of_examples)
