"""
For creating a spectrograms the window of size WINDOW_SIZE is used.
One spectrum is created out of this window. Afterwards, the window is shifted
by FRAME_STEP to create the next spectrum.
NOTE: 75% of overlap has shown the best performance in reconstruction.
"""
WINDOW_SIZE = 256
WINDOW_STEP = 64

"""
Full list of TIMIT phonemes
"""
TOTAL_TIMIT_PHONEME_LIST = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw', 'ux',
                            'er', 'ax', 'ix', 'axr', 'ax-h', 'jh', 'ch', 'b', 'd', 'g', 'p', 't', 'k', 'dx', 's', 'sh',
                            'z', 'zh', 'f', 'th', 'v', 'dh', 'm', 'n', 'ng', 'em', 'nx', 'en', 'eng', 'l', 'r', 'w',
                            'y', 'hh', 'hv', 'el', 'bcl', 'dcl', 'gcl', 'pcl', 'tcl', 'kcl', 'q', 'pau', 'epi', 'h#']

"""
Number of phoneme classes
"""
NUM_PHONOME_CLASSES = len(TOTAL_TIMIT_PHONEME_LIST)

"""
For positive and negative frequences
"""
SPECTROGRAM_HEIGHT = 2 * WINDOW_SIZE

"""
Every chunk contains CHUNK_LENGTH spectrums
"""
SPECTROGRAM_CHUNK_LENGTH = 11

"""
Because we want to take into account the phoneme surroundings we take
5 spectrums from the leaf and 5 from the right
"""
PHONEME_OFFSET = 5 * WINDOW_STEP

"""
For real and img values
"""
SPECTROGRAM_DEPTH = 2

"""
Spectrogram chunks are stored as vectors
"""
CHUNK_VECTOR_SIZE = SPECTROGRAM_HEIGHT * SPECTROGRAM_CHUNK_LENGTH * SPECTROGRAM_DEPTH

"""
The size of a chunk
"""
CHUNK_SHAPE = (SPECTROGRAM_HEIGHT, SPECTROGRAM_CHUNK_LENGTH, SPECTROGRAM_DEPTH)
TF_CHUNK_SHAPE = [SPECTROGRAM_HEIGHT, SPECTROGRAM_CHUNK_LENGTH, SPECTROGRAM_DEPTH]

####### TRAIN PARAMETERS #########
"""
Batch size
"""
BATCH_SIZE = 128
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

####### PATHES #########
"""
Path to the dataset file
"""
#DATESET_FILE_PATH = "TimitStore.tfrecords"
DATESET_FILE_PATH = "/Volumes/BOOTCAMP/EVA/TimitStore.tfrecords"

"""
Path to train TIMIT date set
"""
PATH_TO_TIMIT_TRAIN = "Data/TIMIT/timit/train"

"""
Path to test TIMIT date set
"""
PATH_TO_TIMIT_TEST = "Data/TIMIT/timit/test"

"""
Path to the temp data folder
"""
TEMP_DATA_FOLDER_PATH = "/tmp/data"

"""
Path to the temp phonemes folder
"""
TEMP_PHONEME_FOLDER_PATH = "/tmp/phonemes"
