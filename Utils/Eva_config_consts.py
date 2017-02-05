"""
For creating a spectrograms the window of size WINDOW_SIZE is used.
One spectrum is created out of this window. Afterwards, the window is shifted
by FRAME_STEP to create the next spectrum.
"""
WINDOW_SIZE = 256
WINDOW_STEP = 128

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
CHUNK_SIZE = (SPECTROGRAM_HEIGHT, SPECTROGRAM_CHUNK_LENGTH, SPECTROGRAM_DEPTH)

####### PATHES #########

"""
Dataset file
"""
DATESET_FILE = "TimitStore.tfrecords"

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
