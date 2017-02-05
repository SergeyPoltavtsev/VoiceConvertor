from Spectrogram import Spectrogram


class SpectrogramFactory(object):
    """
    The class for creating a spectrogram from a wav file.
    """

    def __init__(self, window_size, window_step):
        """
        Initialize the spectrogram factory for spectrogram creation

        :param window_size: To create one power spectrum the window of size window_size is used.
        One power spectrum corresponds to one vertical line in a spectrogram
        :param window_step: The size of shift for creating the next power spectrum
        """

        self.window_size = window_size
        self.window_step = window_step

    def create_spectrogram(self, file_path):
        """
        Creates a spectrogram from a wav file

        :param file_path: a path to a wav file
        :return: a spectrogram object
        """

        spectrogram = Spectrogram(file_path, self.window_size, self.window_step)
        return spectrogram
