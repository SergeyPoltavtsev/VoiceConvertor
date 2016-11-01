import os
import numpy as np
import wave
from Spectrogram import Spectrogram

class SpectrogramFactory(object):
    """
    The class for creating spectrograms from a wav file.
    """
    
    def __init__(self, window_size = 256, frame_step = 64):
        """
        Initialize the reader. The default values for window size and frame step are 256 and 64 respectively. Therefore,
        there is a 75% overlap between two coherent power spectrums. The 75% is taking for the purpose of reconstructing 
        a wave signal from spectrogram.

        Inputs:
        - window_size: To create one power spectrum the window of size window_size. 
        One power spectrum corresponds to one vertical line in spectrogram. 
        - frame_step: The size of shift for creating the next power spectrum.
        """
        self.window_size = window_size
        self.frame_step = frame_step

    # this is not needed anymore because the network operates on spectrograms directly instead of spectrogram images.
    # def spectogram_to_image(file_path, out_img_file):
    #     waveform, frame_rate = get_wav_info(file_path)
    #     spect = CreateSpectogram(waveform)
    #     magSpec, freqs, times = SpectrogramForDisplay(spect, frame_rate, frameStep)
    #     A = np.real(magSpec)
    #     normalized, minEl, maxEl = normalize(A)
    #     normedRGB = toRGB(normalized)
    #     img = Image.fromarray(normedRGB)
    #     img.save(out_img_file)
    #     return normedRGB, minEl, maxEl

    def create_spectrogram(self, file_path):
        """
        Creates a spectrogram from a wav file

        Inputs:
        - file_path: a path to a file.
        """

        spectrogram = Spectrogram(file_path, self.window_size, self.frame_step)

        return spectrogram