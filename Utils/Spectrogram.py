import numpy as np
from sound_utils import get_wav_info
import math as math


def InvertSpectrogram(spectrogram, window_size, window_step):
    """
    Invert a spectrogram using IFT (inverse fourier transform)

    :param spectrogram: Spectrogram values
    :param window_size: corresponds to a period N of IFT
    :param window_step: A step to the next window
    :return: a wav signal corresponding to the input spectrogram
    """
    _fftSize, _frameCount = spectrogram.shape
    waveSize = (_frameCount - 1) * window_step + window_size
    waveform = np.zeros(waveSize)
    # To sum up total windowing effect
    totalWindowingSum = np.zeros(waveSize)
    h = 0.54 - 0.46 * np.cos(2 * math.pi * np.arange(window_size) / (window_size - 1))
    fftB = int(math.floor(window_size / 2))
    fftE = int(fftB + window_size)
    for frameNumber in range(0, _frameCount):
        waveB = frameNumber * window_step
        waveE = waveB + window_size
        spectralSlice = spectrogram[:, frameNumber]
        newFrame = np.fft.ifft(spectralSlice)
        newFrame = np.real(np.fft.fftshift(newFrame))
        waveform[waveB:waveE] = waveform[waveB:waveE] + newFrame[fftB:fftE]
        totalWindowingSum[waveB:waveE] = totalWindowingSum[waveB:waveE] + h
    waveform = np.divide(np.real(waveform), totalWindowingSum)
    return waveform


class Spectrogram(object):
    """
    The class for a spectrogram and manipulation with it
    """

    def __init__(self, file_path, window_size, window_step):
        """
        Creates a spectrogram from a wav file

        :param file_path: a path to a file
        :param window_size: window size
        :param window_step: window step
        """
        self.window_size = window_size
        self.window_step = window_step
        self._wave_form, self._frame_rate = get_wav_info(file_path)
        self.spectrogram_values = self._calculate_spectrogram()
        self._averaged_spectrogram_values, self.frequencies, self.times = self._average_spectrogram_intensity()

        # Stores only real part of Fourier Transform
        self.real_spectrogram = np.real(self._averaged_spectrogram_values)

    def _calculate_spectrogram(self):
        """
        Calculates a spectrogram from wav file frames using Fourier transform.

        :return: A matrix which contains intensity both for positive and negative frequencies.
        """
        self._fftSize = 2 * self.window_size
        fftB = int(math.floor(self.window_size / 2))
        fftE = int(fftB + self.window_size)
        fftBuffer = np.zeros(self._fftSize)

        self._frameCount = int(math.floor((self._wave_form.size - self.window_size) / self.window_step) + 1)
        spectrum = np.zeros((self._fftSize, self._frameCount))
        spectrum = spectrum + 0j

        h = 0.54 - 0.46 * np.cos(2 * math.pi * np.arange(self.window_size) / (self.window_size - 1))
        for frameNumber in range(0, self._frameCount):
            waveB = frameNumber * self.window_step
            waveE = waveB + self.window_size
            fftBuffer = 0 * fftBuffer  # Make sure the entire buffer is empty
            fftBuffer[fftB:fftE] = np.multiply(self._wave_form[waveB:waveE], h)
            fftBuffer = np.fft.fftshift(fftBuffer)
            spectrum[:, frameNumber] = np.fft.fft(fftBuffer)

        return spectrum

    def _average_spectrogram_intensity(self):
        """
        The self.spectrogram_values matrix contains intensity both for positive and negative frequencies.

        :return: Average intensities for frequencies which is obtained by multiplying the positive and the negative and
        taking sqrt from the result of multiplication.
        """
        freqs = np.arange(self.window_size + 1) * self._frame_rate / self._fftSize;
        times = np.arange(self._frameCount + 1) * float(self.window_step) / self._frame_rate;
        posFreqs = self.spectrogram_values[0:self.window_size + 1, :];
        negFreqs = np.flipud(self.spectrogram_values[self.window_size:, :])

        powerSpec = posFreqs;
        freqRange = np.arange(1, self.window_size + 1);
        powerSpec[freqRange, :] = np.multiply(powerSpec[freqRange, :], negFreqs)

        magSpec = np.sqrt(powerSpec);
        return magSpec, freqs, times
