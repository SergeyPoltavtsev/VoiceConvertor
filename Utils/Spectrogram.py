import numpy as np
from sound_utils import get_wav_info
import math as math
from PIL import Image

class Spectrogram(object):
    """
    The class for spectrogram and manipulation with it
    """
    
    def __init__(self, file_path, window_size, frame_step):
        """
        Creates a spectrogram from a wav file

        Inputs:
        - file_path: a path to a file.
        - window_size: to create one power spectrum the window of size window_size. 
        One power spectrum corresponds to one vertical line in spectrogram. 
        - frame_step: the size of shift for creating the next power spectrum.
        """
        #self.spectrogram = spectrogram
        self.window_size = window_size
        self.frame_step = frame_step
        self._wave_form, self._frame_rate = get_wav_info(file_path)
        self.spectrogram_values = self._calculate_spectrogram()
        self._averaged_spectrogram_values, self.frequencies, self.times = self._average_spectrogram_intensity()
        
        # Stores only real part of Fourier Transform
        self.real_spectrogram = np.real(self._averaged_spectrogram_values)
        
        # Prepare to show a spectrogram as an image
        self._normalize()
        self._to_RGB()
    
    def _calculate_spectrogram(self):
        """
        Calculates a spectrogram from wav file frames using Fourier transform. The output matrix contains 
        intensity both for positive and negative frequencies.
        
        Inputs:
        - wave_form: wav file frames
        """
        self._fftSize = 2*self.window_size
        fftB = int(math.floor(self.window_size/2))
        fftE = int(fftB + self.window_size)
        fftBuffer = np.zeros(self._fftSize)

        self._frameCount = int(math.floor((self._wave_form.size - self.window_size)/self.frame_step) + 1)
        spectrum = np.zeros((self._fftSize, self._frameCount));
        spectrum = spectrum + 0j

        h = 0.54 - 0.46*np.cos(2*math.pi* np.arange(self.window_size)/(self.window_size-1))
        for frameNumber in range(0,self._frameCount):
            waveB = (frameNumber)*self.frame_step
            waveE = waveB + self.window_size
            fftBuffer = 0*fftBuffer #Make sure the entire buffer is empty
            fftBuffer[fftB:fftE] = np.multiply(self._wave_form[waveB:waveE], h)
            fftBuffer = np.fft.fftshift(fftBuffer)
            spectrum[:,frameNumber] = np.fft.fft(fftBuffer)

        return spectrum
    
    def _average_spectrogram_intensity(self):
        """
        The output matrix contains intensity both for positive and negative frequencies. 
        Thefore, to get intesities for frequencies the positive part is multiplied by the negative and sqrt is taken.
        """
        #fftSize, frameCount = spec.shape
        #windowSize = int(math.floor(self._fftSize/2))
        freqs = np.arange(self.window_size+1)*self._frame_rate/self._fftSize;
        times = np.arange(self._frameCount+1) * float(self.frame_step)/self._frame_rate;
        posFreqs = self.spectrogram_values[0:self.window_size+1, :];
        negFreqs = np.flipud(self.spectrogram_values[self.window_size:,:])

        powerSpec = posFreqs;
        freqRange = np.arange(1,self.window_size+1);
        powerSpec[freqRange,:] = np.multiply(powerSpec[freqRange,:], negFreqs)

        magSpec = np.sqrt(powerSpec);
        return magSpec, freqs, times
    
    def _normalize(self):
        """
        Normalizes the spectrogram values to the range [0,1]
        """
        fliped_real = np.flipud(self.real_spectrogram)
        self._normalized_spectrogram_values = ( fliped_real - fliped_real.min())/( fliped_real.max() - fliped_real.min())
        self._min_spectrogram_value = fliped_real.min()
        self._max_spectrogram_value = fliped_real.max()
        
    def _to_RGB(self):
        """
        Creates rgb spectrogram value from the normalized.
        """
        gray = (self._normalized_spectrogram_values*255.).astype('uint8')
        self.rgb_spectrogram_values = np.repeat(gray[:, :, np.newaxis], 3, axis=2)
        
    def get_spectrogram_image(self):
        """
        Creates spectrogram image
        """
        img = Image.fromarray(self.rgb_spectrogram_values)
        return img
    
    def save_spectrogram_as_image(self, output_img_path):
        """
        Saves the spectrogram as an image to the specified file
        
        Inputs:
        - output_img_path: output file path
        """
        this.get_spectrogram_image()
        img.save(output_img_path)
        
        