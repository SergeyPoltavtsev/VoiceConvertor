import os
import wave


class NistReader(object):
    """
    A class which helps to read Nist files. Convert them to Wav and cut into chunks.
    More info about Nist files can be found here: http://www.ee.columbia.edu/ln/LabROSA/doc/HTKBook21/node64.html
    """

    def __init__(self, sample_rate=16000, sample_n_bytes=2, n_channels=1, nist_extension=".WAV"):
        """
        Initialize the reader.

        :param sample_rate: Sample rate in Hz. A number of samples in one second
        :param sample_n_bytes: A number of bytes per each sample
        :param n_channels: Number of channels (1 = mono, 2 = stereo)
        :param nist_extension: A Nist file extension
        """

        self.sample_rate = sample_rate
        self.sample_n_bytes = sample_n_bytes
        self.n_channels = n_channels
        self.wav_extension = ".wav"
        self.nist_extension = nist_extension

    def __separateNistOnHeaderAndData(self, input_file):
        """
        Separates Nist file header and raw data.
        The header begins with a label of the form NISTxx where xx is a version
        code followed by the number of bytes in the header.
        The header also contain the following information:

        sample_rate - sample rate in Hz
        sample_n_bytes - number of bytes in each sample
        sample_count - number of samples in file
        sample_byte_format - byte order
        sample_coding - speech coding eg pcm,  tex2html_wrap_inline19966 law, shortpack
        channels_interleaved - for 2 channel data only

        :param input_file: a path to an input file
        :return: separated Nist header and data (content)
        """

        # TODO: parce header to get all values
        with open(input_file, 'rb') as nistFile:
            Nist_version = nistFile.readline()
            HeaderBytes = int(nistFile.readline())
            # closing file. Now we know header size

        with open(input_file, 'rb') as nistFile:
            header = nistFile.read(HeaderBytes)
            data = nistFile.read()

        return header, data

    def Nist2Wav(self, input_file, output_folder):
        """
        Converts a Nist file to a wav file. A nist file has an extension of type ".WAV"
        The converter uses predefined sound parameters. The proper way to do it is to read them from the header

        :param input_file: a path to a Nist file
        :param output_folder: a path to output folder where the converted file will be saved.
        The output file will have the same name as the input file
        :return: the path to the wav file
        """

        output_folder = output_folder + "/"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        file_name = os.path.basename(input_file)

        # the converted file will be saved with the same name but with the different extension
        wav_name = file_name.replace(self.nist_extension, self.wav_extension)
        wav_file = output_folder + wav_name
        header, data = self.__separateNistOnHeaderAndData(input_file)
        wavfile = wave.open(wav_file, 'wb')

        # nchannels, sampwidth, framerate, nframes, comptype, compname
        wavfile.setparams((self.n_channels, self.sample_n_bytes, self.sample_rate, 0, 'NONE', 'NONE'))
        wavfile.writeframes(data)
        wavfile.close()
        # audiotools.open(input_file).convert(wav_file, audiotools.WaveAudio)
        return wav_file
