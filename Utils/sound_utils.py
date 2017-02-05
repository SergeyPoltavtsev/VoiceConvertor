import os
import wave
import pylab
import struct
from folder_utils import get_speaker_name

name_separator = "_"
name_counter = 0


def cutIntoChunks(file_path, output_folder, chunk_duration=0.02, chunk_step=0.02, chunk_extension=".wav"):
    """
    Divides a sound file into chunks using time intervals

    :param file_path: a path to a wav file
    :param output_folder: output folder where chunks will be saved
    :param chunk_duration: chunk duration where 0.02 = 20ms
    :param chunk_step: the step between chunks where default 0.02 = 20ms
    :param chunk_extension: chunk extension
    """

    wav = wave.open(file_path, 'r')

    frame_rate = wav.getframerate()
    number_channels = wav.getnchannels()
    sample_width = wav.getsampwidth()
    numSamplesPerChunk = int(chunk_duration * frame_rate)
    numSamplesPerStep = int(chunk_step * frame_rate)
    totalNumSamples = wav.getnframes()
    chunk_counter = 0
    starting_location = 0

    while starting_location + numSamplesPerChunk <= totalNumSamples:
        ending_location = min(starting_location + numSamplesPerChunk - 1, totalNumSamples)
        wav.setpos(starting_location)
        chunk_file_name = str(chunk_counter) + chunk_extension
        output_chunk_path = os.path.join(output_folder, chunk_file_name)

        _cutOneChunk(starting_location, ending_location, wav, output_chunk_path, frame_rate, number_channels,
                     sample_width)

        starting_location += numSamplesPerStep
        chunk_counter += 1

    wav.close()


def cutPhonemeChunk(file_path, output_folder, starting_frame, ending_frame, phoneme):
    """
    Cuts a chunk from a wav file staring and ending from specified frames and saves it

    :param file_path: a path to a wav file
    :param output_folder: output folder where chunks will be saved
    :param starting_frame: starting frame
    :param ending_frame: ending frame
    :param phoneme: A phoneme which is associated with the cut chunk
    :return: A path to cut and saved chunk
    """

    global name_counter
    file_name = os.path.basename(file_path)
    name, extension = os.path.splitext(file_name)
    speaker = get_speaker_name(file_path)

    speaker_folder = os.path.join(output_folder, speaker)
    if not os.path.exists(speaker_folder):
        os.makedirs(speaker_folder)

    name_candidate = phoneme + name_separator + str(starting_frame) + name_separator + str(
        ending_frame) + name_separator + name
    output_file = os.path.join(speaker_folder, name_candidate + extension)
    if os.path.isfile(output_file):
        # file with name name_candidate already exists therefore the name should be changed
        name_candidate = name_candidate + name_separator + str(name_counter) + extension
        output_file = os.path.join(speaker_folder, name_candidate)
        name_counter += 1

    wav = wave.open(file_path, 'r')

    frame_rate = wav.getframerate()
    number_channels = wav.getnchannels()
    sample_width = wav.getsampwidth()
    totalNumSamples = wav.getnframes()

    wav.setpos(starting_frame)
    _cutOneChunk(starting_frame, ending_frame, wav, output_file, frame_rate, number_channels, sample_width)

    wav.close()
    return output_file


def _cutOneChunk(starting_frame, ending_frame, opened_wav_file, output_path, frame_rate, number_channels, sample_width):
    """
    Cuts one chunk from a wave file and saves it to a specified file with specified wave parameters

    :param starting_frame: starting frame
    :param ending_frame: ending frame
    :param opened_wav_file: a wave file
    :param output_path: a path where the chunk will be saved
    :param frame_rate: frame rate
    :param number_channels: number of channels
    :param sample_width: sample width
    """
    chunk_frames = opened_wav_file.readframes(ending_frame - starting_frame)

    chunkAudio = wave.open(output_path, 'w')
    chunkAudio.setnchannels(number_channels)
    chunkAudio.setsampwidth(sample_width)
    chunkAudio.setframerate(frame_rate)
    chunkAudio.writeframes(chunk_frames)
    chunkAudio.close()


def get_wav_info(file_path):
    """
    Reads and returns a wav file frames and corresponding frame_rate

    :param file_path: A path to a wav file
    :return: a wav file frames and corresponding frame_rate
    """

    wav = wave.open(file_path, 'r')
    frames = wav.readframes(-1)
    frames = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()

    byteDepth = wav.getsampwidth()
    bitDepth = byteDepth * 8

    max_nb_bit = float(2 ** (bitDepth - 1))
    frames = frames / (max_nb_bit + 1.0)

    wav.close()
    return frames, frame_rate


def save_to_wav(file_path, waveform, frame_rate=16000, byte_depth=2, nchannels=1, compname="not compressed",
                comptype="NONE"):
    """
    Writes an array to wav file

    :param file_path: A path to a file
    :param waveform:
    :param frame_rate:
    :param byte_depth:
    :param nchannels:
    :param compname:
    :param comptype:
    """

    wav_file = wave.open(file_path, "w")
    wav_file.setparams((nchannels, byte_depth, frame_rate, waveform.size,
                        comptype, compname))
    bit_depth = byte_depth * 8
    max_nb_bit = float(2 ** (bit_depth - 1))
    waveform = waveform * (max_nb_bit + 1.0)

    for s in waveform:
        wav_file.writeframes(struct.pack('h', int(s)))

    wav_file.close()
