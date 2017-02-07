
TOTAL_TIMIT_PHONEME_LIST = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw', 'ux',
                            'er', 'ax', 'ix', 'axr', 'ax-h', 'jh', 'ch', 'b', 'd', 'g', 'p', 't', 'k', 'dx', 's', 'sh',
                            'z', 'zh', 'f', 'th', 'v', 'dh', 'm', 'n', 'ng', 'em', 'nx', 'en', 'eng', 'l', 'r', 'w',
                            'y', 'hh', 'hv', 'el', 'bcl', 'dcl', 'gcl', 'pcl', 'tcl', 'kcl', 'q', 'pau', 'epi', 'h#']

def parse_phoneme_file(path):
    """
    Parses a *.PHN file which contains phonemes with corresponding phoneme frame stamps

    :param path: A file path
    :return: A list of corteges of pattern (starting_frame, ending_frame, phoneme)
    """

    # convert extension to open phonemes file
    if not path.endswith(".PHN"):
        path = path.split('.')[0] + ".PHN"

    phonemes = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('\n')[0]
            line_parts = line.split(' ')
            phoneme = (line_parts[0], line_parts[1], line_parts[2])
            phonemes.append(phoneme)
    return phonemes