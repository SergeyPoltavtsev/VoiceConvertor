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
