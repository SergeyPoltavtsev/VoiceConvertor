import os


def reverse_folder(folder, extension):
    """
    Reverses a folder in depth and outputs all files with specified extension

    :param folder: Starting directory
    :param extension: Extension to search for.
    :return: A list of paths to files with specified extension.
    """

    paths = []
    for dr in os.listdir(folder):
        next_file = folder + "/" + dr

        if os.path.isdir(next_file):
            ps = reverse_folder(next_file, extension)
            paths.extend(ps)
        elif os.path.isfile(next_file):
            if next_file.endswith(extension):
                paths.append(next_file)
    return paths


def clean_folder(folder):
    """
    Deletes all files in a specified folder.

    :param folder: Path to the folder
    :return:
    """

    for a_file in os.listdir(folder):
        file_path = os.path.join(folder, a_file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(e)


def get_speaker_name(file_path, position=2):
    """
    Gets speaker name from a file path. The only way to get a speaker name is from folder name, therefore
    this solution is very specific and not scalable at all.

    :param file_path: Path to a file
    :param position: Specifies the position of a speaker name in a path starting from the end
    :return: A speaker name
    """

    file_parts = file_path.split("/")
    speaker = file_parts[-position]
    return speaker
