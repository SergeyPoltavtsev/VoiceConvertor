import os


def reverse_folder(ROOT, extention):
    """
    Reverces a directory in depth and outputs all files with specified extention

    Inputs:
    - ROOT: Starting directory.
    - extention: Extention to search for.

    Returns: A list of paths to files with specified extention.
    """
    paths = []
    for dr in os.listdir(ROOT):
        next_file = ROOT + "/" + dr

        if os.path.isdir(next_file):
            ps = reverse_folder(next_file, extention)
            paths.extend(ps)
        elif os.path.isfile(next_file):
            if next_file.endswith(extention):
                paths.append(next_file)
    return paths


def clean_folder(ROOT):
    """
    Deletes all files in a specified folder. Does not 

    Inputs:
    - ROOT: Path to a directory
    """
    for a_file in os.listdir(folder):
        file_path = os.path.join(folder, a_file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def get_speaker_name(file_path, position=2):
    """
    Gets speaker name from a file path. Because the only way to get a speaker name is from foldername 
    this solution is very specific and not scalable at all. 
    
    Inputs:
    - file_path: path to a file
    - position: specifies the position of a speaker name in a path starting from the end.
    """
    file_parts = file_path.split("/")
    speaker = file_parts[-position]
    return speaker
