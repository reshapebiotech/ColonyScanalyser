﻿# Create any folder that is needed in a filepath
def create_folderpath(filepath):
    import os
    # Exception handling is required in Python2 to protect against race conditions
    try: 
        os.makedirs(filepath)
    except OSError:
        if not os.path.isdir(filepath):
            raise
    # In Python3 this can be reduced to
    # os.makedirs(path, exist_ok=True)


#Separates the folderpath and filename from a filepath
#Return the filename by default
def separate_filepath(filepath, return_folderpath = False):
    import os
    (folderpath, filename) = os.path.split(os.path.abspath(filepath))
    if return_folderpath:
        return folderpath
    else:
        return filename


def get_subfoldername(data_folder, row, col):
    import os
    return data_folder+'_'.join(['row', str(row), 'col', str(col)])+os.path.sep


# Checks whether a file exists and contains data
def file_exists(filepath):
    import os
    if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
        return True


def mkdir_plates(data_folder, lattice):
    import os
    '''Make subfolders for single plates'''
    for row in xrange(lattice[0]):
        for col in xrange(lattice[1]):
            subfn = get_subfoldername(data_folder, row + 1, col + 1)
            if not os.path.isdir(subfn):
                os.mkdir(subfn)
