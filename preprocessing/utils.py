import errno
import os


def make_directories(directory_path):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
