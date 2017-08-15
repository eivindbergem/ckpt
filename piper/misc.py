import os

def mkdirp(dirname):
    """Same as `mkdir -p <dirname>`"""

    try:
        os.makedirs(dirname)
    except OSError as err:
        if not (err.errno == 17 and os.path.isdir(dirname)):
            raise
