import os

piper_path = ".piper"

def set_piper_path(path):
    global piper_path

    piper_path = path

def get_piper_path():
    return piper_path

def mkdirp(dirname):
    """Same as `mkdir -p <dirname>`"""

    try:
        os.makedirs(dirname)
    except OSError as err:
        if not (err.errno == 17 and os.path.isdir(dirname)):
            raise

def add_defaults(config, defaults):
    return {key: (value
                  if key not in config
                  else config[key])
            for key, value in defaults.items()}
