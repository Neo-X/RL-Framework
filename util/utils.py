

import subprocess

def get_git_revision_hash():
    out = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    return out.decode('UTF-8')

def get_git_revision_short_hash():
    out = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
    return out.decode('UTF-8')

def checkSetting(settings, key, value):
    
    if (checkSettingExists(settings, key) and
        (settings[key] == value)):
        return True
    else:
        return False
        
def checkSettingExists(settings, key):
    
    if (key in settings):
        return True
    else:
        return False
    
def rlPrint(settings=None, level="train", text=""):
    if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"][level]):
        print (text)
        
def load_keras_model(filename, custom_objects={}):
    from keras.models import load_model
    import keras_layer_normalization
    custom_objects["LayerNormalization"] = keras_layer_normalization.LayerNormalization
    model = load_model(filename, custom_objects)
    return model

import json
class NumpyEncoder(json.JSONEncoder):
    """
        Allows json to serialize numpy arrays
    """
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':
    
    print ("get_git_revision_hash: ", str(get_git_revision_hash()))
    
    print ("get_git_revision_short_hash: ", get_git_revision_short_hash())