

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
    
def saveVAEBatch(directory, model):
    """
        Used to produce and save VAE outputs from the model
    """
    # import scipy.misc
    from model.ModelUtil import scale_state
    
    states, actions, result_states, rewards, falls, G_ts, exp_actions, advantage = model.getFDExperience().get_batch(32)
    vae_out = model.getForwardDynamics().predict_batch(states, actions)
    # vae_out = scale_state(vae_out, model.getForwardDynamics().getStateBounds())
    print("vae_out: ", vae_out)
    import matplotlib
    import numpy as np

    for i in range ( len(vae_out)):
        # print ("states.shape: ", states[i].shape)
        state = states[i]
        state = np.array(np.reshape(state[:64*64*3], (64,64,3)))
        state = state + -min(np.min(state), 0)
        state = state / np.max(state)
        
        img = np.array(np.reshape(vae_out[i][:64*64*3], (64,64,3)))
        img = img + -min(np.min(img), 0)
        img = img / np.max(img)
        img = np.concatenate((state, img), axis=1)
        # print (img.shape)
        matplotlib.image.imsave(directory + '/name_'+str(i)+'.png', img)
    
    # scipy.misc.imsave('outfile.jpg', vae_out[0])
    
def rlPrint(settings=None, level="train", text=""):
    if (settings["print_levels"][settings["print_level"]] >= settings["print_levels"][level]):
        print (text)
        
def load_keras_model(filename, custom_objects={}):
    from keras.models import load_model
    import keras_layer_normalization
    custom_objects["LayerNormalization"] = keras_layer_normalization.LayerNormalization
    model = load_model(filename, custom_objects)
    return model

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

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