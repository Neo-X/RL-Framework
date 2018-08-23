import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


if __name__ == "__main__":
    
    gpus = get_available_gpus()
    
    print (gpus)
    devices = get_available_devices()
    print (devices)
    
    import tensorflow as tf
    tf.device(devices[0])
    os.environ['KERAS_BACKEND'] = "tensorflow"
    import keras
    
    x = tf.placeholder(tf.float32, shape=(None, 20, 64))
    y = keras.layers.LSTM(32)(x)  # all ops in the LSTM layer will live on CPU:0
    

    print("y: ", y)
    print ("tf.get_default_graph(): ", tf.get_default_graph())
    print (keras.backend.get_session().list_devices())