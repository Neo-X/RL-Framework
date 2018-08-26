import os
import sys
import re
print ("args: ", sys.argv)
raw_devices = files = [f for f in os.listdir("/dev") if re.match('nvidia[0-9]+', f)]
print ("raw_devices: ", raw_devices)

def getComputeDevice(index=1):
    if (index > (len(raw_devices)-1)):
        print ("\n Not enough GPU devices returning default (0) \n")
        return "0"
    raw_devices.sort()
    print ("sorted devices: ", raw_devices)
    print ("selected device: ", raw_devices[index])
    ### return BUS ID
    return raw_devices[index][6:]
    
if (len(sys.argv) == 2):
    if (sys.argv[1] == "second"):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = getComputeDevice()
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = getComputeDevice(int(sys.argv[1]))
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
else:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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