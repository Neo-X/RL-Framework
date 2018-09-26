

from pynvml import *
nvmlInit()
print ("Driver Version:", nvmlSystemGetDriverVersion())
deviceCount = nvmlDeviceGetCount()

for i in range(deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    print ("Device", i, ":", nvmlDeviceGetName(handle))

nvmlShutdown()

