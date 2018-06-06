

## Silly module with some odd state variables

global __STATE
__STATE = 3

def add(x):
    if __STATE == 3:
        return x + __STATE
    else:
        return 0

def finish():
    global __STATE
    __STATE = None
    del __STATE
    return 
    
    