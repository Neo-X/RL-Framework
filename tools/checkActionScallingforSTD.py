"""
    This script will help understand how the distribution changes
    when samples are taken from a Gaussian distribution in different ways.
"""

import numpy as np

if __name__ == '__main__':
    
    samples = 200
    x = []
    std_ = 3.4
    avg_ = 2.7
    stds = []
    y=[]
    for i in range(samples):
        # stds.append(np.random.normal(loc=std_, scale=1.1, size=1)[0])
        # x.append(np.random.normal(loc=i*avg_/10.0, scale=stds[i], size=1)[0])
        x.append(np.random.normal(loc=i*avg_/10.0, scale=std_, size=1)[0])
        
    x = np.array(x)
    
    print("mean: ", np.mean(x))
    print("std: ", np.std(x))
    print("mean of stds: ", np.mean(stds))
    y = (x - avg_)/std_
    print("mean scaled: ", np.mean(y))
    print("std scaled: ", np.std(y))
    ## scaling back
    y = (y * std_) + avg_
    print("mean scaled: ", np.mean(y))
    print("std scaled: ", np.std(y))
    
    y = np.array(list(range(samples))) - 100
    std__ = np.std(y)*2.0
    y_ = y / std__
    
    print( "y std: ", std__)
    print( "y_: ", y_)
    
    ### Testing out MBAE normalization method
    ### Trying to get the average std to be a certain value.
    for i in range(10):
        # x_ = np.array([0.04, -0.03, 0.055234, 0.057245, -0.084566345])
        x_ = np.random.normal(0, 0.1, 10)
        avg_std = 0.34
        std__ = x_ / (np.mean(np.abs(x_))) * avg_std
        print ("print std: ", std__)
        print ("print mean: ", np.mean(np.abs(std__)))
    
    
    