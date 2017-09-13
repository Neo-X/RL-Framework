"""
    This script will help understand how the distribution changes
    when samples are taken from a Gaussian distribution in different ways.
"""

import numpy as np

if __name__ == '__main__':
    
    samples = 200
    x = []
    std_ = 3.0
    avg_ = 4.7
    for i in range(samples):
        x.append(np.random.normal(loc=avg_, scale=std_, size=1)[0])
        
    x = np.array(x)
    print("mean: ", np.mean(x))
    print("std: ", np.std(x))
    y = (x - avg_)/std_
    print("mean scaled: ", np.mean(y))
    print("std scaled: ", np.std(y))