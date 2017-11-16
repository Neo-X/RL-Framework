
import numpy as np

if __name__ == '__main__':
    
    x = np.array(np.random.normal(0,2.5,100000))
    
    
    
    print ("x mean: ", np.mean(x))
    x_mean = x[0]
    x_mean2 = x[0]
    for i in range(1,len(x)):
        x_mean = x_mean + ((x[i] - x_mean)/i)
    
        x_mean2 = (((i-1) * x_mean2) + x[i])/i
    
    print ("Running x_mean: ", x_mean)
    print ("Running x_mean2: ", x_mean2)