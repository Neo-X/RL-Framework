
def doSomethingComplicated(x):
    
    # x = x + y
    # import some libraries
    import stateFullStuff 
    # create some of its own processes and run them
    x = stateFullStuff.add(x)
    print (x)
    stateFullStuff.finish()
    # print ("finished")
    return x

from multiprocessing import Pool

if __name__ == '__main__':
    
    x_ = range(100)
    pool = Pool(3)
    results = pool.map(doSomethingComplicated,x_)
    print ("results: ", results)