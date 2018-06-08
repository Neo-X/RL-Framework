
def doSomethingComplicated(x):
    
    # x = x + y
    # import some libraries
    import stateFullStuff 
    # create some of its own processes and run them
    y = stateFullStuff.add(x)
    print (y)
    stateFullStuff.finish()
    # print ("finished")
    return y

from multiprocessing import Pool

if __name__ == '__main__':
    
    x_ = range(1, 100)
    pool = Pool(4, maxtasksperchild=1)
    results = pool.map(doSomethingComplicated,x_)
    print ("results: ", results)