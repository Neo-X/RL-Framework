
def addLogData(trainData, key, data):
    if key in trainData:
        trainData[key].append(data)
    else:
        trainData[key] = [data]
