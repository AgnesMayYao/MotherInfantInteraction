import csv
import numpy as np
import os
import pandas
import random
import time
import re
import shutil
import datetime
import pickle
import copy
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def getData(isTest):
    if isTest:
        base = 'data/Nov11/'
    else:
        base = 'data/Oct28/'
    files = os.listdir(base)
    #empty for now
    labels = []
    hrData = []
    for f in files:
        if 'Acc' in f:
            if 'Mom' in f:
                accMomData = (pandas.read_csv(base + str(f)))
            else:
                accData = (pandas.read_csv(base + str(f)))
        elif 'Barometer' in f:
            if 'Mom' in f:
                pressureMomData = (pandas.read_csv(base + str(f)))
            else:
                pressureData = (pandas.read_csv(base + str(f)))
        elif 'Polar' in f:
            if 'Mom' in f:
                hrMomData = (pandas.read_csv(base + str(f)))
            else:
                hrData = (pandas.read_csv(base + str(f)))
        elif 'ManuallyCleanMultiStates' in f:
            labels = (pandas.read_csv(base + str(f)))

    return [(accData, accMomData),  (pressureData, pressureMomData), (hrData, hrMomData), labels]

def getWindows(data, size):
    bins = []
    times = data['epoc (ms)']
    timesNew = [0] * len(times)
    prevSec = 0
    currBin = []
    times = [ t / 1000.0 for t in times]
    for i, t in enumerate(times):
        if int(t) <= prevSec + size - 1:
            currBin.append(data.ix[i])
        else:
            bins.append(currBin)
            currBin = []
            prevSec = int(t)

    return np.array(bins)

def getWindowsHR(data, size):
    bins = [None] * len(data['Times'])
    j = 0
    hr = data['HR']
    hr=np.array(hr)
    pos=0
    for j in range(len(bins)):
        curr = []
        for i in range(size):
            curr.append(hr[pos])
        pos+=1
        bins[j] = curr
        j = j + 1
    return np.array(bins)

def convertDataPressure(baby, mom):
    #convert formats for press
    for w in range(len(baby)):
        for i in range(len(baby[w])):
            baby[w][i] = baby[w][i]['pressure (Pa)']
        for j in range(len(mom[w])):
            mom[w][j] = mom[w][j]['pressure (Pa)']

def convertDataAccel(baby, mom):
    #convert formats for acc
    t = 0
    samplesInWindow = len(baby[0]) - 1
    for i in range(samplesInWindow):
        t += abs(baby[0][i]['epoc (ms)'] - baby[0][i + 1]['epoc (ms)'])


    deltat = t / float(1000 * samplesInWindow)
    for w in range(len(baby)):
        for i in range(len(baby[w])):
            baby[w][i]=np.sqrt(np.square(baby[w][i]['x-axis (g)'])+np.square(baby[w][i]['y-axis (g)'])+np.square(baby[w][i]['z-axis (g)']))
            #baby[w][i] = (abs(baby[w][i]['x-axis (g)']) + abs(baby[w][i]['y-axis (g)']) + abs(baby[w][i]['z-axis (g)'])) / 3.0
        for j in range(len(mom[w])):
            #mom[w][j] = (abs(mom[w][j]['x-axis (g)']) + abs(mom[w][j]['y-axis (g)']) + abs(mom[w][j]['z-axis (g)']) )/ 3.0
            mom[w][j]=np.sqrt(np.square(mom[w][j]['x-axis (g)'])+np.square(mom[w][j]['y-axis (g)'])+np.square(mom[w][j]['z-axis (g)']))


    return deltat

def pressureDiff(baby, mom):
    diffs = np.zeros((len(baby)))

    #find differences
    for i in range(len(baby)):
        babyRange = max(baby[i]) - min(baby[i])
        momRange = max(mom[i]) - min(mom[i])
        diffs[i] = abs(momRange - babyRange)
    return diffs

def featAvg(baby, mom):
    avgs = np.zeros((len(baby)))
    #find averages
    for i in range(len(baby)):
        momAvg = np.mean(baby[i])

        babyAvg = np.mean(mom[i])
        avgs[i] = (momAvg + babyAvg) / 2.0
    return avgs


def distComp(baby, mom, deltat):
    #integrate over windows
    ratios = np.zeros((len(baby)))
    avgs = np.zeros((len(baby)))
    for i in range(len(baby)):
        #return values for distance travelled during a window
        windowDistancesBaby = (integrate(baby[i], deltat))
        windowDistancesMom = (integrate(mom[i], deltat))
        distTraveledMom = max(windowDistancesMom) - min(windowDistancesMom)
        distTraveledBaby = max(windowDistancesBaby) - min(windowDistancesBaby)
        ratios[i] = distTraveledMom / distTraveledBaby
        avgs[i] = (distTraveledMom + distTraveledBaby) / 2.0
    return (ratios, avgs)

def integrate(acc, deltat):
    deltat = float(deltat)
    v = [0] * len(acc)
    for i in range(len(acc)):
        if i < len(acc) - 1:
            v[i+1] = acc[i] * deltat + v[i]
    d = [0] * len(acc)
    for i in range(len(v)):
        if i < len(acc) - 1:
            d[i + 1] = v[i]*deltat + d[i]
    return d


def crossCorr(baby, mom):
    correlations = np.zeros((len(baby)))
    for i in range(len(baby)):
        babyC = np.array(baby[i])
        momC = np.array(mom[i])
        correlations[i] = np.mean(np.correlate(babyC, momC))
    return correlations

def hrComp(baby, mom):
    ratios = np.zeros((len(baby)))
    for i in range(len(baby)):
        if mom[i]==0:
            ratios[i]=baby[i]
        else:
            ratios[i] = float(baby[i]) / mom[i]
    return ratios

def convertLabels(labels, numStates):
    times = labels['epoc (ms)']
    times = [t / 1000.0 for t in times]
    try:
        rawStates = labels[str(numStates) + ' states']
    except KeyError:
        rawStates = labels[' ' + str(numStates) + ' states']
    categories = np.unique(rawStates)
    #print categories
    statesMap = {}
    for i, c in enumerate(categories):
        statesMap[c] = i
    plottableStates = [statesMap[s] for s in rawStates]
    return (times, plottableStates, statesMap)

def plotFeatures(features, rawData, labels):
    fig1 = plt.figure(1)
    x = np.arange(0, len(features[0][0]))
    n = len(features)

    #Raw data plots
    titles = ['Raw Accelerometer Baby', 'Raw Accelerometer Mom', 'Raw Pressure Baby',  'Raw Pressure Mom', 'Raw HR Baby', 'Raw HR Mom']
    m = len(titles)
    numRaw = 0
    for j in range(3):
        for p in rawData[j]:
            plot = fig1.add_subplot(m+1, 1,numRaw+1)
            plot.set_title(titles[numRaw])
            if j is not 2:
                times = p['epoc (ms)']
                times = [int(int(t) / 1000) for t in times]
            if j == 0:
                plot.plot(times, p['x-axis (g)'])
                plot.plot(times, p['y-axis (g)'])
                plot.plot(times, p['z-axis (g)'])
            if j == 1:
                plot.plot(times, p['pressure (Pa)'])
            if j == 2:
                times = p['Times']
                times = [int(int(t) / 1000) for t in times]
                plot.plot(times, p['HR'])
            numRaw += 1

    plt.tight_layout()
    #Features plots
    fig2 = plt.figure(2)
    for i, f in enumerate(features):
        plot = fig2.add_subplot(n+1, 1, i+1)
        plot.set_title(f[1])
        plot.plot(x, f[0])
    #Elan plot
    plot = fig2.add_subplot(n+1, 1, i+2)
    plot.set_title("Elan Labels")
    plot.scatter(labels[0], labels[1])
    plot.set_xlim(0, len(features[0][0]))
    plt.tight_layout()
    plt.show()

def filterFeats(X, y, pca):
    if pca is None:
        pca = PCA(n_components = 5)
    #CHECK HERE
    filteredX = pca.fit_transform(X, y)
    indices = []
    components = np.argsort(pca.components_)
    for i in range(len(components[0])):
        for j in range(len(components)):
            if len(indices) < 5:
                if components[j][i] not in indices:
                    indices.append(components[j][i])
            else:
                break
    return (filteredX, pca, indices)

def normalHR(baby,mom):
    all_data=[]
    for i in range(len(baby)):
        for j in range(len(baby[i])):
            all_data.append(baby[i][j])
    for i in range(len(mom)):
        for j in range(len(mom[i])):
            all_data.append(mom[i][j])

    mean_value=np.mean(all_data)
    sd_value=np.std(all_data)

    for i in range(len(baby)):
        baby[i]=(int(baby[i])-mean_value)/sd_value
    for i in range(len(mom)):
        mom[i]=(int(mom[i])-mean_value)/sd_value
    return baby,mom
    
def normal(baby,mom):
    all_data=[]
    for i in range(len(baby)):
        for j in range(len(baby[i])):
            all_data.append(baby[i][j])
    for i in range(len(mom)):
        for j in range(len(mom[i])):
            all_data.append(mom[i][j])

    mean_value=np.mean(all_data)
    sd_value=np.std(all_data)

    for i in range(len(baby)):
        baby[i]=np.divide(np.subtract(baby[i],mean_value),sd_value)
    for i in range(len(mom)):
        mom[i]=np.divide(np.subtract(mom[i],mean_value),sd_value)
    return baby,mom

def storeData(babyAcc,momAcc,babyPre,momPre,babyHr,momHr):
    f = open("save", "wb")
    babyAcc_t=[]
    for i in range(len(babyAcc)):
        babyAcc_t.append(np.mean(babyAcc[i]))
    pickle.dump(babyAcc_t, f)
    momAcc_t=[]
    for i in range(len(momAcc)):
        momAcc_t.append(np.mean(momAcc[i]))
    pickle.dump(momAcc_t, f)
    babyPre_t=[]
    for i in range(len(babyPre)):
        babyPre_t.append(np.mean(babyPre[i]))
    pickle.dump(babyPre_t, f)
    momPre_t=[]
    for i in range(len(momPre)):
        momPre_t.append(np.mean(momPre[i]))
    pickle.dump(momPre_t, f)
    babyHr_t=[]
    for i in range(len(babyHr)):
        babyHr_t.append(np.mean(babyHr[i]))
    pickle.dump(babyHr_t, f)
    momHr_t=[]
    for i in range(len(momHr)):
        momHr_t.append(np.mean(momHr[i]))
    pickle.dump(momHr_t, f)
    f.close()

    


def getFeatures(windowSize, numStates, pca = None, isTest = False):
    #parse data
    data = getData(isTest)

    #break into windows
    accBins = getWindows(data[0][0], windowSize)
    accMomBins = getWindows(data[0][1], windowSize)
    pressureBins = getWindows(data[1][0], windowSize)
    pressureMomBins = getWindows(data[1][1], windowSize)

    hrBins = getWindowsHR(data[2][0], windowSize)
    hrMomBins = getWindowsHR(data[2][1], windowSize)
    #assuming labels are already second-value pairs
    labels = data[3]

    #cut off data at the same point
    minLength = np.min([len(accBins), len(accMomBins), len(pressureBins), len(pressureMomBins), len(labels)])
    #minLength = np.min([len(accBins), len(accMomBins), len(pressureBins), len(pressureMomBins), len(labels),len(hrBins),len(hrMomBins)])
    accBins = accBins[0:minLength]
    accMomBins = accMomBins[0:minLength]
    pressureBins = pressureBins[0:minLength]   
    pressureMomBins = pressureMomBins[0:minLength]
    #hrBins=hrBins[0:minLength]
    #hrMomBins=hrMomBins[0:minLength]
    labels = labels[0:minLength]

    #convert data into usable format
    deltat = convertDataAccel(accBins, accMomBins)
    convertDataPressure(pressureBins, pressureMomBins)


    labels = convertLabels(labels, numStates)

    '''
    preSec=copy.deepcopy(labels[1])
    preSec.pop()
    preSec.insert(0,0)
    
    pre2Sec=copy.deepcopy(labels[1])
    pre2Sec.pop()
    pre2Sec.pop()
    pre2Sec.insert(0,0)
    pre2Sec.insert(0,0)
    '''
    



    #store data in pickle to plot later
    #storeData(accBins,accMomBins,pressureBins,pressureMomBins,hrBins,hrMomBins)

    #normalization
    accBins,accMomBins=normal(accBins,accMomBins)
    pressureBins,pressureMomBins=normal(pressureBins,pressureMomBins)
    #hrBins,hrMomBins=normalHR(hrBins,hrMomBins)


    #get features
    pDiff = pressureDiff(pressureBins, pressureMomBins)
    aAvg = featAvg(pressureBins, pressureMomBins)

    dRatio, dAvg = distComp(accBins, accMomBins, deltat)
    aCorr = crossCorr(accBins, accMomBins)
    #hrAvg = featAvg(hrBins, hrMomBins)
    #hrRatio = hrComp(hrBins, hrMomBins)
    #hrCorr = crossCorr(hrBins, hrMomBins)

    #combine into feature vector
    #X = [dRatio, dAvg,aCorr,preSec,pre2Sec]



    
    '''
    x=[1,2,3,4,5]

    prePressureDiff=pDiff[:]
    np.delete(prePressureDiff,-1)
    #prePressureDiff=np.zeros((5,1))+prePressureDiff
    print prePressureDiff.shape
    prePressureDiff=[0]+prePressureDiff
    difPressureDiff=[]
    for i in range(0,len(pDiff)):
        difPressureDiff.append(np.polyfit(x,prePressureDiff[i:i+5],1)[0])



    preAAvg=aAvg[:]

    np.delete(preAAvg,-1)
    preAAvg=np.zeros((5,1))+preAAvg
    difAAvg=[]
    for i in range(0,len(aAvg)):
        difAAvg.append(np.polyfit(x,preAAvg[i:i+5],1)[0])


    preDRatio=dRatio[:]
    np.delete(preDRatio,-1)
    preDRatio=[[0]*5]+preDRatio
    difDRatio=[]
    for i in range(0,len(dRatio)):
        difDRatio.append(np.polyfit(x,preDRatio[i:i+5],1)[0])

    preDAvg=dAvg[:]
    np.delete(preDAvg,-1)
    preDAvg=[[0]*5]+preDAvg
    difDAvg=[]
    for i in range(0,len(pDiff)):
        difDAvg.append(np.polyfit(x,preDAv[i:i+5],1)[0])


    preACorr=aCorr[:]
    np.delete(preACorr,-1)
    preACorr=[[0]*5]+preACorr
    difACorr=[]
    for i in range(0,len(aCorr)):
        difACorr.append(np.polyfit(x,preACorr[i:i+5],1)[0])

    '''


    #X = [difPressureDiff,difAAvg,difDRatio,difDAvg,difACorr,pDiff, aAvg, dRatio, dAvg, aCorr]
    #X = [dRatio, dAvg, aCorr]
    X = [pDiff, aAvg, dRatio, dAvg, aCorr]

    #X = [pDiff, aAvg, dRatio, dAvg, aCorr, hrAvg, hrRatio, hrCorr,preSec,pre2Sec]
    X = np.array(X).T
    Y = labels[1]
    Y = np.array(Y)
    dictionary = labels[2]

    print dictionary
    #reducing via PCA
    #X, pca, indices = filterFeats(X, Y, pca)

    
    #plotting features
    if not isTest:
        #plotX = [(pDiff, "Difference in Altitude"), (aAvg, "Average Altitude"), (dRatio, "Distance Ratio"), (dAvg, "Average Distance"), (aCorr, "Accel Correlation"), (hrAvg, "Average Heart Rate"), (hrRatio, "Heart Rate Ratio"), (hrCorr, "Heart Rate Corr")]
        plotX = [(pDiff, "Difference in Altitude"), (aAvg, "Average Altitude"), (dRatio, "Distance Ratio"), (dAvg, "Average Distance"), (aCorr, "Accel Correlation")]

        #incorporating PCA results
        #plotX = [plotX[i] for i in indices]
        plotFeatures(plotX, data, labels)
    

    return (X, Y, pca, dictionary)

################# MAIN METHOD ####################
def getInput(windowSize = 1, numStates = 5):
    response = input("Are you planning on using one session as training data and one session as testing data? (0/1)")
    if response is 1:
        confirm = input("Are the synched training files in a directory called data/ and the synched testing files in a directory called dataTest/ (0/1)")
        while confirm is not 1:
            confirm = input("Have you moved the files to the appropriate directories? (0/1)")
        trainX, trainY, pca, dictionary = getFeatures(windowSize, numStates)
        testX, testY, pca, dictionary = getFeatures(windowSize, numStates, pca, True)
        print "The dimensions of trainX are: " + str(trainX.shape)
        print "The dimensions of trainY are: " + str(trainY.shape)
        print "The dimensions of testX are: " + str(testX.shape)
        print "The dimensions of testY are: " + str(testY.shape)
        return [(trainX, trainY), (testX, testY), dictionary]
    else:
        trainX, trainY, pca, dictionary = getFeatures(windowSize, numStates)
        print "The dimensions of X are: " + str(trainX.shape)
        print "The dimensions of Y are: " + str(trainY.shape)
        print "Split this data on your own."
        return (trainX, trainY, dictionary)
