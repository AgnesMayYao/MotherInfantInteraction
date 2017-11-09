import oneSecond
import parsehr
import syncher
import plotter

def synchAll(date, origFile, momHR, babyHR, momBarometer, momAccelerometer, babyBarometer, babyAccelorometer, synchFile, windowSize = 1.0, elanOffset = 0):
    oneSecond.splitInWindows(date, origFile, windowSize)
    parsehr.convertFile(date, momHR, babyHR)
    syncher.synchIt(date, momBarometer, momAccelerometer, babyBarometer, babyAccelorometer, synchFile)
    plotter.plotIt(date, elanOffset)

#synchAll("Nov11", "clean multi states _ p1 nov 11 home visit pickup-putdown.csv", "MomPolar-Nov11.tcx", "P1 _ baby polar _ 2016-11-11_13-53-06.tcx", "Mom_2016-11-11T13.57.58.368_D4D7C843B379_Pressure (1).csv", "Mom_2016-11-11T13.57.58.368_D4D7C843B379_Accelerometer.csv", "Baby_2016-11-11T13.57.58.368_FB7D0BFC0351_Pressure.csv", "Baby_2016-11-11T13.57.58.368_FB7D0BFC0351_Accelerometer.csv", "p1.nov11_synch.csv")
