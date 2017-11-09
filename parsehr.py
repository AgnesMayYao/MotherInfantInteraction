import time
import csv
from datetime import datetime
from xml.dom import minidom

#Pull values and times from xml file
def parseXML(filename, returnFileName):
    toplevel = []
    pairs = []

    xmldoc = minidom.parse(filename)
    toplevel.append(["Times"] + ["HR"])

    allHR = xmldoc.getElementsByTagName('Value')
    allTimes = xmldoc.getElementsByTagName('Time')

    length = min(len(allTimes), len(allHR))
    for i in range(1, length - 1):
        hrNode = allHR[i + 1]
        timeNode = allTimes[i]
        polarTime = timeNode.firstChild.data
        unixTime = convertTime(polarTime)
        pair = [unixTime] + [hrNode.firstChild.data]
        pairs.append(pair)
    return generateCSV(toplevel, pairs, returnFileName)

#Convert to CSV
def generateCSV(toplevel, pairs, fileName):
    file = open(fileName, 'wb')
    wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    print toplevel[0]
    wr.writerow(toplevel[0])
    for pair in pairs:
        wr.writerow(pair)
    return fileName

#Helper method to convert time stamps
def convertTime(polarTime):
    #strip time zone information
    timeNoTZ = datetime.strptime(polarTime, '%Y-%m-%dT%H:%M:%S.%fZ')
    #convert to unixtime
    unixTime = time.mktime(timeNoTZ.timetuple())
    #4 hour offset
    unixTime += 4*60*60
    #miliseconds
    unixTime *= 1000
    return unixTime

def convertFile(date, momFile, babyFile):
    parseXML(momFile, "MomHR" + date + ".csv")
    parseXML(babyFile, "BabyHR" + date + ".csv")

#convertFile("Nov11", "MomPolar-Nov11.tcx", "P1 _ baby polar _ 2016-11-11_13-53-06.tcx")
