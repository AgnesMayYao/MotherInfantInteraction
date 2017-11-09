def synchIt(date, momBarometer, momAccelerometer, babyBarometer, babyAccelorometer, synch):
    elan = open("cleanMultiStates" + date + ".csv")
    elanOut = open("synchedCleanMultiStates" + date + ".csv", "w+")
    momAccel = open(momAccelerometer)
    momAccelOut = open("synchedMomAccelerometer" + date + ".csv", "w+")
    momBaro = open(momBarometer)
    momBaroOut = open("synchedMomBarometer" + date + ".csv", "w+")
    momPolar = open("MomHR" + date + ".csv")
    momPolarOut = open("synchedMomPolar" + date + ".csv", "w+")
    babyAccel = open(babyAccelorometer)
    babyAccelOut = open("synchedBabyAccelerometer" + date + ".csv", "w+")
    babyBaro = open(babyBarometer)
    babyBaroOut = open("synchedBabyBarometer" + date + ".csv", "w+")
    babyPolar = open("BabyHR" + date + ".csv")
    babyPolarOut = open("synchedBabyPolar" + date + ".csv", "w+")
    synchFile = open(synch)

    i = 0
    synchPoints = []
    latest = 0
    time = 0
    for line in synchFile:
        data = line.strip().split(",")
        point = int(float(data[2]) * 1000)
        synchPoints.append(point)
        if point > latest:
            latest = point
        if i == 3:
            time = int(data[-1])
        i += 1
    for i in range(len(synchPoints)):
        synchPoints[i] = (latest - synchPoints[i])

    mbList = [momAccel, momBaro, babyAccel, babyBaro]
    mbListOut = [momAccelOut, momBaroOut, babyAccelOut, babyBaroOut]
    synchDiff = synchPoints[0]

    latestTime = 0
    for i in range(len(mbList)):
        isFirst = True
        isSecond = False
        for line in mbList[i]:
            if isFirst:
                isFirst = False
                isSecond = True
                continue
            if isSecond:
                isSecond = False
                data = line.strip().split(",")
                if int(data[0]) > latestTime:
                    latestTime = int(data[0])
                break

    for i in range(len(mbList)):
        f = mbList[i]
        f.seek(0)
        out = mbListOut[i]
        out.flush()
        isFirst = True
        for line in f:
            data = line.strip().split(",")
            if isFirst:
                isFirst = False
                if len(data) == 4:
                    out.write("%s,%s\n" % (data[0], data[3]))
                else:
                    out.write("%s,%s,%s,%s\n" % (data[0], data[3], data[4], data[5]))
                continue
            currentTime = int(data[0])
            relativeTime = currentTime - latestTime - synchDiff
            # if relativeTime >= 0:
            if len(data) == 4:
                out.write("%d,%s\n" % (relativeTime, data[3]))
            else:
                out.write("%d,%s,%s,%s\n" % (relativeTime, data[3], data[4], data[5]))

    synchDiff = synchPoints[1]
    momPolarOut.flush()
    isFirst = True
    isSecond = False
    startTime = 0
    for line in momPolar:
        data = line.strip().replace("\"", "").split(",")
        if isFirst:
            isFirst = False
            isSecond = True
            momPolarOut.write("%s,%s\n" % (data[0], data[1]))
            continue
        currentTime = int(float(data[0]))
        if isSecond:
            isSecond = False
            startTime = currentTime
        relativeTime = currentTime - startTime - synchDiff
        momPolarOut.write("%d,%s\n" % (relativeTime, data[1]))

    synchDiff = synchPoints[2]
    babyPolarOut.flush()
    isFirst = True
    isSecond = False
    for line in babyPolar:
        data = line.strip().replace("\"", "").split(",")
        if isFirst:
            isFirst = False
            isSecond = True
            babyPolarOut.write("%s,%s\n" % (data[0], data[1]))
            continue
        currentTime = int(float(data[0]))
        if isSecond:
            isSecond = False
            startTime = currentTime
        relativeTime = currentTime - startTime - synchDiff
        babyPolarOut.write("%d,%s\n" % (relativeTime, data[1]))


    synchDiff = synchPoints[3]
    isFirst = True
    elanOut.flush()
    for line in elan:
        data = line.strip().split(",")
        currentTime = int(float(data[0]) * 1000)
        if isFirst:
            isFirst = False
            startTime = currentTime
        relativeTime = currentTime - startTime - synchDiff
        if relativeTime >= 0:
            elanOut.write("%d,%s,%s,%s,%s\n" % (relativeTime, data[1], data[2], data[3], data[4]))

# synchIt("Nov11", "Mom_2016-11-11T13.57.58.368_D4D7C843B379_Pressure (1).csv", "Mom_2016-11-11T13.57.58.368_D4D7C843B379_Accelerometer.csv", "Baby_2016-11-11T13.57.58.368_FB7D0BFC0351_Pressure.csv", "Baby_2016-11-11T13.57.58.368_FB7D0BFC0351_Accelerometer.csv", "p1.nov11_synch.csv")


# elan = open("cleanMultiStatesNov11.csv")
# elanOut = open("synchedCleanMultiStatesNov11.csv", "w+")
# momAccel = open("Mom_2016-11-11T13.57.58.368_D4D7C843B379_Accelerometer.csv")
# momAccelOut = open("synchedMomAccelerometerNov11.csv", "w+")
# momBaro = open("Mom_2016-11-11T13.57.58.368_D4D7C843B379_Pressure (1).csv")
# momBaroOut = open("synchedMomBarometerNov11.csv", "w+")
# momPolar = open("MomHR.csv")
# momPolarOut = open("synchedMomPolarNov11.csv", "w+")
# babyAccel = open("Baby_2016-11-11T13.57.58.368_FB7D0BFC0351_Accelerometer.csv")
# babyAccelOut = open("synchedBabyAccelerometerNov11.csv", "w+")
# babyBaro = open("Baby_2016-11-11T13.57.58.368_FB7D0BFC0351_Pressure.csv")
# babyBaroOut = open("synchedBabyBarometerNov11.csv", "w+")
# babyPolar = open("BabyHR.csv")
# babyPolarOut = open("synchedBabyPolarNov11.csv", "w+")
# synchFile = open("p1.nov11_synch.csv")
#
# i = 0
# synchPoints = []
# latest = 0
# time = 0
# for line in synchFile:
#     data = line.strip().split(",")
#     point = int(float(data[2]) * 1000)
#     synchPoints.append(point)
#     if point > latest:
#         latest = point
#     if i == 3:
#         time = int(data[-1])
#     i += 1
# for i in range(len(synchPoints)):
#     synchPoints[i] = (latest - synchPoints[i])
#
# mbList = [momAccel, momBaro, babyAccel, babyBaro]
# mbListOut = [momAccelOut, momBaroOut, babyAccelOut, babyBaroOut]
# synchDiff = synchPoints[0]
#
# latestTime = 0
# for i in range(len(mbList)):
#     isFirst = True
#     isSecond = False
#     for line in mbList[i]:
#         if isFirst:
#             isFirst = False
#             isSecond = True
#             continue
#         if isSecond:
#             isSecond = False
#             data = line.strip().split(",")
#             if int(data[0]) > latestTime:
#                 latestTime = int(data[0])
#             break
#
# for i in range(len(mbList)):
#     f = mbList[i]
#     f.seek(0)
#     out = mbListOut[i]
#     out.flush()
#     isFirst = True
#     for line in f:
#         data = line.strip().split(",")
#         if isFirst:
#             isFirst = False
#             if len(data) == 4:
#                 out.write("%s,%s\n" % (data[0], data[3]))
#             else:
#                 out.write("%s,%s,%s,%s\n" % (data[0], data[3], data[4], data[5]))
#             continue
#         currentTime = int(data[0])
#         relativeTime = currentTime - latestTime - synchDiff
#         # if relativeTime >= 0:
#         if len(data) == 4:
#             out.write("%d,%s\n" % (relativeTime, data[3]))
#         else:
#             out.write("%d,%s,%s,%s\n" % (relativeTime, data[3], data[4], data[5]))
#
# synchDiff = synchPoints[1]
# momPolarOut.flush()
# isFirst = True
# isSecond = False
# startTime = 0
# for line in momPolar:
#     data = line.strip().replace("\"", "").split(",")
#     if isFirst:
#         isFirst = False
#         isSecond = True
#         momPolarOut.write("%s,%s\n" % (data[0], data[1]))
#         continue
#     currentTime = int(float(data[0]))
#     if isSecond:
#         isSecond = False
#         startTime = currentTime
#     relativeTime = currentTime - startTime - synchDiff
#     momPolarOut.write("%d,%s\n" % (relativeTime, data[1]))
#
# synchDiff = synchPoints[2]
# babyPolarOut.flush()
# isFirst = True
# isSecond = False
# for line in babyPolar:
#     data = line.strip().replace("\"", "").split(",")
#     if isFirst:
#         isFirst = False
#         isSecond = True
#         babyPolarOut.write("%s,%s\n" % (data[0], data[1]))
#         continue
#     currentTime = int(float(data[0]))
#     if isSecond:
#         isSecond = False
#         startTime = currentTime
#     relativeTime = currentTime - startTime - synchDiff
#     babyPolarOut.write("%d,%s\n" % (relativeTime, data[1]))
#
#
# synchDiff = synchPoints[3]
# isFirst = True
# elanOut.flush()
# for line in elan:
#     data = line.strip().split(",")
#     currentTime = int(float(data[0]) * 1000)
#     if isFirst:
#         isFirst = False
#         startTime = currentTime
#     relativeTime = currentTime - startTime - synchDiff
#     # if relativeTime >= 0:
#     elanOut.write("%d,%s,%s,%s,%s\n" % (relativeTime, data[1], data[2], data[3], data[4]))
