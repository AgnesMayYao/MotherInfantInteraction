import numpy as np
import pylab as pl

def plotIt(date, elanOffset = 0):
    momBaro = open("synchedMomBarometer" + date + ".csv")
    momAccel = open("synchedMomAccelerometer" + date + ".csv")
    elan = open("synchedCleanMultiStates" + date + ".csv")
    elanOut = open("synchedManuallyCleanMultiStates" + date + ".csv", "w+")
    x = []
    y = []
    isFirst = True
    for line in momBaro:
        if isFirst:
            isFirst = False
            continue
        data = line.strip().split(",")
        x.append(int(data[0]))
        y.append(float(data[1]))
    momBaro.close()

    X = []
    values = {}
    values["nearby"] = 1
    values["holding"] = 2
    values["picking up/ putting down"] = 3
    Y = []
    elanOut.flush()
    for line in elan:
        data = line.strip().split(",")
        X.append(int(data[0]) + elanOffset)
        Y.append(values[str(data[3])])
        elanOut.write("%d,%s,%s,%s,%s\n" % (int(data[0]) + elanOffset, data[1], data[2], data[3], data[4]))
    elan.close()

    babyBaro = open("synchedBabyBarometer" + date + ".csv")
    babyAccel = open("synchedBabyAccelerometer" + date + ".csv")
    x2 = []
    y2 = []
    isFirst = True
    for line in babyBaro:
        if isFirst:
            isFirst = False
            continue
        data = line.strip().split(",")
        x2.append(int(data[0]))
        y2.append(float(data[1]))
    babyBaro.close()

    pl.figure(1)
    pl.subplot(211)
    pl.plot(x, y, 'r')
    pl.plot(x2, y2, 'g')
    pl.xlim(0, 100000)

    pl.subplot(212)
    pl.plot(X, Y, 'b')
    pl.xlim(0, 100000)
    pl.ylim(0, 4)
    pl.show()

# momBaro = open("synchedMomBarometerNov11.csv")
# momAccel = open("synchedMomAccelerometerNov11.csv")
# elan = open("synchedCleanMultiStatesNov11.csv")
# elanOut = open("synchedManuallyCleanMultiStatesNov11.csv", "w+")
# x = []
# y = []
# isFirst = True
# for line in momBaro:
#     if isFirst:
#         isFirst = False
#         continue
#     data = line.strip().split(",")
#     x.append(int(data[0]))
#     y.append(float(data[1]))
# momBaro.close()
#
# X = []
# values = {}
# values["nearby"] = 1
# values["holding"] = 2
# values["picking up/ putting down"] = 3
# Y = []
# elanOut.flush()
# elanOffset = 2000
# for line in elan:
#     data = line.strip().split(",")
#     X.append(int(data[0]) + elanOffset)
#     Y.append(values[str(data[3])])
#     elanOut.write("%d,%s,%s,%s,%s\n" % (int(data[0]) + elanOffset, data[1], data[2], data[3], data[4]))
# elan.close()
#
# babyBaro = open("synchedBabyBarometerNov11.csv")
# babyAccel = open("synchedBabyAccelerometerNov11.csv")
# x2 = []
# y2 = []
# isFirst = True
# for line in babyBaro:
#     if isFirst:
#         isFirst = False
#         continue
#     data = line.strip().split(",")
#     x2.append(int(data[0]))
#     y2.append(float(data[1]))
# babyBaro.close()
#
# # pl.figure(1)
# # pl.subplot(211)
# # pl.plot(x, y, 'r')
# # pl.plot(x2, y2, 'g')
# # pl.xlim(0, 100000)
# #
# # pl.subplot(212)
# # pl.plot(X, Y, 'b')
# # pl.xlim(0, 100000)
# # pl.ylim(0, 4)
# # pl.show()
#
# polar = open("synchedMomPolarNov11.csv")
# a = []
# b = []
# isFirst = True
# for line in momAccel:
#     if isFirst:
#         isFirst = False
#         continue
#     data = line.strip().split(",")
#     a.append(int(data[0]))
#     b.append((abs(float(data[1])) + abs(float(data[2])) + abs(float(data[3]))) / 3)
#
# a1 = []
# b1 = []
# isFirst = True
# for line in polar:
#     if isFirst:
#         isFirst = False
#         continue
#     data = line.strip().split(",")
#     a1.append(int(data[0]))
#     b1.append(int(data[1]))
#
# pl.figure(1)
# pl.subplot(211)
# pl.plot(a, b, 'r')
# pl.xlim(0, 100000)
#
# pl.subplot(212)
# pl.plot(a1, b1, 'b')
# pl.xlim(0, 100000)
# pl.show()
