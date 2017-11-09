import math

def splitInWindows(date, origFile, windowSize = 1.0):
    f = open(origFile)
    out = open("cleanMultiStates" + date + ".csv", "w+")
    out.flush()
    isFirst = True
    startTime = 0.0
    currentTime = 0.0
    endTime = 0.0
    priority = {}
    priority["picking up"] = 4
    priority["putting down"] = 4
    priority["holding still"] = 3
    priority["holding walking"] = 3
    priority["hovering"] = 2
    priority["nearby"] = 1
    prevState = ""
    prevPriority = 0
    prevTime = float(0)
    for line in f:
        line = line.strip()
        r = line.split(",")
        currentTime = float(r[3])
        endTime = float(r[5])
        dispState = "%s,%s,%s,%s" % (r[8].strip(),r[9].strip(),r[10].strip(),r[11].strip())
        if isFirst:
            isFirst = False;
            startTime = currentTime
        elif priority[r[8].strip()] < prevPriority:
            if endTime < startTime + windowSize:
                continue
            else:
                out.write("%.3f,%s\n" % (startTime, prevState))
                startTime += windowSize
        elif priority[r[8].strip()] == prevPriority:
            if currentTime - startTime < prevTime:
                out.write("%.3f,%s\n" % (startTime, prevState))
                startTime += windowSize
        while startTime <= endTime - windowSize:
            out.write("%.3f,%s\n" % (startTime, dispState))
            startTime += windowSize
        prevState = dispState
        prevPriority = priority[r[8].strip()]
        prevTime = endTime - startTime
    out.write("%.3f,%s\n" % (startTime, prevState))

# splitInWindows("Nov11", "clean multi states _ p1 nov 11 home visit pickup-putdown.csv")

# windowSize = 1.0
# f = open("clean multi states _ p1 nov 11 home visit pickup-putdown.csv")
# out = open("cleanMultiStatesNov11.csv", "w+")
# out.flush()
# isFirst = True
# startTime = 0.0
# currentTime = 0.0
# endTime = 0.0
# priority = {}
# priority["picking up"] = 4
# priority["putting down"] = 4
# priority["holding still"] = 3
# priority["holding walking"] = 3
# priority["hovering"] = 2
# priority["nearby"] = 1
# prevState = ""
# prevPriority = 0
# prevTime = float(0)
# for line in f:
#     line = line.strip()
#     r = line.split(",")
#     currentTime = float(r[3])
#     endTime = float(r[5])
#     dispState = "%s,%s,%s,%s" % (r[8].strip(),r[9].strip(),r[10].strip(),r[11].strip())
#     if isFirst:
#         isFirst = False;
#         startTime = currentTime
#     elif priority[r[8].strip()] < prevPriority:
#         if endTime < startTime + windowSize:
#             continue
#         else:
#             out.write("%.3f,%s\n" % (startTime, prevState))
#             startTime += windowSize
#     elif priority[r[8].strip()] == prevPriority:
#         if currentTime - startTime < prevTime:
#             out.write("%.3f,%s\n" % (startTime, prevState))
#             startTime += windowSize
#     while startTime <= endTime - windowSize:
#         out.write("%.3f,%s\n" % (startTime, dispState))
#         startTime += windowSize
#     prevState = dispState
#     prevPriority = priority[r[8].strip()]
#     prevTime = endTime - startTime
# out.write("%.3f,%s\n" % (startTime, prevState))
