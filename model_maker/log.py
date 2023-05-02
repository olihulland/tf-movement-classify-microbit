import serial
import sys
import statistics

if (len(sys.argv) != 3):
    print("usage: python log.py [port] [outputFile]")
    exit(-1);

port = sys.argv[1]
outputFileName = sys.argv[2]

class AccelValues:
    def __init__(self, line):
        asList = line.split(",")
        self.x = int(asList[0])
        self.y = int(asList[1])
        self.z = int(asList[2])
        self.strength = int(asList[3])
        self.classification = int(asList[4])

    def __str__(self):
        return f"{self.x},{self.y},{self.z},{self.strength},{self.classification}"
    
def calculateStats(valuesList):
    xValues = [v.x for v in valuesList]
    yValues = [v.y for v in valuesList]
    zValues = [v.z for v in valuesList]
    strengthValues = [v.strength for v in valuesList]

    xMax = max(xValues)
    xMin = min(xValues)
    xStd = statistics.stdev(xValues)

    # calculate the number of positive peaks in the x axis
    numPosPeaks = 0
    for i in range(1, len(xValues)-1):
        if xValues[i-1] < xValues[i] and xValues[i] > xValues[i+1]:
            numPosPeaks += 1


    yMax = max(yValues)
    yMin = min(yValues)
    yStd = statistics.stdev(yValues)

    zMax = max(zValues)
    zMin = min(zValues)
    zStd = statistics.stdev(zValues)

    meanStrength = statistics.mean(strengthValues)

    # calculate number of peaks in x,y,z using mean plus a multiple of the standard deviation
    mult = 3
    xPeaks = yPeaks = zPeaks = 0
    xMean = statistics.mean(xValues); yMean = statistics.mean(yValues); zMean = statistics.mean(zValues)
    for i, (x, y, z) in enumerate(zip(xValues, yValues, zValues)):
        if x > xMean + (mult * xStd):
            xPeaks += 1
        if y > yMean + (mult * yStd):
            yPeaks += 1
        if z > zMean + (mult * zStd):
            zPeaks += 1

    return xMax, xMin, xStd, xPeaks, yMax, yMin, yStd, yPeaks, zMax, zMin, zStd, zPeaks, meanStrength

with open(outputFileName, "a") as file:
    with serial.Serial(port, 115200) as ser:
        currentID = None
        currentValues = []

        while True:
            line = ser.readline().decode().strip()
            
            if line.startswith("#START"):
                currentID = line.split(" ")[-1]

            if line.startswith("#END"):
                stats = calculateStats(currentValues) + (currentValues[0].classification,)
                print(stats)
                currentID = None
                currentValues = []
                file.write(",".join([str(s) for s in stats]) + "\n")

            if not line.startswith("#") and currentID is not None:
                accelValues = AccelValues(line)
                currentValues.append(accelValues)