import random
from math import tanh, cosh, exp

alpha = 0.5
eta = 0.15

class Connection:
    def __init__(self, w, dw):
        self.w = w
        self.dw = dw

class Neuron:
    def __init__(self, myInd, numInp):
        self.myInd = myInd
        self.numInp = numInp
        self.inputVal = 1
        self.outputVal = 1
        self.connections = []
        self.sigm = 0
        for nNum in range(self.numInp):
            self.connections.append(Connection(w=randomWeight(), dw=0))

    def activationFunc(self, x):
        # return (x / (abs(x) + 1) + 1) / 2
        return 1 / (1 + exp(-x))

    def dActivationFunc(self, x):
        # return 0.5 / (1 +abs(x)) ** 2
        return self.activationFunc(x) * (1 - self.activationFunc(x))

    def calcVal(self, prevLayer):
        summ = 0
        for nNum in range(self.numInp):
            summ += prevLayer[nNum].outputVal * self.connections[nNum].w

        self.setVal(summ)

    def setVal(self, x):
        self.inputVal = x
        self.outputVal = self.activationFunc(x)

    def updateConnections(self, prevLayer):
        for prevNeu in prevLayer:
            self.connections[prevNeu.myInd].dw = alpha * self.connections[prevNeu.myInd].dw + (1 - alpha) * eta * self.sigm * prevNeu.outputVal

    def applyUpdates(self):
        for prevNeuInd in range(self.numInp):
            self.connections[prevNeuInd].w += self.connections[prevNeuInd].dw
            self.connections[prevNeuInd].dw = 0

class Net:
    def __init__(self, topology):
        assert len(topology) >= 3

        self.nInp = topology[0] + 1
        self.hidLaySize = [elem + 1 for elem in topology[1:-1]]
        self.nHid = len(topology) - 2
        self.nOut = topology[-1]

        self.inpLayer = []
        self.hidLayers = [[] for i in range(self.nHid)]
        self.outLayer = []

        for nNum in range(self.nInp):
            self.inpLayer.append(Neuron(nNum, 0))

        for lNum in range(self.nHid):
            for nNum in range(self.hidLaySize[lNum]):
                self.hidLayers[lNum].append(Neuron(nNum, self.nInp if lNum == 0 else self.hidLaySize[lNum - 1]))

        for nNum in range(self.nOut):
            self.outLayer.append(Neuron(nNum, self.hidLaySize[-1]))

    def feedForw(self, inpVal):
        assert len(inpVal) == self.nInp - 1
        for nNum in range(self.nInp - 1):
            self.inpLayer[nNum].outputVal = (inpVal[nNum])
        for lNum in range(self.nHid):
            for nNum in range(self.hidLaySize[lNum] - 1):
                self.hidLayers[lNum][nNum].calcVal(self.inpLayer if lNum == 0 else self.hidLayers[lNum - 1])
        for nNum in range(self.nOut):
            self.outLayer[nNum].calcVal(self.hidLayers[-1])

    def backProp(self, targetVal):
        assert len(targetVal) == self.nOut

        for neu in self.outLayer:
            neu.sigm = neu.outputVal * (1 - neu.outputVal) * (targetVal[neu.myInd] - neu.outputVal)

        for lNum in range(self.nHid - 1, -1, -1):
            for curNeu in self.hidLayers[lNum]:
                nextLayer = self.outLayer if lNum == self.nHid - 1 else self.hidLayers[lNum + 1]
                summ = 0
                for nextNeu in nextLayer:
                    summ += nextNeu.sigm * nextNeu.connections[curNeu.myInd].w

                curNeu.sigm = curNeu.outputVal * (1 - curNeu.outputVal) * summ

        for lNum in range(self.nHid):
            prevLayer = self.inpLayer if lNum == 0 else self.hidLayers[lNum - 1]
            for curNeu in self.hidLayers[lNum]:
                curNeu.updateConnections(prevLayer)
                
        for neu in self.outLayer:
            neu.updateConnections(self.hidLayers[-1])

        self.applyUpdates()

    def applyUpdates(self):
        for hidLayer in self.hidLayers:
            for neu in hidLayer:
                neu.applyUpdates()
        for neu in self.outLayer:
            neu.applyUpdates()

    def out(self):
        for neu in self.inpLayer:
            print(neu.outputVal, end=" ")
        print()
        for lNum in range(self.nHid):
            for neu in self.hidLayers[lNum]:
                print(neu.outputVal, end=" ")
            print()
        for neu in self.outLayer:
            print(neu.outputVal, end=" ")
        print()

    def learn(self, inp, tar):
        self.feedForw(inp)
        self.backProp(tar)

    def test(self, inp, debug=True):
        if debug:
            print("Testing on", inp)    
        self.feedForw(inp)
        if debug:
            print("Got result", [x.outputVal for x in self.outLayer])
        return [x.outputVal for x in self.outLayer]

    def backup(self, filename="./nets/backup.txt"):
        with open(filename, "w") as f:
            f.write(str(self.nInp - 1) + " ")
            for size in self.hidLaySize:
                f.write(str(size - 1) + " ")
            f.write(str(self.nOut) + "\n")
            f.write("\n")
            
            for hidLayer in self.hidLayers:
                for neu in hidLayer:
                    for connection in neu.connections: 
                        f.write(str(connection.w) + " ")
                    f.write("\n")
                f.write("\n")
            for neu in self.outLayer:
                for connection in neu.connections: 
                    f.write(str(connection.w) + " ")
                f.write("\n")


def randomWeight():
    n = 1000
    return random.randint(0, n) / (100 * n)

