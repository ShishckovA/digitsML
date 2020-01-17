import random
from math import tanh, cosh, exp

alpha = 0.8

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
        self.error = 0
        for nNum in range(self.numInp):
            self.connections.append(Connection(w=randomWeight(), dw=0))

    def activationFunc(self, x):
        # return (x / (abs(x) + 1) + 1) / 2
        return 1 / (1 + exp(-x))

    def dActivationFunc(self, x):
        # return 0.5 / (1 + abs(x)) ** 2
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
        for nNum in range(self.numInp):
            self.connections[nNum].dw += alpha * self.error * prevLayer[nNum].outputVal

    def applyUpdates(self):
        for nNum in range(self.numInp):
            self.connections[nNum].w += self.connections[nNum].dw
            self.connections[nNum].dw = 0

class Net:
    def __init__(self, nInp, nHid, nOut):
        self.inpLayer = []
        self.hidLayer = []
        self.outLayer = []
        self.nInp = nInp + 1
        self.nHid = nHid + 1
        self.nOut = nOut

        for nNum in range(self.nInp):
            self.inpLayer.append(Neuron(nNum, 0))
        for nNum in range(self.nHid):
            self.hidLayer.append(Neuron(nNum, self.nInp))
        for nNum in range(self.nOut):
            self.outLayer.append(Neuron(nNum, self.nHid))


    def feedForw(self, inpVal):
        assert len(inpVal) == self.nInp - 1
        for nNum in range(self.nInp - 1):
            self.inpLayer[nNum].outputVal = (inpVal[nNum])
        for nNum in range(self.nHid - 1):
            self.hidLayer[nNum].calcVal(self.inpLayer)
        for nNum in range(self.nOut):
            self.outLayer[nNum].calcVal(self.hidLayer)

    def backProp(self, targetVal):
        assert len(targetVal) == self.nOut

        for nNum in range(self.nOut):
            error = (targetVal[nNum] - self.outLayer[nNum].outputVal) * self.outLayer[nNum].dActivationFunc(self.outLayer[nNum].inputVal)
            self.outLayer[nNum].error = error

        for nNum in range(self.nHid):
            sigm_in = 0
            for nextNNum in range(self.nOut):
                sigm_in += self.outLayer[nextNNum].error * self.outLayer[nextNNum].connections[nNum].w

            self.hidLayer[nNum].error = sigm_in * self.hidLayer[nNum].dActivationFunc(self.hidLayer[nNum].inputVal)

        for hidN in self.hidLayer:
            hidN.updateConnections(self.inpLayer)
        for outN in self.outLayer:
            outN.updateConnections(self.hidLayer)

    def applyUpdates(self):
        for hidN in self.hidLayer:
            hidN.applyUpdates()
        for outN in self.outLayer:
            outN.applyUpdates()

    def out(self):
        for f in self.inpLayer:
            print(f.outputVal, end=" ")
        print()
        for s in self.hidLayer:
            print(s.outputVal, end=" ")
        print()
        for t in self.outLayer:
            print(t.outputVal, end=" ")
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

    def backup(self):
        with open("backup.txt", "w") as f:
            f.write(str(self.nInp - 1) + " " + str(self.nHid - 1) + " " + str(self.nOut) + "\n")

            f.write("\n")
            for neuron in self.hidLayer:
                for connection in neuron.connections: 
                    f.write(str(connection.w) + " ")
                f.write("\n")
            f.write("\n")
            for neuron in self.outLayer:
                for connection in neuron.connections: 
                    f.write(str(connection.w) + " ")
                f.write("\n")

def fromFile():
    filename = "backup.txt"
    with open(filename, "r") as f:
        nInp, nHid, nOut = map(int, f.readline().split())
        net = Net(nInp, nHid, nOut)
        f.readline()
        for nNum in range(nHid + 1):
            values = f.readline().split()
            for prevN in range(nInp + 1):
                net.hidLayer[nNum].connections[prevN] = Connection(float(values[prevN]), 0)
        f.readline()
        for nNum in range(nOut):
            values = f.readline().split()
            for prevN in range(nHid + 1):
                net.outLayer[nNum].connections[prevN] = Connection(float(values[prevN]), 0)
    return net

def randomWeight():
    n = 1000
    return random.randint(0, n) / (130 * n)
    # return random.randint(0, n) / (130 * n)

def readTr():
    f = open("out_tr.txt", "r")
    lines = f.readlines()
    tests = []
    for line in lines:
        ans, arr = line.split()
        ans = int(ans)
        arr = list(map(int, list(arr)))
        tests.append((ans, arr))
    return tests

def readTe():
    f = open("out_te_All.txt", "r")
    lines = f.readlines()
    tests = []
    for line in lines:
        ans, arr = line.split()
        ans = int(ans)
        arr = list(map(int, list(arr)))
        tests.append((ans, arr))
    return tests

def testImage(i):
    print("Test with image #%d" % (i + 1))
    ans, inp = test[i]
    print("True answer is %d" % ans)
    res = net.test(inp, debug=False)
    print("Got answer", res)
    resAr = [(res[x], x) for x in range(10)]
    resAr.sort(reverse=True)
    print("I think the answer is %d, confidense - %f" % (resAr[0][1], resAr[0][0]))
    print()
    return resAr[0][1], ans

net = Net(784, 100, 10)
# net = fromFile()

print("Reading data...")
train = readTr()
test = readTe()
print("Read")

n = 500
e = 0
go = True
while go:
    print("=========================")
    for i in range(n):
        trainExercice = random.choice(train)
        print("\rEpoche %d, test %d" % (e + 1, i + 1), end="")
        ans, inp = trainExercice
        ansArr = [0 for i in range(10)]
        ansArr[ans] = 1
        net.learn(inp, ansArr)
        net.applyUpdates()
    print("\nEpoche ended!")
    e += 1
    net.backup()

    print("Starting testing")
    nCorrectAns = 0
    nTests = 500
    for i in range(nTests):
        imageInd = random.randint(0, len(test) - 1)
        neuronNetAns, correctAns = testImage(imageInd)
        if neuronNetAns == correctAns:
            nCorrectAns += 1
    print("Testing complete, success - %s%%" % (nCorrectAns * 100 / nTests))
    print()
