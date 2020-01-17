import random
from net import Net
from utils import sayMeNum, fromFile, readTests

def myKey(a):
    if a["tested"] == 0:
        return 1
    return a["success"] / a["tested"]
        

# net = Net([784, 70, 70, 10])
net = fromFile("./nets/ myML.txt")


print("Reading data...")
train = readTests("./data/data.txt")
random.shuffle(train)
l = len(train) // 2
test = train[l:]
train = train[:l]

# print(len(test), len(train))
# train = train[:150]
# train = readTests()
# test = readTests("out_te_All.txt")
# print("Read")

n = 3000
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
    print("\nEpoche ended!")
    e += 1
    net.backup("./nets/myML.txt")

    print("Starting testing")
    nCorrectAns = 0
    nTests = 5000
    byDigits = [{"tested" : 0, "success" : 0, "num" : i, "sim" : [0 for i in range(10)]} for i in range(10)]

    for i in range(nTests):
        imageInd = random.randint(0, len(test) - 1)
        correctAns, inp = test[imageInd]
        neuronNetAns, neuronNetConf = sayMeNum(net, inp)
        byDigits[correctAns]["tested"] += 1
        if neuronNetAns == correctAns:
            nCorrectAns += 1
            byDigits[correctAns]["success"] += 1
        else:
            byDigits[correctAns]["sim"][neuronNetAns] += 1
    byDigits.sort(key=myKey)
    print("Worst number:")
    for i in range(3):
        el = byDigits[i]
        sim = el["sim"]
        print("%d (%f%% correct, %d of %d)" % (el["num"], el["success"] / el["tested"] * 100, el["success"], el["tested"]))
        for i in range(10):
            print(i, sim[i] / sum(sim) * 100)
        print()

    print("Testing complete, success - %s%%" % (nCorrectAns * 100 / nTests))
    print()


