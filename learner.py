import random
from net import Net
from utils import sayMeNum, fromFile, readTests 

net = Net([3, 10, 10, 1])


net.feedForw([0, 0, 1])
for neu in net.outLayer:
    print(neu.inputVal)
# print(len(net.outLayer[0].connections))

n = 100000
i = 0
for i in range(5):
    i += 1
    for a in range(2):
        for b in range(2):
            for c in range(2):
                inp = [a, b, c]
                ans = a ^ b ^ c
                if ans:
                    out = [1]
                else:
                    out = [0]
                net.learn(inp, out)
    if i % 10000 == 0:
        net.backup("out.txt")
        sq = 0
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    inp = [a, b, c]
                    ans = a ^ b ^ c
                    res = net.test(inp, debug=False)[0]
                    print(res, inp)
                    sq += (ans - res) ** 2
        print(sq)


net.backup("tmp")
net1 = fromFile("tmp")

