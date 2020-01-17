from net import Net, Connection

def readTests(filename="out_tr.txt"):
    f = open(filename, "r")
    lines = f.readlines()
    tests = []
    for line in lines:
        ans, arr = line.split(":")
        ans = int(ans)
        arr = list(map(float, arr.split()))
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

def fromFile(filename="./nets/backup.txt"):
    with open(filename, "r") as f:
        topology = list(map(int, f.readline().split()))

        net = Net(topology)
        f.readline()
        for hidLayer in net.hidLayers:
            for neu in hidLayer:
                values = f.readline().split()
                for prevNeuInd in range(neu.numInp):
                    neu.connections[prevNeuInd] = Connection(float(values[prevNeuInd]), 0)
            f.readline()
        for neu in net.outLayer:
            values = f.readline().split()
            for prevNeuInd in range(neu.numInp):
                neu.connections[prevNeuInd] = Connection(float(values[prevNeuInd]), 0)
    return net

def sayMeNum(net, inp):
    res = net.test(inp, debug=False)
    resAr = [(res[x], x) for x in range(10)]
    resAr.sort(reverse=True)
    summ = 0
    for elem in resAr:
        summ += elem[0]
    return resAr[0][1], resAr[0][0] / summ

def CropAndCompressToArray(imgName):
    from PIL import Image
    im = Image.open(imgName)
    im = im.convert('1')
    w, h = im.size
    inpArr = [[0 for i in range(w)] for j in range(h)]
    minI = h
    maxI = 0
    minJ = w
    maxJ = 0
    for i in range(h):
        for j in range(w):
            col = im.getpixel((j, i))
            if col < 255:
                if i > maxI:
                    maxI = i
                if i < minI:
                    minI = i 
                if j > maxJ:
                    maxJ = j 
                if j < minJ:
                    minJ = j
    im = im.crop((minJ, minI, maxJ, maxI))
    size = 28, 28
    im.thumbnail(size, Image.ANTIALIAS)
    arr = [[0 for i in range(size[0])] for i in range(size[1])]
    for i in range(im.size[0]): 
        for j in range(im.size[1]):
            col = 1 - im.getpixel((i, j)) / 255
            arr[j][i] = col

    out = []
    for elem in arr:
        out += elem

    return out
