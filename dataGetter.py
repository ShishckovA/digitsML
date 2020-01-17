import os
from PIL import Image
from utils import CropAndCompressToArray

f = open("./data/data.txt", "w")
for i in range(10): 
    imageNames = os.listdir("./examples/%d" % i)
    for imageName in imageNames:
        print(i, imageName)
        path = './examples/%d/%s' % (i, imageName)
        ar = CropAndCompressToArray(path)
        f.write(str(i) + ":")
        f.write(" ".join(map(str, ar)))
        f.write("\n")
f.close()