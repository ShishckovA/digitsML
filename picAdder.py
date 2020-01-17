import shutil
import os

filename = "input.bmp"
n = 15
while 1:
    a = int(input())
    files = (os.listdir("./examples/%d" % a))
    newN = 1
    while "%d.bmp" % newN in files:
        newN += 1
    shutil.copyfile(filename, "./examples/%d/%d.bmp" % (a, newN))
    print(filename, "./examples/%d/%d.bmp" % (a, newN))
