import random
from net import Net
from utils import fromFile, sayMeNum, CropAndCompressToArray    
from PIL import Image
  
net = fromFile("./nets/myML.txt")

im = Image.open('input.bmp')
inpArr = CropAndCompressToArray("input.bmp")
print("Я считаю, что это %d с уверенностью %f" % sayMeNum(net, inpArr))

