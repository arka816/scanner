import sys
import numpy as np
sys.path.append('../edge/')
import cv2
import matplotlib.pyplot as plt
from canny import Canny
print(cv2.__version__)

image = np.array(cv2.imread('test1.jpeg', cv2.IMREAD_GRAYSCALE))
canny = Canny(image, sd=1.4, kernel_size=7)
res = canny()
plt.imshow(res, cmap='gray')
cv2.imwrite("res1.png", res)
