import sys
import numpy as np
sys.path.append('../edge/')
import cv2
import matplotlib.pyplot as plt
from canny import Canny
from contours import ContourDetector

sample = 1

inputfile = f"./input/test{sample}.jpeg"
outputfile = f"./output/res{sample}.png"

image = np.array(cv2.imread(inputfile, cv2.IMREAD_GRAYSCALE))
canny = Canny(image, sd=1.4, kernel_size=7)
res = canny()
plt.imshow(res, cmap='gray')
cv2.imwrite(outputfile, res)

cd = ContourDetector(res)
_, contours = cd.traverse()
contour_image = cd.plot_contour()
plt.imshow(contour_image)
cv2.imwrite(f"./output/contours{sample}.png", contour_image)
