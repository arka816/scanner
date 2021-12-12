import sys
import numpy as np
import time
from filters import gaussian, prewitt, sobel


class Canny:
    THRESHOLD_HIGH = 0.15
    THRESHOLD_LOW  = 0.05

    STRONG_VAL = 255
    WEAK_VAL   = 75

    HYSTERESIS_KERNEL_SIZE = 5

    def __init__(self, bitmap=None, sd=1, kernel_size=5, edge_op='sobel') -> None:
        self.bitmap = bitmap
        self.edge_op = edge_op
        self.sd = sd
        self.kernel_size = kernel_size

    def __call__(self):
        # Step 1:   Gaussian Kernel
        start = time.time()
        filtered_bitmap = gaussian(self.bitmap, kernel_size=self.kernel_size, sd=self.sd)
        end = time.time()
        print('step 1', 'clocked at: ', end - start)

        # Step 2:   intensity gradient
        start = time.time()
        if self.edge_op == 'prewitt':
            Gx, Gy = prewitt(filtered_bitmap)
        elif self.edge_op == 'sobel':
            Gx, Gy = sobel(filtered_bitmap)
        G, theta = self.polar_intensity(Gx, Gy)
        end = time.time() 
        print('step 2', 'clocked at: ', end - start)       
        
        # Step 3:   lower bound thresholding / non-maximal suppression
        start = time.time()
        suppressed_image = self.non_maximal_suppression(G, theta)
        end = time.time()
        print('step 3', 'clocked at: ', end - start)

        # Step 4:   dual threshold
        start = time.time()
        thresholded_image = self.threshold(suppressed_image)
        end = time.time()
        print('step 4', 'clocked at: ', end - start)

        # Step 5:   edge tracking by hysteresis
        start = time.time()
        hysteresis_image = self.hysteresis(thresholded_image)
        end = time.time()
        print('step 5', 'clocked at: ', end - start)

        return hysteresis_image



    def polar_intensity(self, Gx, Gy):
        G = np.hypot(Gx, Gy)
        G = G / G.max() * 255
        theta = np.arctan2(Gy, Gx) * 180 / np.pi

        for row in theta:
            for i in range(len(row)):
                angle = row[i]
                if (angle >= -22.5 and angle <= 22.5) or angle >= 157.5 or angle <= -157.5:
                    row[i] = 0
                elif (angle >= 67.5 and angle <= 112.5) or (angle <= -67.5 and angle >= -112.5):
                    row[i] = 90
                elif (angle >= 22.5 and angle <= 67.5) or (angle <= -112.5 and angle >= -157.5):
                    row[i] = 45
                else:
                    row[i] = 135

        return G, theta

    def non_maximal_suppression(self, G, theta):
        M, N = G.shape
        Z = np.zeros((M, N), dtype=np.int32)
        for i in range(1, M-1):
            for j in range(1, N-1):
                try:
                    p, n = 255, 255
                    angle = theta[i, j]
                    if angle == 0:
                        p = G[i, j+1]
                        n = G[i, j-1]
                    elif angle == 45:
                        p = G[i+1, j-1]
                        n = G[i-1, j+1]
                    elif angle == 90:
                        p = G[i+1, j]
                        n = G[i-1, j]
                    else:
                        p = G[i-1, j-1]
                        n = G[i+1, j+1]
                        
                    if G[i, j] >= p and G[i, j] >= n:
                        Z[i, j] = G[i, j]
                    else:
                        Z[i, j] = 0
                    
                except:
                    pass
        return Z

    def threshold(self, image):
        high = self.THRESHOLD_HIGH * image.max()
        low = self.THRESHOLD_LOW * image.max()
        
        M, N = image.shape
        res = np.zeros((M, N), dtype=np.int32)
        
        strong_i, strong_j = np.where(image >= high)
        weak_i, weak_j = np.where((image < high) & (image >= low))
        # zero_i, zero_j = np.where(image < low)
        
        res[strong_i, strong_j] = self.STRONG_VAL
        res[weak_i, weak_j] = self.WEAK_VAL
        
        return res

    def hysteresis(self, image):
        c = self.HYSTERESIS_KERNEL_SIZE // 2
        M, N = image.shape
        for i in range(1, M-1):
            for j in range(1, N-1):
                try:
                    if image[i, j] == self.WEAK_VAL:
                        isEdge = False
                        for k in range(i-c, i+c+1):
                            for l in range(j-c, j+c+1):
                                if image[k, l] == self.STRONG_VAL:
                                    isEdge = True
                                    break
                        if isEdge == True:
                            image[i, j] = self.STRONG_VAL
                        else:
                            image[i, j] = 0
                except:
                    pass
                
        return image


if __name__ == "__main__":
    try:
        filename = sys.argv[1]
    except:
        raise Exception('input filename not given')
    
    if len(sys.argv) > 2:
        if sys.argv[2] == '-o':
            if len(sys.argv) > 3:
                output_filename = sys.argv[3]
            else:
                raise Exception('output filename not given')
        else:
            output_filename = "result.png"
    else:
        output_filename = "result.png"

    import cv2
    import matplotlib.pyplot as plt

    image = np.array(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))
    canny = Canny(image, sd=1.4, kernel_size=7)
    res = canny()
    plt.imshow(res, cmap='gray')
    cv2.imwrite(output_filename, res)
