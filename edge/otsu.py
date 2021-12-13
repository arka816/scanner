import numpy as np

def otsu(image):
    X, Y = image.shape
    minPixel, maxPixel = image.min(), image.max()

    hist = np.zeros(256, dtype=np.float64)
    for i in range(X):
        for j in range(Y):
            hist[image[i][j]] += 1

    
    w0, w1 = hist[minPixel], sum(hist[minPixel+1:maxPixel + 1])
    m0, m1 = minPixel, sum(np.arange(minPixel+1, maxPixel+1) * hist[minPixel+1:maxPixel+1])/w1
    maxSigma = sigma = w0 * w1 * (m1 - m0)**2
    threshold = 0
    
    for t in range(minPixel+1, maxPixel):
        m0 = (w0 * m0 + t*hist[t]) / (w0 + hist[t])
        m1 = (w1 * m1 - t*hist[t]) / (w1 - hist[t])
        w0 += hist[t]
        w1 -= hist[t]
        
        sigma = w0 * w1 * (m1 - m0)**2
        if sigma > maxSigma:
            maxSigma = sigma
            threshold = t
            
    threshold += 10
    binarized_image = 255*(image < threshold)
    return binarized_image, threshold
