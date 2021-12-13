import numpy as np
import cv2

class ContourDetector:
    def __init__(self, image):
        self.image = image
        self.dir_x = [1, 0, -1, 0, 1, 1, -1, -1]
        self.dir_y = [0, 1, 0, -1, 1, -1, 1, -1]
        self.X, self.Y = image.shape
        self.visited = np.zeros((self.X, self.Y))
        self.objCount = 0
        self.contours = list()
        self.characters = list()
        
    def detectLines(self):
        for contour in self.contours:
            l = len(contour)
            if l > 0:
                X_mean, Y_mean = sum([c[0] for c in contour])/l, sum([c[1] for c in contour])/l
                self.characters.append((X_mean, Y_mean))
                
        return [c[1] for c in self.characters], [c[0] for c in self.characters]
        
    def bounds(self):
        bounds = list()
        for contour in self.contours:
            if len(contour) > 0:
                minX, minY = min(contour)
                maxX, maxY = max(contour)
                bounds.append((minX, minY, maxX, maxY))
            
        return bounds

    def bfs(self, i, j):
        q = list()
        q.append((i, j))
        self.visited[i, j] = 1
        self.image[i, j] = 100
        contour =[(i, j)]
        
        while len(q) > 0:
            x, y = q.pop(0)
            for dx, dy in zip(self.dir_x, self.dir_y):
                new_x, new_y = x + dx, y + dy
                if new_x >= 0 and new_x < self.X and new_y >= 0 and new_y < self.Y:
                    if self.visited[new_x, new_y] == 0 and self.image[new_x, new_y] == 255:
                        self.visited[new_x, new_y] = 1
                        q.append((new_x, new_y))
                        ###self.image[new_x, new_y] = 100
                        contour.append((new_x, new_y))
                        
        return contour

    def plot_contour(self, limit = 4, contour_color = (0, 255, 0)):
        # plot top 4 longest contours
        self.contours = sorted(self.contours, key= lambda c : -len(c))
        contours = self.contours[:4]
        original_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        for contour in contours:
            for point in contour:
                original_image[point[0]][point[1]] = contour_color
        return original_image
    
    def traverse(self):
        for i in range(self.X):
            for j in range(self.Y):
                if self.image[i, j] == 255 and self.visited[i, j] == 0:
                    self.contours.append(self.bfs(i, j))
                    self.objCount += 1
                    
        return self.image, self.contours