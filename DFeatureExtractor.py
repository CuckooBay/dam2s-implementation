import cv2
import numpy as np

from skimage.feature import hog
from skimage.feature import local_binary_pattern


class DFeatureExtractor:
    def __init__(self, numPoints=24, radius=3):
        self.numPoints = numPoints
        self.radius = radius

    def __call__(self, images):
        fds = np.array([ self.getFeature(self.convertColorToGrayscale(image)) for image in images ])
        return fds

    def convertColorToGrayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def getFeature(self, image):
        hog_fd = self.getHOGDES(image).flatten()
        lbp_fd = self.getLBPDES(image).flatten()
        fd = np.concatenate((hog_fd, lbp_fd), axis=None)
        print(hog_fd)
        print(lbp_fd)
        return fd
    
    def getHOGDES(self, image):
        fd = hog(image, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=False, feature_vector=True, channel_axis=None)
        return fd

    def getLBPDES(self, image, eps=1e-7):
        lbp = local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints+3), range=(0, self.numPoints+2))
        fd = hist.astype("float")
        fd /= (hist.sum() + eps)
        return fd


