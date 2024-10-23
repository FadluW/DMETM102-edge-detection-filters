import numpy as np
import cv2
import os
from util import MatLike, bcolors, printCompletedStarting, getCurrTimeMs

# https://learnopencv.com/image-filtering-using-convolution-in-opencv/
def applyAverageFilter(originalImage: MatLike, filterSize=3) -> MatLike:
    print(f"{bcolors.HEADER}Applying average filter...{bcolors.ENDC}")
    print(f"{bcolors.HEADER}with filter size: {filterSize}x{filterSize}...{bcolors.ENDC}")
    startMs = getCurrTimeMs()

    kernel = np.ones((filterSize, filterSize), np.float32) / filterSize ** 2
    blurredImage = cv2.filter2D(src=originalImage, ddepth=-1, kernel=kernel)

    printCompletedStarting(startMs)
    return blurredImage


def applyAndSaveAllAverageFilter(originalImage: MatLike, targetDirectory: str) -> None:
    if not os.path.exists(targetDirectory):
        os.mkdir(targetDirectory)
    
    cv2.imwrite(f"{targetDirectory}average-filter-3.jpg", applyAverageFilter(originalImage, filterSize=3))
    cv2.imwrite(f"{targetDirectory}average-filter-5.jpg", applyAverageFilter(originalImage, filterSize=5))
    cv2.imwrite(f"{targetDirectory}average-filter-9.jpg", applyAverageFilter(originalImage, filterSize=9))