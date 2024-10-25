import numpy as np
import cv2
import os
from util import MatLike, bcolors, printCompletedStarting, getCurrTimeMs


def applyAverageFilter(originalImage: MatLike, filterSize=3) -> MatLike:
    print(f"{bcolors.HEADER}Applying average filter...{bcolors.ENDC}")
    print(f"{bcolors.HEADER}with filter size: {filterSize}x{filterSize}...{bcolors.ENDC}")
    startMs = getCurrTimeMs()

    kernel = np.ones((filterSize, filterSize), np.float32) / filterSize ** 2
    blurredImage = cv2.filter2D(src=originalImage, ddepth=-1, kernel=kernel)

    printCompletedStarting(startMs)
    return blurredImage


def applyToNoisyAndSaveAllAverageFilter(targetDirectory: str) -> None:
    print(f"{bcolors.HEADER}Passing noisy images through average filters:\n{bcolors.ENDC}")

    if not os.path.exists(targetDirectory):
        print(f"{bcolors.FAIL}Target directory doesn't exist!{bcolors.ENDC}")
    
    for noisyFile in os.scandir(targetDirectory):
        if not noisyFile.name.startswith("noise"):
            continue
        
        noisyImage = cv2.imread(noisyFile)
        cv2.imwrite(f"{targetDirectory}average-filter-3-{noisyFile.name}", applyAverageFilter(noisyImage, filterSize=3))
        cv2.imwrite(f"{targetDirectory}average-filter-5-{noisyFile.name}", applyAverageFilter(noisyImage, filterSize=5))
        cv2.imwrite(f"{targetDirectory}average-filter-9-{noisyFile.name}", applyAverageFilter(noisyImage, filterSize=9))