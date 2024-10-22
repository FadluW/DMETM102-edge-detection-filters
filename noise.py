import numpy as np
import matplotlib.pyplot as plt
import time
import os
import cv2
import math

# https://medium.com/@ms_somanna/guide-to-adding-noise-to-your-data-using-python-and-numpy-c8be815df524
# https://www.askpython.com/python/examples/adding-noise-images-opencv
# https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_star_poly.html#sphx-glr-gallery-lines-bars-and-markers-scatter-star-poly-py
# https://github.com/scikit-image/scikit-image/blob/main/skimage/util/noise.py#L39-L233

# Types for type-hinting
from cv2.typing import MatLike

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def printCompletedStarting(startMs: int) -> None:
    print(f"Completed in: {bcolors.BOLD}{int(time.time() * 1000) - startMs}{bcolors.ENDC}ms\n")

def getImageHistogram(image: MatLike) -> None:
    # https://stackoverflow.com/questions/22159160/python-calculate-histogram-of-image
    vals = image.mean(axis=2).flatten()
    return np.histogram(vals, range(257))

def saveImageHistograms(originalFileName: str):
    count = 6

    # plt.xlim([-0.5, 255.5])
    fig, axs = plt.subplots(math.ceil(count / 2.0), 2, sharex=True, sharey=True, layout="constrained")

    counts, bins = getImageHistogram(cv2.imread(originalFileName))
    axs[0][0].bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
    axs[0][0].set_title(originalFileName)

    figureCount = 1
    for filename in os.scandir("./results"):
        if not filename.is_file():
            continue

        counts, bins = getImageHistogram(cv2.imread(filename))
        axs[figureCount % (count // 2)][figureCount // (count // 2)].bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
        axs[figureCount % (count // 2)][figureCount // (count // 2)].set_title(filename.name)
        figureCount += 1

    plt.savefig("image-histograms.png")

def addSaltAndPepperNoise(originalImage: MatLike, noiseRatio = 0.2) -> MatLike:
    print(f"{bcolors.HEADER}Adding salt and pepper...{bcolors.ENDC}")
    startMs = int(time.time() * 1000)

    noisyImage = originalImage.copy()

    noisyPixelCount = int(noiseRatio * noisyImage.size)
    random_indices = np.random.choice(noisyImage.size, noisyPixelCount)
    noise = np.random.choice([originalImage.min(), originalImage.max()], noisyPixelCount)

    noisyImage.flat[random_indices] = noise
    
    printCompletedStarting(startMs)
    return noisyImage

def addGaussianNoise(originalImage: MatLike, mean=0, sigma=25) -> MatLike:
    print(f"{bcolors.HEADER}Adding gaussian noise...{bcolors.ENDC}")
    startMs = int(time.time() * 1000)

    noise = np.random.normal(mean, sigma, originalImage.shape).astype(np.uint8)
    noisyImage = np.clip(originalImage + noise, 0, 255).astype(np.uint8)

    printCompletedStarting(startMs)
    return noisyImage

def addPoissonNoise(originalImage: MatLike) -> MatLike:
    print(f"{bcolors.HEADER}Adding poisson noise...{bcolors.ENDC}")
    startMs = int(time.time() * 1000)

    uniqueValues = len(np.unique(originalImage))
    nextPowerOfTwo = 2 ** np.ceil(np.log2(uniqueValues))
    noisyImage = np.random.poisson(originalImage * nextPowerOfTwo) / float(nextPowerOfTwo)

    printCompletedStarting(startMs)
    return noisyImage

def addRandomNoise(originalImage: MatLike, intensity=75) -> MatLike:
    print(f"{bcolors.HEADER}Adding random noise...{bcolors.ENDC}")
    startMs = int(time.time() * 1000)
    
    noisyImage = originalImage.copy()
    
    noise = np.random.randint(-intensity, intensity + 1, noisyImage.shape)
    noisyImage = np.clip(noisyImage + noise, 0, 255).astype(np.uint8)

    printCompletedStarting(startMs)
    return noisyImage