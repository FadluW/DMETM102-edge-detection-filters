import numpy as np
import os
import cv2

from util import MatLike, bcolors, printCompletedStarting, getCurrTimeMs

# https://medium.com/@ms_somanna/guide-to-adding-noise-to-your-data-using-python-and-numpy-c8be815df524
# https://www.askpython.com/python/examples/adding-noise-images-opencv
# https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_star_poly.html#sphx-glr-gallery-lines-bars-and-markers-scatter-star-poly-py
# https://github.com/scikit-image/scikit-image/blob/main/skimage/util/noise.py#L39-L233


def addSaltAndPepperNoise(originalImage: MatLike, noiseRatio = 0.2) -> MatLike:
    print(f"{bcolors.HEADER}Adding salt and pepper...{bcolors.ENDC}")
    startMs = getCurrTimeMs()

    noisyImage = originalImage.copy()

    noisyPixelCount = int(noiseRatio * noisyImage.size)
    random_indices = np.random.choice(noisyImage.size, noisyPixelCount)
    noise = np.random.choice([originalImage.min(), originalImage.max()], noisyPixelCount)

    noisyImage.flat[random_indices] = noise
    
    printCompletedStarting(startMs)
    return noisyImage


def addGaussianNoise(originalImage: MatLike, mean=0, sigma=25) -> MatLike:
    print(f"{bcolors.HEADER}Adding gaussian noise...{bcolors.ENDC}")
    startMs = getCurrTimeMs()

    noise = np.random.normal(mean, sigma, originalImage.shape).astype(np.uint8)
    noisyImage = np.clip(originalImage + noise, 0, 255).astype(np.uint8)

    printCompletedStarting(startMs)
    return noisyImage


def addPoissonNoise(originalImage: MatLike) -> MatLike:
    print(f"{bcolors.HEADER}Adding poisson noise...{bcolors.ENDC}")
    startMs = getCurrTimeMs()

    uniqueValues = len(np.unique(originalImage))
    nextPowerOfTwo = 2 ** np.ceil(np.log2(uniqueValues))
    noisyImage = np.random.poisson(originalImage * nextPowerOfTwo) / float(nextPowerOfTwo)

    printCompletedStarting(startMs)
    return noisyImage


def addRandomNoise(originalImage: MatLike, intensity=75) -> MatLike:
    print(f"{bcolors.HEADER}Adding random noise...{bcolors.ENDC}")
    startMs = getCurrTimeMs()
    
    noisyImage = originalImage.copy()
    
    noise = np.random.randint(-intensity, intensity + 1, noisyImage.shape)
    noisyImage = np.clip(noisyImage + noise, 0, 255).astype(np.uint8)

    printCompletedStarting(startMs)
    return noisyImage

def addAndSaveAllNoise(originalImage: MatLike, targetDirectory: str) -> None:
    if not os.path.exists(targetDirectory):
        os.mkdir(targetDirectory)

    cv2.imwrite(f"{targetDirectory}noise-salt-and-peper.jpg", addSaltAndPepperNoise(originalImage))
    cv2.imwrite(f"{targetDirectory}noise-gaussian.jpg", addGaussianNoise(originalImage))
    cv2.imwrite(f"{targetDirectory}noise-poisson.jpg", addPoissonNoise(originalImage))
    cv2.imwrite(f"{targetDirectory}noise-random.jpg", addRandomNoise(originalImage))