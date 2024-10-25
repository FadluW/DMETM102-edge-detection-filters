import matplotlib.pyplot as plt
import math
import numpy as np
import time
import os
import cv2

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


def getCurrTimeMs() -> int:
    return int(time.time() * 1000)


def printCompletedStarting(startMs: int) -> None:
    print(f"Completed in: {bcolors.BOLD}{int(time.time() * 1000) - startMs}{bcolors.ENDC}ms\n")


def getImageHistogram(image: MatLike) -> None:
    vals = image.mean(axis=2).flatten()
    return np.histogram(vals, range(257))

def saveImageHistograms(originalFileName: str):
    print(f"{bcolors.HEADER}Generating image histograms...{bcolors.ENDC}")
    startMs = int(time.time() * 1000)

    count = float(len(os.listdir("./results"))) + 1
    if (count < 2):
        print(f"{bcolors.FAIL}No images in result!{bcolors.ENDC}")
        return

    fig, axs = plt.subplots(math.ceil(count / 2), 2, sharex=True, layout="compressed", dpi=200, figsize=[9.6, 6.4*count/8])

    counts, bins = getImageHistogram(cv2.imread(originalFileName))
    axs[0][0].bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
    axs[0][0].set_title(originalFileName)

    figureCount = 1
    for filename in os.scandir("./results"):
        if not filename.is_file():
            continue

        counts, bins = getImageHistogram(cv2.imread(filename))
        xIndex = int(figureCount % (count // 2))
        yIndex = int(figureCount // (count // 2))

        axs[xIndex][yIndex].bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
        axs[xIndex][yIndex].set_title(filename.name)
        figureCount += 1

    print(f"{bcolors.OKGREEN}Generated {figureCount} image histograms!{bcolors.ENDC}")
    plt.savefig("image-histograms.png", dpi=200)
    printCompletedStarting(startMs)