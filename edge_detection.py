import cv2
import os

from util import MatLike, bcolors, getCurrTimeMs, printCompletedStarting

# https://learnopencv.com/edge-detection-using-opencv/#sobel-edge
def applySobelEdgeDetection(originalImage: MatLike, kernelSize=5, blurred=True) -> MatLike:
    print(f"{bcolors.HEADER}Applying Sobel edge detection...{bcolors.ENDC}")
    startMs = getCurrTimeMs()

    image = originalImage.copy()
    if (blurred):
        print(f"{bcolors.HEADER}with blur...{bcolors.ENDC}")
        image = cv2.GaussianBlur(image, (9,9), sigmaX=0, sigmaY=0)

    edgesImage = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=kernelSize)

    printCompletedStarting(startMs)
    return edgesImage


def applyCannyEdgeDetection(originalImage: MatLike, lowThreshold=100, highThreshold=200, blurred=True):
    print(f"{bcolors.HEADER}Applying Canny edge detection...{bcolors.ENDC}")
    startMs = getCurrTimeMs()

    image = originalImage.copy()
    if (blurred):
        print(f"{bcolors.HEADER}with blur...{bcolors.ENDC}")
        image = cv2.GaussianBlur(image, (9,9), sigmaX=0, sigmaY=0)

    edgesImage = cv2.Canny(image=image, threshold1=lowThreshold, threshold2=highThreshold)

    printCompletedStarting(startMs)
    return edgesImage


def applyAndSaveAllEdgeDetection(originalImage: MatLike, targetDirectory: str) -> None:
    if not os.path.exists(targetDirectory):
        os.mkdir(targetDirectory)
    
    cv2.imwrite(f"{targetDirectory}edge-sobel-5-blurred.jpg", applySobelEdgeDetection(originalImage, blurred=True))
    cv2.imwrite(f"{targetDirectory}edge-sobel-5.jpg", applySobelEdgeDetection(originalImage, blurred=False))
    cv2.imwrite(f"{targetDirectory}edge-canny-blurred.jpg", applyCannyEdgeDetection(originalImage))