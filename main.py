import cv2
import os

from noise import bcolors, addSaltAndPepperNoise, addGaussianNoise, addPoissonNoise, addRandomNoise, saveImageHistograms

# Modify file name accordingly
originalFileName = "original.jpg"
targetDirectory = "./results"

originalImage = cv2.imread(filename=originalFileName)

if originalImage is None:
    raise Exception("Image not loaded, please check path.")

print(f"{bcolors.OKGREEN}Image {originalFileName} loaded.{bcolors.ENDC}")
print(f"{bcolors.BOLD}Dimensions:{bcolors.ENDC} {originalImage.shape[1]}x{originalImage.shape[0]}")
print(f"{bcolors.BOLD}Colors:{bcolors.ENDC} {originalImage.shape[2]}\n")

# Change directory to save images into
if not os.path.exists(targetDirectory):
    os.mkdir(targetDirectory)
os.chdir(targetDirectory)

# cv2.imwrite("noise-salt-and-peper.jpg", addSaltAndPepperNoise(originalImage))
# cv2.imwrite("noise-gaussian.jpg", addGaussianNoise(originalImage))
# cv2.imwrite("noise-poisson.jpg", addPoissonNoise(originalImage))
# cv2.imwrite("noise-random.jpg", addRandomNoise(originalImage))
os.chdir("../")
saveImageHistograms(originalFileName)

print(f"{bcolors.OKGREEN}\nAll done!\n{bcolors.ENDC}")