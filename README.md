# DMET M102 - Computer Vision

## Assignment 1

This project consists of a Python script that takes an image and applies the following operations:

- Edge Detection using Canny
- Adding Noise
    1. Salt and Pepper Noise
    2. Gaussian Noise
    3. Poisson Noise
    4. Random Noise
- Averaging Filter
    1. 3x3
    2. 5x5
    3. 9x9

and saves the results of each into the `./results` folder.

## Prerequisites

- Python (3.10.4)
- Install Python Modules
    - matplotlib
    - numpy
    - opencv-python

Install required modules by running:
```bash
pip install -r requirements.txt
```

## How To Run
The script performs all operations on any image named `original.jpg` in the root directory. However, if you'd like to use a different image with a different name, simply change the value of `originalFileName` in line 7 of `main.py`.

```bash
python ./main.py
```