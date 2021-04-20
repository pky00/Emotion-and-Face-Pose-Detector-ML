import os
import cv2
from pathlib import Path

import numpy
from typing import List

def splitPath(path: str) -> List:
    """
    Split path string into elements
    """
    parts = list(Path(path).parts)
    return parts

def saveAsJpg(path: str, img: numpy.ndarray) -> None:
    """
    Save as a .jpg
    
    Some files are not jpgs, or are named weirdly like .JPG or .jpeg, so this fixes that
    
    Will create the directory if it doesn't exist
    """
    fName = os.path.basename(path) # Get just filename e.g. image.jpg from /a/b/c/image.jpg
    fName = os.path.splitext(fName)[0] # Strip file extension
    fName = fName + ".jpg" # Set extension to .jpg
    fName = "NEW_" + fName # Prepend "NEW_"
    
    dir = os.path.dirname(path)
    newPath = os.path.join(dir, fName)
    
    if not os.path.exists(dir):
        os.makedirs(dir)

    cv2.imwrite(newPath, img)

def preprocessImage(imagePath: str, newPath: str) -> None:
    """
    Preprocess the image
    
    Detect the face, expand the bounding box by 10%, crop it, convert to grayscale,
    and save it as a jpg
    """
    image = cv2.imread(imagePath)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = faceCascade.detectMultiScale(
        grayImage,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(48, 48)
    )

    for (x, y, w, h) in faces[:1]:
        scaleFactor = 0.10
        
        sideLength = w # Is a square
        
        left = int(x - scaleFactor * sideLength)
        left = max(0, left) # Ensure doesn't overflow the bounds of the original image
        
        right = int(x + sideLength + scaleFactor * sideLength)
        right = min(image.shape[1], right)
        
        top = int(y - scaleFactor * sideLength)
        top = max(0, top)
        
        bottom = int(y + sideLength + scaleFactor * sideLength)
        bottom = min(image.shape[0], bottom)
        
        grayImage = grayImage[top:bottom, left:right]
        
        resizedImage = cv2.resize(grayImage, (48, 48))
        saveAsJpg(newPath, resizedImage)
