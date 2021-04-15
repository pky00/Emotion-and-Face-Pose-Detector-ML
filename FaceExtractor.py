import cv2
import sys

def fixImage(imagePath):

	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
	
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.3,
		minNeighbors=5,
		minSize=(48, 48)
	)

	for (x, y, w, h) in faces:
		gray_image = gray[y:y + h, x:x + w]
		resized_image = cv2.resize(gray_image, (48,48))
		cv2.imwrite("NEW_"+str(imagePath), resized_image)
	
	return


def resizeImage(imagePath):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	resized_image = cv2.resize(gray, (48,48))
	cv2.imwrite("NEW_"+str(imagePath), resized_image)
	return

fixImage(sys.argv[1])
