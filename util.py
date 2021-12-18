import numpy as np
import os, random, cv2

def load_random_image():
	curdir = os.getcwd()
	rfile = random.choice(os.listdir(curdir))
	image = cv2.imread(rfile)
	nm = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	nm = cv2.resize(nm, (50,50))
	return nm

def image_show(images):
	for w in images:
		cv2.imshow("image",w)
		cv2.waitKey(0)
	cv2.destroyAllWindows()
