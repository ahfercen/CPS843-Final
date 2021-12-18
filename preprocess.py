import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


# preprocess individual images

# first preprocess
def preprocess1(input):
	# Right now leave it as no change
	# Sobels gradient
	########################################
	preX = cv.Sobel(input, ddepth=cv.CV_8U, dx=1, dy=0)
	preY = cv.Sobel(input, ddepth=cv.CV_8U, dx=0, dy=1)
	outputX = cv.convertScaleAbs(preX)
	outputY = cv.convertScaleAbs(preY)
	# combine the gradient representations into a single image
	output = cv.addWeighted(outputX, 0.5, outputY, 0.5, 0)
	#cv.imshow("addWeighted", output)
	#cv.waitKey(0);
	########################################
	return output


# second preprocess
def preprocess2(input):
	# Histogram equalization
	image_histogram, bins = np.histogram(input.flatten(), 256, density=True)
	cdf = image_histogram.cumsum()  # cumulative distribution function
	cdf = 255 * cdf / cdf[-1]  # normalize

	image_equalized = np.interp(input.flatten(), bins[:-1], cdf)
	output = image_equalized.reshape(input.shape)

	# print(output)
	# plt.imshow(output, aspect="auto")
	# plt.show()
	# cv.imshow("test", output)
	return output


# third preprocess
def preprocess3(input):
	# Use this mask for the smoothing spacial filter
	#mask = np.ones([3, 3], dtype=int)
	#mask = mask / 9
	# We can change this to a better mask if we need to
	# we will have to change how they are added / multiplied in the loop to match
	# algorithm of that mask type
	mask = [
		[0, -1, 0],
		[-1, 5, -1],
		[0, -1, 0]
	]
	m, n = input.shape
	img_new = np.zeros([m, n])
	for i in range(1, m - 1):
		for j in range(1, n - 1):
			_a = input[i - 1, j - 1] * mask[0][0]
			_b = input[i - 1, j] * mask[0][1]
			_c = input[i - 1, j + 1] * mask[0][2]
			_d = input[i, j - 1] * mask[1][0]
			_e = input[i, j] * mask[1][1]
			_f = input[i, j + 1] * mask[1][2]
			_g = input[i + 1, j - 1] * mask[2][0]
			_h = input[i + 1, j] * mask[2][1]
			_i = input[i + 1, j + 1] * mask[2][2]
			temp = _a + _b + _c + _d + _f + _g + _h + _i
			img_new[i, j] = temp

	output = img_new.astype(np.uint8)

	### Uncomment this to show images ###
	#plt.imshow(output, aspect="auto")
	#plt.show()
	output = input
	return output


# fourth preprocess
def preprocess4(input):
	# Laplacian
	########################################
	# output = cv.Laplacian(input, cv.CV_8U)
	########################################

	

	# Canny edge (this gets decent results)
	########################################
	t_lower = 50  # Lower Threshold
	t_upper = 150  # Upper threshold
	output = cv.Canny(input, t_lower, t_upper)
	#cv.imshow("Canny", output)
	#cv.waitKey(0);
	########################################

	# Gaussian smoothing
	########################################
	# the (5, 5) param is the kernel size, increasing it creates a larger blur effect
	# output = cv.GaussianBlur(input, (1, 1), cv.BORDER_DEFAULT)
	########################################

	### Uncomment this to show images ###
	# plt.imshow(output, aspect="auto")
	# plt.show()
	return output
