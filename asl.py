import util
from aslBuilder import aslBuilder
import preprocess

import os
import cv2 as cv
import numpy as np
import tensorflow as ts
import matplotlib.pyplot as plt
import tensorflow as ts
from tensorflow import keras
from tensorflow.keras import utils, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split


def main(image_path,train_path):
	#String to convert to ASL
	text_file = open("test_String", "r")
	starter = text_file.read()
	text_file.close()
	#init the ASL Builder object with string and path
	aslB = aslBuilder(starter,image_path)
	#Parse string
	aslB.parseString()
	# ASL images to each char
	aslB.char2Image()
	#display the images
	#util.image_show(aslB.aslArray)
	
	#Creates and displays the classes.jpg file, doesn't need to run every time, just once to generate the jpg
	#displayClasses(train_path)

	#loads the training data for processing and save the results so we dont have to process data everytime
	#x_train, x_test, y_train, y_test = load_train(train_path)
	#np.save("x_train_Canny", x_train)
	#np.save("x_test_Canny", x_test)
	#np.save("y_train_Canny", y_train)
	#np.save("y_test_Canny", y_test)


	#load training data from files (gs = grayscale)
	#x_train and y_train are only needed to train new models, 
	#x_test, y_test are for testing the model
	#x_train = np.load('x_train_p2.npy')
	#x_test = np.load('x_test_p2.npy')
	#y_train = np.load('y_train_p2.npy')
	#y_test = np.load('y_test_p2.npy')

	#configure the model
	#model = modelConfig()
	#fit the model with data and save the fitted model
	#history = fit(model,x_train,y_train)
	#print(history)
	#load fitted model from file
	#model = load_fit('model_params_grayscale')
	#print("Testing with GrayScale Model")
	#run_all(aslB, model)

	
	#model = load_fit('model_params_sobel')
	#print("Testing with Sobel Model")
	#run_all(aslB, model)

	
	#model = load_fit('model_params_histogram_equl')
	#print("Testing with Histogram Equalized Model")
	#run_all(aslB, model)

	
	# model = load_fit('model_params_mask')
	# print("Testing with Mask Model")
	# run_all(aslB, model)

	
	#model = load_fit('model_params_Canny')
	#print("Testing with Canny Edge Detector Model")
	#run_all(aslB, model)


	#results(model,x_test,y_test)



def displayClasses(train_path):
	classes = ['A','B','C','D','E','F','G','H','I', 'J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','space']
	plt.figure(figsize=(11, 11))
	for i in range(0,27):
		plt.subplot(7,7,i+1)
		plt.xticks([])
		plt.yticks([])
		path = train_path + "/{0}/{0}1.jpg".format(classes[i])
		img = plt.imread(path)
		plt.imshow(img)
		plt.xlabel(classes[i])
	plt.savefig("classes.jpg")
def load_train(train_path):
	im = []
	label = []
	#use this to quickly adjust image size
	size = 50,50 
	i = -1
	for f in os.listdir(train_path):
		i += 1
		for image in os.listdir(train_path+"/"+f):
			#print(image)
			temp = cv.imread(train_path+'/'+f+'/'+image)
			temp = cv.resize(temp, size)
			temp = cv.cvtColor(temp,cv.COLOR_BGR2GRAY)

			#Call the appropriate preprocess to train the model with that preprocessing method
			temp = preprocess.preprocess1(temp)
			#temp = preprocess.preprocess2(temp)
			#temp = preprocess.preprocess3(temp)
			#temp = preprocess.preprocess4(temp)

			im.append(temp)
			label.append(i)
	im = np.array(im)
	im = im.astype('uint8')/255.0
	label = utils.to_categorical(label)
	x_train, x_test, y_train, y_test = train_test_split(im, label, test_size = 0.1)

	print('Loaded', len(x_train),'images for training,','Train data shape =', x_train.shape)
	print('Loaded', len(x_test),'images for testing','Test data shape =', x_test.shape)

	return x_train, x_test, y_train, y_test

def modelConfig():
	model = ts.keras.models.Sequential()

	model.add(Conv2D(64, (3,3), padding='same', input_shape=(50,50,1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (3, 3), padding='same', input_shape=(50, 50,1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(256, (3, 3), padding='same', input_shape=(50, 50,1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(BatchNormalization())

	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(1024, activation='sigmoid'))
	model.add(Dense(27, activation='softmax'))

	return model
def fit(model,x_train,y_train):
	x_train = ts.expand_dims(x_train, axis=-1)
	classes = 27
	batch = 64
	epochs = 5
	learning_rate = 0.001
	adam = Adam(learning_rate=learning_rate)
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
	history = model.fit(x_train, y_train,batch_size=batch, epochs=epochs, validation_split=0.1, shuffle=True,verbose=1)
	model.summary()
	#save the model with appropriate name
	model.save('model_params_Canny')
	plt.figure(figsize=(12, 12))
	plt.subplot(3, 2, 1)
	plt.plot(history.history['accuracy'], label = 'train_accuracy')
	plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.legend()
	plt.subplot(3, 2, 2)
	plt.plot(history.history['loss'], label = 'train_loss')
	plt.plot(history.history['val_loss'], label = 'val_loss')
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.legend()
	plt.show()
	return history

def load_fit(name):
	return ts.keras.models.load_model(name)


def results(model,x_test,y_test):
	x_test = ts.expand_dims(x_test, axis=-1)
	test_loss, test_acc = model.evaluate(x_test,y_test)
	print('Test accuracy:', test_acc)
	print('Test loss:', test_loss)


def run_all(aslB, model):

	x_test_1 = aslB.preprocessASL(1)
	x_test_2 = aslB.preprocessASL(2)
	#x_test_3 = aslB.preprocessASL(3)
	x_test_4 = aslB.preprocessASL(4)
	y_test = aslB.makeLabels()
	y_test = np.array(y_test)

	print("Testing GrayScale input")
	results(model,aslB.aslArray, y_test)

	print("Testing Sobel input")
	results(model, x_test_1, y_test)

	print("Testing Histogram Equalized input")
	results(model, x_test_2, y_test)

	#print("Testing Mask Input")
	#results(model, x_test_3, y_test)

	print("Testing Canny Edge Detector Input")
	results(model, x_test_4, y_test)


if __name__ == '__main__':
    main(image_path='hands2',train_path='hands2')