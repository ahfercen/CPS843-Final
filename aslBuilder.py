import os
import util
import numpy as np
import preprocess

class aslBuilder:
	def __init__(self,s,i):
		self.string = s #original string
		self.path = i #path for asl image files
		self.aslArray = [] #Image data representing the chars of our string

	def parseString(self):
		'''
			Builds the list of strings
		'''
		self.string = self.string.lower()
	def char2Image(self):
		'''
			Load random asl image for each char
		'''
		os.chdir(self.path)
		self.aslArray = np.zeros(shape = (len(self.string),50,50), dtype=np.uint8)
		i = 0
		for c in self.string: #Words
			if (c == " "):
				os.chdir('space')
			else:
				os.chdir(c)
			self.aslArray[i] = util.load_random_image()
			i+=1
			os.chdir('..')
		os.chdir('..')

	def preprocessASL(self, pnum):
		if (pnum == 1):
			ret = []
			for im in self.aslArray:
				ret.append(preprocess.preprocess1(im))
			return ret
		elif(pnum == 2):
			ret = []
			for im in self.aslArray:
				ret.append(preprocess.preprocess2(im))
			return ret
		elif(pnum == 3):
			ret = []
			for im in self.aslArray:
				ret.append(preprocess.preprocess3(im))
			return ret
		elif(pnum == 4):
			ret = []
			for im in self.aslArray:
				ret.append(preprocess.preprocess4(im))
			return ret
		else:
			return self.aslArray

	def makeLabels(self):
		classes = ['A','B','C','D','E','F','G','H','I', 'J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','space']
		ret = []
		for c in self.string:
			cur = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
			if (c == " "):
				cur[26] = 1
			else:
				i = 0;
				while (c != classes[i].lower()):
					i+=1
					if (i==26):
						break
				cur[i] = 1
			ret.append(cur)
		return ret





