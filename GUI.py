#import opencv library
import collections
import cv2
#import matplotlib
from matplotlib import pyplot as plt
import numpy
import numpy as np

# importing libraries
from skimage import metrics
from skimage.util import random_noise
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys

import cgitb
cgitb.enable(format = 'text')


class DisplayImageWidget(QWidget):
    def __init__(self ,title ,img , hist=[], parent=None):
        super(DisplayImageWidget, self).__init__()
        self.image = img
        self.title = title
        self.hist = hist
        self.setWindowTitle(self.title)
        self.image_frame = QLabel()
        self.image_frame.setAccessibleDescription(self.title)
        self.show_image()
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_frame)
        if len(img.shape)==2:
            if len(self.hist)!=0:
                self.graphWidget = pg.PlotWidget()
                self.graphWidget.plot(range(256), self.hist)
                self.layout.addWidget(self.graphWidget)
        self.setLayout(self.layout)
    @QtCore.pyqtSlot()
    def show_image(self):
        if len(self.image.shape) == 2:
            self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_Indexed8).rgbSwapped()
        else:
            self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))



class Window(QMainWindow):

	def __init__(self):
		super().__init__()

		self.setStyleSheet("background-color: #00305F;")
		# setting title
		self.setWindowTitle("Image processing task")


		# calling method
		self.UiComponents()

		# showing all the widgets
		self.show()

	# method for widgets
	def UiComponents(self):

		self.original_img =cv2.imread("lena.png",0)   #flag=0 >>> Grayscale image
		self.rows,self.cols=self.original_img.shape

		self.brighten_button = QPushButton("brighten effect", self)
		self.brighten_button.setStyleSheet('QPushButton {background-color: cyan; color: black; font-weight: bold;}')
		self.brighten_button.setGeometry(650, 350, 150, 40)
		self.brighten_button.pressed.connect(self.brighten)

		self.darken_button = QPushButton("darken", self)
		self.darken_button.setStyleSheet('QPushButton {background-color: cyan; color: black; font-weight: bold;}')
		self.darken_button.setGeometry(850, 350, 150, 40)
		self.darken_button.pressed.connect(self.darken)

		self.add_contrast_button = QPushButton("add_contrast", self)
		self.add_contrast_button.setStyleSheet('QPushButton {background-color: cyan; color: black; font-weight: bold;}')
		self.add_contrast_button.setGeometry(1050, 350, 150, 40)
		self.add_contrast_button.pressed.connect(self.add_contrast)

		self.less_contrast_button = QPushButton("less_contrast", self)
		self.less_contrast_button.setStyleSheet('QPushButton {background-color: cyan; color: black; font-weight: bold;}')
		self.less_contrast_button.setGeometry(650, 450, 150, 40)
		self.less_contrast_button.pressed.connect(self.less_contrast)

		self.contrast_stretching_button = QPushButton("contrast_stretching", self)
		self.contrast_stretching_button.setStyleSheet('QPushButton {background-color: cyan; color: black; font-weight: bold;}')
		self.contrast_stretching_button.setGeometry(850, 450, 150, 40)
		self.contrast_stretching_button.pressed.connect(self.contrast_stretching)

		self.contrast_equalization_button = QPushButton("contrast_equalization", self)
		self.contrast_equalization_button.setStyleSheet('QPushButton {background-color: cyan; color: black; font-weight: bold;}')
		self.contrast_equalization_button.setGeometry(1050, 450, 150, 40)
		self.contrast_equalization_button.pressed.connect(self.contrast_equalization)

		self.rgb_add_button = QPushButton("rgb addition", self)
		self.rgb_add_button.setStyleSheet('QPushButton {background-color: cyan; color: black; font-weight: bold;}')
		self.rgb_add_button.setGeometry(650, 550, 150, 40)
		self.rgb_add_button.pressed.connect(self.rgb_add)

		self.original_rgb_button = QPushButton("show original rgb image", self)
		self.original_rgb_button.setStyleSheet('QPushButton {background-color: cyan; color: black; font-weight: bold;}')
		self.original_rgb_button.setGeometry(850, 550, 150, 40)
		self.original_rgb_button.pressed.connect(self.show_rgb_img)

		self.original_grey_button = QPushButton("show original grey image", self)
		self.original_grey_button.setStyleSheet('QPushButton {background-color: cyan; color: black; font-weight: bold;}')
		self.original_grey_button.setGeometry(1050, 550, 150, 40)
		self.original_grey_button.pressed.connect(self.show_grey_img)

		self.restoration_button = QPushButton("restoration", self)
		self.restoration_button.setStyleSheet(
			'QPushButton {background-color: cyan; color: black; font-weight: bold;}')
		self.restoration_button.setGeometry(650, 650, 150, 40)
		self.restoration_button.pressed.connect(self.restoration)

		self.morphing_button = QPushButton("morphing", self)
		self.morphing_button.setStyleSheet(
			'QPushButton {background-color: cyan; color: black; font-weight: bold;}')
		self.morphing_button.setGeometry(850, 650, 150, 40)
		self.morphing_button.pressed.connect(self.morphing)

		self.filter_button = QPushButton("all filters", self)
		self.filter_button.setStyleSheet(
			'QPushButton {background-color: cyan; color: black; font-weight: bold;}')
		self.filter_button.setGeometry(1050, 650, 150, 40)
		self.filter_button.pressed.connect(self.filters)

		self.showMaximized()

	def histogram_computation(self, Image):
		Image_Height = Image.shape[0]
		Image_Width = Image.shape[1]

		Histogram = numpy.zeros([256])

		for x in range(0, Image_Height):
			for y in range(0, Image_Width):
				Histogram[Image[x, y]] += 1
		return Histogram


	def brighten(self):
		img = cv2.imread("lena.png", 0)
		for i in range(self.rows):
			for j in range(self.cols):
				pixel = img[i, j]
				new_pixel = pixel + 128
				if new_pixel > 255:
					img[i, j] = 255
				elif new_pixel < 0:
					img[i, j] = 0
				else:
					img[i, j] = new_pixel
		hist = None
		if len(img.shape)==2:
			hist = self.histogram_computation(img)
		self.d = DisplayImageWidget('brighten image', img, hist)
		self.d.show()



	def darken(self):
		img = cv2.imread("lena.png", 0)
		for i in range(self.rows):
			for j in range(self.cols):
				pixel = img[i, j]
				new_pixel = pixel - 128
				if new_pixel > 255:
					img[i, j] = 255
				elif new_pixel < 0:
					img[i, j] = 0
				else:
					img[i, j] = new_pixel
		hist = None
		if len(img.shape) == 2:
			hist = self.histogram_computation(img)
		self.d = DisplayImageWidget('darken effect', img, hist)
		self.d.show()

		# division
	def add_contrast(self):
		img = cv2.imread("lena.png", 0)
		for i in range(self.rows):
			for j in range(self.cols):
				pixel = img[i, j]
				new_pixel = pixel / 2
				if new_pixel > 255:
					img[i, j] = 255
				elif new_pixel < 0:
					img[i, j] = 0
				else:
					img[i, j] = new_pixel
		hist = None
		if len(img.shape) == 2:
			hist = self.histogram_computation(img)
		self.d = DisplayImageWidget('high contrast image', img, hist)
		self.d.show()

		# multiplication
	def less_contrast(self):
		img = cv2.imread("lena.png", 0)
		for i in range(self.rows):
			for j in range(self.cols):
				pixel = img[i, j]
				new_pixel = pixel * 2
				if new_pixel > 255:
					img[i, j] = 255
				elif new_pixel < 0:
					img[i, j] = 0
				else:
					img[i, j] = new_pixel
		hist = None
		if len(img.shape) == 2:
			hist = self.histogram_computation(img)
		self.d = DisplayImageWidget('less_contrast image', img, hist)
		self.d.show()

	def contrast_stretching(self):
		img = cv2.imread("lena.png", 0)
		max_pixel = numpy.max(img)
		min_pixel = numpy.min(img)
		for i in range(self.rows):
			for j in range(self.cols):
				pixel = img[i, j]
				if pixel >= max_pixel:
					img[i, j] = 255
				elif pixel <= min_pixel:
					img[i, j] = 0
				else:
					img[i, j] = ((pixel - min_pixel) / (max_pixel - min_pixel)) * 255
		hist = None
		if len(img.shape) == 2:
			hist = self.histogram_computation(img)
		self.d = DisplayImageWidget('contrast_stretching', img, hist)
		self.d.show()

	def contrast_equalization(self):
		img = cv2.imread("lena.png", 0)
		img_1d = img.flatten()
		repetitions = collections.Counter(img_1d)
		total_size = self.rows * self.cols
		pdf = {}
		for i in repetitions:
			pdf[i] = repetitions[i] / total_size
		s = 0
		cdf = {}
		h = {}
		for i in pdf:
			s += pdf[i]
			cdf[i] = s
			h[i] = s * 255

		for i in range(self.rows):
			for j in range(self.cols):
				img[i][j] = h[self.original_img[i][j]]
		hist = None
		if len(img.shape) == 2:
			hist = self.histogram_computation(img)
		self.d = DisplayImageWidget('contrast_equalization', img, hist)
		self.d.show()

	def rgb_add(self):
		img = cv2.imread("lena.png")
		r,c,d = img.shape
		for i in range(r):
			for j in range(c):
				for k in range(d):
					pixel = img[i][j][k]
					new_pixel = pixel + 64
					if new_pixel > 255:
						img[i][j][k] = 255
					elif new_pixel < 0:
						img[i][j][k] = 0
					else:
						img[i][j][k] = new_pixel
		hist = None
		if len(img.shape) == 2:
			hist = self.histogram_computation(img)
		self.d = DisplayImageWidget('rgb addition', img, hist)
		self.d.show()

	def show_grey_img(self):
		img = cv2.imread("lena.png", 0)
		hist = None
		if len(img.shape) == 2:
			hist = self.histogram_computation(img)
		self.d = DisplayImageWidget('original gray image', img, hist)
		self.d.show()

	def show_rgb_img(self):
		img = cv2.imread("lena.png")
		hist = None
		if len(img.shape) == 2:
			hist = self.histogram_computation(img)
		self.d = DisplayImageWidget('original rgb image', img, hist)
		self.d.show()
	def restoration(self):
		img = cv2.imread("lena.png", 0)
		RS = random_noise(img, mode='s&p', amount=0.3)
		NI = (255 * RS).astype(np.uint8)
		Restored_Image = cv2.medianBlur(NI, 5)

		SNR = metrics.peak_signal_noise_ratio(Restored_Image, img)

		#img = cv2.resize(img, (300, 300))
		#RS = cv2.resize(RS, (300, 300))
		#Restored_Image = cv2.resize(Restored_Image, (300, 300))
		self.d = DisplayImageWidget('original image', img)
		self.d.show()
		self.d2 = DisplayImageWidget('noised image', RS)
		self.d2.show()
		self.d3 = DisplayImageWidget('Restored_Image', Restored_Image)
		self.d3.show()

	def morphing(self):
		img = cv2.imread("lena.png", 0)
		ret, binary_image = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)

		kernel = np.ones((5, 5), np.uint8)
		dilation = cv2.dilate(binary_image, kernel, iterations=1)
		erosion = cv2.erode(binary_image, kernel, iterations=1)
		opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
		closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=1)

		img = cv2.resize(img, (300, 300))
		binary_image = cv2.resize(binary_image, (300, 300))
		dilation = cv2.resize(dilation, (300, 300))
		erosion = cv2.resize(erosion, (300, 300))
		opening = cv2.resize(opening, (300, 300))
		closing = cv2.resize(closing, (300, 300))
		self.d = DisplayImageWidget("original", img)
		self.d.show()

		self.d2 = DisplayImageWidget("binary_image", binary_image)
		self.d2.show()

		self.d3 = DisplayImageWidget("dilation", dilation)
		self.d3.show()

		self.d4 = DisplayImageWidget("erosion", erosion)
		self.d4.show()

		self.d5 = DisplayImageWidget("opening", opening)
		self.d5.show()

		self.d6 = DisplayImageWidget("closing", closing)
		self.d6.show()

	def filters(self):
		original_img = cv2.imread("lena.png", 0)
		size_mask = (3,3)


		img2 = cv2.GaussianBlur(original_img, size_mask, 3)
		self.d1 = DisplayImageWidget("GaussianBlur", img2)
		self.d1.show()
		img3 = cv2.blur(original_img, size_mask)
		self.d2 = DisplayImageWidget("Avg", img3)
		self.d2.show()
		img4 = cv2.medianBlur(original_img, size_mask[0])
		self.d3 = DisplayImageWidget("Median", img4)
		self.d3.show()


		img5 = cv2.Laplacian(original_img, cv2.CV_8U)
		self.d4 = DisplayImageWidget("Laplacian", img5)
		self.d4.show()



# create pyqt5 app
App = QApplication(sys.argv)

# create the instance of our Window
window = Window()

# start the app
sys.exit(App.exec())
