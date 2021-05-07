import os

KERNEL = ["3X3","5X5", "9X9"]
IMAGES = []
FILTERS = ["moyenne","Gaussian"]
GOAL = ["blur","contour","edge detection"]
IMAGETYPE = ['GREY', 'RGB', 'BINARY']
FILTERSTYPE = ["Elementary Transformation","Spatial domain Filters","Frequency domain filters","Mathematical morphology","Points of interest"]
ELTRANSFORMATION = ["Adjust Image Brightness","Color inversion","Gray Scale image","Binarization by thresholding","Gaussian-Noise","Poivre&Sel-Noise","Contrast enhancement"]
FILTERSPATIAL = ["Median Filter","Average Filter","Gaussian Filter","Contour detection-Laplacian","Contour detection-Sobel","Contour detection-Gradient","Contour detection-Prewitt","Contour detection-Roberts","Contour detection-Laplacian-Gaussian","Contour detection-Kirsch","Contour detection-Canny"]
FILTRAGEFREQUENTIAL = ["Contour detection-FFT High Pass","image enhancement-FFT Band Pass","image enhancement-FFT Low Pass","Contour detection-Butterworth High Pass","Butterworth Low Pass"]
MORPHOLOGIEM = ["Erosion","Dilation","Opening","Closing","White Top Hat","Black Top Hat","Contour Detection Gradient"]
INTERESTP = ["Susan","Harris","Electrostatic model"]

def loadAllImages():
    listOfFiles = os.listdir('./resources/images')
    for entry in listOfFiles:
        IMAGES.append(entry)

