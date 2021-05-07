from frequencyFilter import frequencyFilterCMD 
from mathMorphology import mathematicalMorphologyCMD
from pointsOfInterest import pointsOfInterestCMD
from spatiatFilter import spatiatFilterCMD
from eTransformation import ElementaryTransformationCMD
import  numpy as np
import cv2
from matplotlib import pyplot as plt
import utils 
from tkinter import ttk
from tkinter.filedialog import asksaveasfilename

def runCommandDispatcher(root,leftSetting,image,result):
    selectedsetting  = leftSetting.getFilterType()
    if selectedsetting == "Elementary Transformation":
        ElementaryTransformationCMD(root,leftSetting,image,result)

    if selectedsetting == "Spatial domain Filters":
        spatiatFilterCMD(leftSetting,image,result)

    if selectedsetting == "Frequency domain filters":
        frequencyFilterCMD(leftSetting,image,result)
    
    if selectedsetting == "Mathematical morphology":
        mathematicalMorphologyCMD(leftSetting,image,result)
    
    if selectedsetting == "Points of interest":
        pointsOfInterestCMD(leftSetting,image,result)
    
def processHistogramCommand(leftSetting,image,result):
    img1 = image.getOimagePil()
    img = utils.pillToCv2(img1)
    bool = False
    is_grey = utils.is_grey_scale(img1)
    if is_grey or img.shape[2] == 2:
        bool = True
        #dst = cv2.calcHist(img, [0], None, [256], [0,256])
        plt.hist(img.ravel(),256,[0,256])
        plt.title('grayscale Histogramme')
        plt.show()
        bool = False
    
    elif img.shape[2] == 3 and not bool:
        #img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        color = ('b','g','r') 
        plt.figure()
        for i,col in enumerate(color):
            histr = cv2.calcHist([img],[i],None,[256],[0,256])
            plt.plot(histr,color = col)
            plt.xlim([0,256])
        plt.title('3 channels Histogramme')
        plt.show()
def processOpenCvCommand(leftSetting,image,result):
    img1 = image.getOimagePil()
    img2 = result.getPillImage()
    cvimgO = utils.pillToCv2(img1)
    if not isinstance(img2, str):
        cvimgR = utils.pillToCv2(img2)
        cv2.imshow("Processed Image",cvimgR)
    cv2.imshow("Original image",cvimgO)
    

def processSaveCommand(leftSetting,image,result):
    file_path = asksaveasfilename(
                filetypes=[("png files","*.png")],defaultextension=".png"
            )
    img1 = result.getPillImage().copy()
    img1 = img1.save(file_path)

