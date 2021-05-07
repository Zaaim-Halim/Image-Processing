import  numpy as np
import  utils
from PIL import Image, ImageTk
import cv2
import math

def erosion(leftSetting,image,result):
    img1 = image.getOimagePil()
    structurant_size = utils.decideKernel(leftSetting)
    data = utils.pillToCv2(img1)
    # if data.shape[2] == 3:
    #     data = cv2.cvtColor(data,cv2.COLOR_RGB2GRAY)
    # kernel = np.ones((structurant_size,structurant_size),dtype=np.uint8)

    # img = utils.operation(data,kernel,operation="erosion")
    kernel = np.ones((structurant_size,structurant_size),np.uint8)
    img_final = cv2.erode(data,kernel,iterations = 1)
    img_final = cv2.cvtColor(img_final,cv2.COLOR_RGB2BGR)
    result.setPillImage(Image.fromarray(img_final))  
    image = ImageTk.PhotoImage(Image.fromarray(img_final))
    result.addProccedImage(image)

def dilation(leftSetting,image,result):
    img1 = image.getOimagePil()
    structurant_size = utils.decideKernel(leftSetting)
    data = utils.pillToCv2(img1)
    kernel = np.ones((structurant_size,structurant_size),np.uint8)
    img_final = cv2.dilate(data,kernel,iterations = 1)
    img_final = cv2.cvtColor(img_final,cv2.COLOR_RGB2BGR)
    result.setPillImage(Image.fromarray(img_final))  
    image = ImageTk.PhotoImage(Image.fromarray(img_final))
    result.addProccedImage(image)

def opening(leftSetting,image,result):
    img1 = image.getOimagePil()
    structurant_size = utils.decideKernel(leftSetting)
    data = utils.pillToCv2(img1)
    kernel = np.ones((structurant_size,structurant_size),np.uint8)
    img_final = cv2.morphologyEx(data, cv2.MORPH_OPEN, kernel)
    ## this is equivalent to 
    # opening = 
    #
    img_final = cv2.cvtColor(img_final,cv2.COLOR_RGB2BGR)
    result.setPillImage(Image.fromarray(img_final))  
    image = ImageTk.PhotoImage(Image.fromarray(img_final))
    result.addProccedImage(image)

def closing(leftSetting,image,result):
    img1 = image.getOimagePil()
    structurant_size = utils.decideKernel(leftSetting)
    data = utils.pillToCv2(img1)
    kernel = np.ones((structurant_size,structurant_size),np.uint8)
    img_final = cv2.morphologyEx(data, cv2.MORPH_CLOSE, kernel,iterations=1)
    img_final = cv2.cvtColor(img_final,cv2.COLOR_RGB2BGR)
    result.setPillImage(Image.fromarray(img_final))  
    image = ImageTk.PhotoImage(Image.fromarray(img_final))
    result.addProccedImage(image)

def whiteTopHat(leftSetting,image,result):
    img1 = image.getOimagePil()
    structurant_size = utils.decideKernel(leftSetting)
    data = utils.pillToCv2(img1)
    data =  cv2.cvtColor(data,cv2.COLOR_RGB2GRAY)
    # Cross-shaped Kernel
    kernel =  cv2.getStructuringElement(cv2.MORPH_CROSS,(structurant_size,structurant_size))
    img_final = data - cv2.morphologyEx(data, cv2.MORPH_OPEN, kernel,iterations=2)
    img_final = cv2.cvtColor(img_final,cv2.COLOR_RGB2BGR)
    result.setPillImage(Image.fromarray(img_final))  
    image = ImageTk.PhotoImage(Image.fromarray(img_final))
    result.addProccedImage(image)

def blackTopHat(leftSetting,image,result):
    img1 = image.getOimagePil()
    structurant_size = utils.decideKernel(leftSetting)
    data = utils.pillToCv2(img1)
    data =  cv2.cvtColor(data,cv2.COLOR_RGB2GRAY)
    #Cross-shaped Kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(structurant_size,structurant_size))
    img_final = cv2.morphologyEx(data, cv2.MORPH_CLOSE, kernel,iterations=2) - data
    img_final = cv2.cvtColor(img_final,cv2.COLOR_RGB2BGR)
    result.setPillImage(Image.fromarray(img_final))  
    image = ImageTk.PhotoImage(Image.fromarray(img_final))
    result.addProccedImage(image)
def gradient(leftSetting,image,result):
    img1 = image.getOimagePil()
    structurant_size = utils.decideKernel(leftSetting)
    data = utils.pillToCv2(img1)
    data =  cv2.cvtColor(data,cv2.COLOR_RGB2GRAY)
    #Cross-shaped Kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(structurant_size,structurant_size))
    dil = cv2.dilate(data,kernel,iterations = 2)
    ero = cv2.erode(data,kernel,iterations = 1)
    img_final = dil - ero
    img_final = cv2.cvtColor(img_final,cv2.COLOR_RGB2BGR)
    result.setPillImage(Image.fromarray(img_final))  
    image = ImageTk.PhotoImage(Image.fromarray(img_final))
    result.addProccedImage(image)

def mathematicalMorphologyCMD(leftSetting,image,result):
    selectedFilter = leftSetting.getFilterGoal()

    if selectedFilter == "Erosion":
        erosion(leftSetting,image,result)

    if selectedFilter == "Dilation":
        dilation(leftSetting,image,result)
       
    if selectedFilter == "Opening":
        opening(leftSetting,image,result)
        
    if selectedFilter == "Closing":
         closing(leftSetting,image,result)
     
    if selectedFilter == "White Top Hat":
         whiteTopHat(leftSetting,image,result)
     
    if selectedFilter == "Black Top Hat":
        blackTopHat(leftSetting,image,result)

    if selectedFilter == "Contour Detection Gradient":
        gradient(leftSetting,image,result)