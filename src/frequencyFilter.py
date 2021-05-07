import  numpy as np
import  utils
from PIL import Image, ImageTk
import cv2
import math


def contourDetectionFFT(leftSetting,image,result):
    img1 = image.getOimagePil()
    mask_size = utils.decideKernel(leftSetting)
    data = utils.pillToCv2(img1)
    if data.shape[2] == 3:
        data = cv2.cvtColor(data,cv2.COLOR_RGB2GRAY)
    data = cv2.GaussianBlur(data, (3, 3), 0)

    ### construct a high pass mask  ######### 
    rows , cols = data.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # center
    mask = np.ones((rows, cols),dtype=np.uint8)
    r = mask_size*6
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0
    ########### visulize the mask #########
    # cv2.imshow("mask",mask*255)

    img = np.fft.fft2(data)
    img = np.fft.fftshift(img)
    img = img * mask
    img_final = np.fft.ifft2(img)
    img_final = np.uint8(np.absolute(img_final))
    result.setPillImage(Image.fromarray(img_final))  
    image = ImageTk.PhotoImage(Image.fromarray(img_final))
    result.addProccedImage(image)

def imageEnhancementFFT(leftSetting,image,result):
    img1 = image.getOimagePil()
    mask_size = utils.decideKernel(leftSetting)
    data = utils.pillToCv2(img1)
    img_final = ""

    ## construct a mask #
    sigmax, sigmay = (mask_size-1)*14, (mask_size-1)*14
    cy, cx = data.shape[0]/2,  data.shape[1]/2
    x = np.linspace(0, data.shape[1], data.shape[1])
    y = np.linspace(0, data.shape[0], data.shape[0])
    X, Y = np.meshgrid(x, y)
    gmask = np.exp(-(((X-cx)/sigmax)**2 + ((Y-cy)/sigmay)**2))
    # cv2.imshow("mask",gmask)
    if data.shape[2] == 3:

        r,g,b = cv2.split(data)

        r_fft = np.fft.fft2(r)
        g_fft = np.fft.fft2(g)
        b_fft = np.fft.fft2(b)
        
        ##### shifted #####
        r_fft = np.fft.fftshift(r_fft)
        g_fft = np.fft.fftshift(g_fft)
        b_fft = np.fft.fftshift(b_fft)

        ## apply the mask to each channel
        r_fft = r_fft * gmask
        g_fft = g_fft * gmask
        b_fft = b_fft * gmask
        # cv2.imshow("r",np.uint8(np.absolute(r_fft)))
        # cv2.imshow("g",np.uint8(np.absolute(g_fft)))
        # cv2.imshow("b",np.uint8(np.absolute(b_fft)))
        ## ifft inverse fft ##
        r_fft = np.fft.ifft2(r_fft)
        g_fft = np.fft.ifft2(g_fft)
        b_fft = np.fft.ifft2(b_fft)
        
        img_final = cv2.merge((np.uint8(np.absolute(r_fft)),np.uint8(np.absolute(g_fft)),np.uint8(np.absolute(b_fft))))
        ## BGR sometimes in tkinter is the RGB in CV2 
        # Strange conversion needed!!! 
        img_final = cv2.cvtColor(img_final,cv2.COLOR_BGR2RGB)
    else:
        img = np.fft.fft2(data)
        img = np.fft.fftshift(img)
        img = img * gmask
        img = np.fft.ifft2(img)
        img_final = img

    result.setPillImage(Image.fromarray(img_final))  
    image = ImageTk.PhotoImage(Image.fromarray(img_final))
    result.addProccedImage(image)

def imageEnhancementFFTBandPass(leftSetting,image,result):
    img1 = image.getOimagePil()
    mask_size = utils.decideKernel(leftSetting)
    data = utils.pillToCv2(img1)

    rows , cols = data.shape[0], data.shape[1]
    crow, ccol = int(rows / 2), int(cols / 2)  # center
    gmask = np.zeros((rows, cols))
    r_out = mask_size*18
    r_in = mask_size/2 
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                           ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
    gmask[mask_area] = 1
    ########### visulize the mask #########
    #cv2.imshow("mask",gmask*255)
    
    if data.shape[2] == 3:
        
        r,g,b = cv2.split(data)

        r_fft = np.fft.fft2(r)
        g_fft = np.fft.fft2(g)
        b_fft = np.fft.fft2(b)
        
        ##### shifted #####
        r_fft = np.fft.fftshift(r_fft)
        g_fft = np.fft.fftshift(g_fft)
        b_fft = np.fft.fftshift(b_fft)

        ## apply the mask to each channel
        r_fft = r_fft * gmask
        g_fft = g_fft * gmask
        b_fft = b_fft * gmask
        # cv2.imshow("r",np.uint8(np.absolute(r_fft)))
        # cv2.imshow("g",np.uint8(np.absolute(g_fft)))
        # cv2.imshow("b",np.uint8(np.absolute(b_fft)))
        ## ifft inverse fft ##
        r_fft = np.fft.ifft2(r_fft)
        g_fft = np.fft.ifft2(g_fft)
        b_fft = np.fft.ifft2(b_fft)
        
        img_final = cv2.merge((np.uint8(np.absolute(r_fft)),np.uint8(np.absolute(g_fft)),np.uint8(np.absolute(b_fft))))
        ## BGR sometimes in tkinter is the RGB in CV2 
        # Strange conversion needed!!! 
        img_final = cv2.cvtColor(img_final,cv2.COLOR_BGR2RGB)
    else:
        img = np.fft.fft2(data)
        img = np.fft.fftshift(img)
        img = img * gmask
        img = np.fft.ifft2(img)
        img_final = img
    
    img_final = np.uint8(np.absolute(img_final))
    img_final = cv2.cvtColor(img_final,cv2.COLOR_RGB2BGR)
    result.setPillImage(Image.fromarray(img_final))  
    image = ImageTk.PhotoImage(Image.fromarray(img_final))
    result.addProccedImage(image)

def contourDetectionButterworth(leftSetting,image,result):
    img1 = image.getOimagePil()
    mask_size = utils.decideKernel(leftSetting)
    data = utils.pillToCv2(img1)
    if data.shape[2] == 3:
        data = cv2.cvtColor(data,cv2.COLOR_RGB2GRAY)
    data = cv2.GaussianBlur(data, (3, 3), 0)
    n = 2
    D0 = mask_size * 3
    mask = np.zeros(data.shape[:2])
    rows, cols = data.shape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            mask[y,x] = 1-1/(1+(utils.distance((y,x),center)/D0)**(2*n))
    #cv2.imshow("mask",mask)
    img = np.fft.fft2(data)
    img = np.fft.fftshift(img)
    img = img * mask
    img = np.fft.ifft2(img)

    img_final = np.uint8(np.absolute(img))
    result.setPillImage(Image.fromarray(img_final))  
    image = ImageTk.PhotoImage(Image.fromarray(img_final))
    result.addProccedImage(image)

def ButterworthLowPass(leftSetting,image,result):
    img1 = image.getOimagePil()
    mask_size = utils.decideKernel(leftSetting)
    data = utils.pillToCv2(img1)
    n = 2
    D0 = mask_size * 12
    gmask = np.zeros(data.shape[:2])
    rows, cols = data.shape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            gmask[y,x] = 1/(1+(utils.distance((y,x),center)/D0)**(2*n))
    cv2.imshow("mask",gmask)
    
    if data.shape[2] == 3:
        
        r,g,b = cv2.split(data)

        r_fft = np.fft.fft2(r)
        g_fft = np.fft.fft2(g)
        b_fft = np.fft.fft2(b)
        
        ##### shifted #####
        r_fft = np.fft.fftshift(r_fft)
        g_fft = np.fft.fftshift(g_fft)
        b_fft = np.fft.fftshift(b_fft)

        ## apply the mask to each channel
        r_fft = r_fft * gmask
        g_fft = g_fft * gmask
        b_fft = b_fft * gmask
        # cv2.imshow("r",np.uint8(np.absolute(r_fft)))
        # cv2.imshow("g",np.uint8(np.absolute(g_fft)))
        # cv2.imshow("b",np.uint8(np.absolute(b_fft)))
        ## ifft inverse fft ##
        r_fft = np.fft.ifft2(r_fft)
        g_fft = np.fft.ifft2(g_fft)
        b_fft = np.fft.ifft2(b_fft)
        
        img_final = cv2.merge((np.uint8(np.absolute(r_fft)),np.uint8(np.absolute(g_fft)),np.uint8(np.absolute(b_fft))))
        ## BGR sometimes in tkinter is the RGB in CV2 
        # Strange conversion needed!!! 
        img_final = cv2.cvtColor(img_final,cv2.COLOR_BGR2RGB)
    else:
        img = np.fft.fft2(data)
        img = np.fft.fftshift(img)
        img = img * gmask
        img = np.fft.ifft2(img)
        img_final = img
    
    img_final = np.uint8(np.absolute(img_final))
    #img_final = cv2.cvtColor(img_final,cv2.COLOR_RGB2BGR)
    result.setPillImage(Image.fromarray(img_final))  
    image = ImageTk.PhotoImage(Image.fromarray(img_final))
    result.addProccedImage(image)



def frequencyFilterCMD(leftSetting,image,result):
    selectedFilter = leftSetting.getFilterGoal()

    if selectedFilter == "Contour detection-FFT High Pass":
        contourDetectionFFT(leftSetting,image,result)

    if selectedFilter == "image enhancement-FFT Band Pass":
        imageEnhancementFFTBandPass(leftSetting,image,result)

    if selectedFilter == "image enhancement-FFT Low Pass":
         imageEnhancementFFT(leftSetting,image,result)
       
    
    if selectedFilter == "Contour detection-Butterworth High Pass":
        contourDetectionButterworth(leftSetting,image,result)
    
    if selectedFilter == "Butterworth Low Pass":
        ButterworthLowPass(leftSetting,image,result)
        