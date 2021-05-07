import  numpy as np
import  utils
from PIL import Image, ImageTk
import cv2
import math
from scipy import signal

def susan(leftSetting,image,result):
    img1 = image.getOimagePil()
    r = utils.decideKernel(leftSetting)
    raduis = utils.reduceReduis(r) # reduce the raduis to 2 3 5
    data_output = utils.pillToCv2(img1)
    data_output = utils.add_padding(data_output, raduis, 0)
   
    data = cv2.cvtColor(data_output,cv2.COLOR_RGB2GRAY)
    data = cv2.medianBlur(data,3) ### to denoise the image 
    data = cv2.GaussianBlur(data,(5,5),0) ### to smooth the image 
    data = data.astype(np.float64)
    nucleus = utils.susanNucleus(raduis)
    g = utils.susanSum(nucleus)//2 ### 3*utils.susanSum(nucleus)/4
    for i in range(raduis,data.shape[0]-raduis):
        for j in range(raduis,data.shape[1]-raduis):
            ir=np.array(data[i-raduis:i+raduis+1, j-raduis:j+raduis+1])
            ir =  ir[nucleus==1]
            ir0 = data[i,j]
            n=np.sum(np.exp(-((ir-ir0)/10)**6)) ## t = 10 
            if n<=g: ## if n>g means this is an homog erea
                n=g-n
            else:
                n=0
            ## we could test only if n != 0 so we capture all edges pixels
            ## instead we  make sure that is a truly an edge
            if n < g//2 and n > 0: 
                ## so give it green color
                data_output[i-1:i+1,j-1:j+1] = (0,0,255)
            
    ##################### remove the padding #################
    img_final = data_output[raduis+2:data_output.shape[0]-raduis-2,raduis+2:data_output.shape[1]-raduis-2]
    img_final = cv2.cvtColor(img_final,cv2.COLOR_RGB2BGR)
    result.setPillImage(Image.fromarray(img_final))  
    image = ImageTk.PhotoImage(Image.fromarray(img_final))
    result.addProccedImage(image)

def harris(leftSetting,image,result):
    img1 = image.getOimagePil()
    k = 0.04
    threshold = 60000
    offset = int(utils.decideKernel(leftSetting)/2)
    data = utils.pillToCv2(img1)
    
    data_output = cv2.cvtColor(data,cv2.COLOR_RGB2GRAY)
    data_output = cv2.GaussianBlur(data_output,(5,5),0) ### to smooth the image 
    dy, dx = np.gradient(data_output)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    for y in range(offset, data.shape[0]-offset):
        for x in range(offset, data.shape[1]-offset):
            
            #Values of sliding window
            start_y = y - offset
            end_y = y + offset + 1
            start_x = x - offset
            end_x = x + offset + 1
            
            windowIxx = Ixx[start_y : end_y, start_x : end_x]
            windowIxy = Ixy[start_y : end_y, start_x : end_x]
            windowIyy = Iyy[start_y : end_y, start_x : end_x]
            
            #Sum of squares of intensities of partial derevatives 
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            #Calculate determinant and trace of the matrix
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            
            #Calculate r for Harris Corner equation
            r = det - k*(trace**2)

            if r > threshold:
                data[y,x] = (0,0,255)

    img_final = cv2.cvtColor(data,cv2.COLOR_RGB2BGR)
    result.setPillImage(Image.fromarray(img_final))  
    image = ImageTk.PhotoImage(Image.fromarray(img_final))
    result.addProccedImage(image)

def electrostaticModel(leftSetting,image,result):
    img1 = image.getOimagePil()
    k = 0.04
    threshold = 80000
    offset = int(utils.decideKernel(leftSetting)/2)
    data = utils.pillToCv2(img1)
    
    data_output = cv2.cvtColor(data,cv2.COLOR_RGB2GRAY)
    data_output = cv2.GaussianBlur(data_output,(3,3),0) ### to smooth the image 
     
    #too slow
    #dxa = utils.convolve2DGREY(data_output,np.array([[math.sqrt(2)/4,0,-math.sqrt(2)/4],[1,0,-1],[math.sqrt(2)/4,0,-math.sqrt(2)/4]]))
    #dya = utils.convolve2DGREY(data_output,np.array([[math.sqrt(2)/4,1,math.sqrt(2)/4],[0,0,0],[-math.sqrt(2)/4,-1,-math.sqrt(2)/4]]))
    #dxr = utils.convolve2DGREY(data_output,np.array([[-math.sqrt(2)/4,0,math.sqrt(2)/4],[-1,0,1],[-math.sqrt(2)/4,0,math.sqrt(2)/4]]))
    #dyr = utils.convolve2DGREY(data_output,np.array([[-math.sqrt(2)/4,-1,-math.sqrt(2)/4],[0,0,0],[math.sqrt(2)/4,1,math.sqrt(2)/4]]))

    dxa = signal.convolve2d(data_output, np.array([[math.sqrt(2)/4,0,-math.sqrt(2)/4],[1,0,-1],[math.sqrt(2)/4,0,-math.sqrt(2)/4]]),mode='same')
    dya = signal.convolve2d(data_output,np.array([[math.sqrt(2)/4,1,math.sqrt(2)/4],[0,0,0],[-math.sqrt(2)/4,-1,-math.sqrt(2)/4]]),mode='same')
    dxr = signal.convolve2d(data_output, np.array([[-math.sqrt(2)/4,0,math.sqrt(2)/4],[-1,0,1],[-math.sqrt(2)/4,0,math.sqrt(2)/4]]),mode='same')
    dyr = signal.convolve2d(data_output,np.array([[-math.sqrt(2)/4,-1,-math.sqrt(2)/4],[0,0,0],[math.sqrt(2)/4,1,math.sqrt(2)/4]]),mode='same')

    dxx = np.uint8(np.absolute(dxa)) + np.uint8(np.absolute(dxr))
    dyy = np.uint8(np.absolute(dya)) + np.uint8(np.absolute(dyr))
    
    Ixx = dxx**2   
    Ixy = dyy*dxx
    Iyy = dyy**2
    for y in range(offset, data.shape[0]-offset):
        for x in range(offset, data.shape[1]-offset):
            
            #Values of sliding window
            start_y = y - offset
            end_y = y + offset + 1
            start_x = x - offset
            end_x = x + offset + 1
            
            windowIxx = Ixx[start_y : end_y, start_x : end_x]
            windowIxy = Ixy[start_y : end_y, start_x : end_x]
            windowIyy = Iyy[start_y : end_y, start_x : end_x]
            
            #Sum of squares of intensities of partial derevatives 
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()
            #Calculate determinant and trace of the matrix
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            #Calculate r for Harris Corner equation
            r = det - k*(trace**2)
            
            if r > threshold:
                data[y-1:y+1,x-1:x+1] = (0,0,255) ## color a (2,2) window

    img_final = np.uint8(np.absolute(data))
    img_final = cv2.cvtColor(img_final,cv2.COLOR_RGB2BGR)
    result.setPillImage(Image.fromarray(img_final))  
    image = ImageTk.PhotoImage(Image.fromarray(img_final))
    result.addProccedImage(image)
    

def pointsOfInterestCMD(leftSetting,image,result):

    selectedFilter = leftSetting.getFilterGoal()
    
    if selectedFilter == "Susan":
        susan(leftSetting,image,result)

    if selectedFilter == "Harris":
        harris(leftSetting,image,result)
       
    if selectedFilter == "Electrostatic model":
        electrostaticModel(leftSetting,image,result)
        
     