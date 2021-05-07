import  numpy as np
import  utils
from PIL import Image, ImageTk
import cv2

def decideKernel(leftSetting):
    kernel  = leftSetting.getKernel()
    if kernel=="3X3":
        return 3
    elif kernel =="5X5":
        return 5
    elif kernel=="9X9":
        return 9

def  medianFilter(leftSetting,image,result):
    img1 = image.getOimagePil()
    kernel_size = decideKernel(leftSetting)
    data = utils.pillToCv2(img1)
    h, w, ch = data.shape
    r,g,b = cv2.split(data)
    indexer = kernel_size // 2
    tempR = []
    tempB = []
    tempG = []
    r_img = np.zeros((h,w),dtype=np.uint8)
    b_img = np.zeros((h,w),dtype=np.uint8)
    g_img = np.zeros((h,w),dtype=np.uint8)
   
    for i in range(h):

        for j in range(w):

            for z in range(kernel_size):
                if i + z - indexer < 0 or i + z - indexer > h - 1:
                    for c in range(kernel_size):
                        tempR.append(0)
                        tempG.append(0)
                        tempB.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > w - 1:
                        tempR.append(0)
                        tempG.append(0)
                        tempB.append(0)
                    else:
                        for k in range(kernel_size):
                            tempB.append(b[i + z - indexer][j + k - indexer])
                            tempG.append(g[i + z - indexer][j + k - indexer])
                            tempR.append(r[i + z - indexer][j + k - indexer])
            tempB.sort()
            tempG.sort()
            tempB.sort()
            if len(tempR) == kernel_size ** 2:
                r_img[i,j] = tempR[len(tempR)//2]
                b_img[i,j] = tempB[len(tempR)//2]
                g_img[i,j] = tempG[len(tempR)//2]
        
            tempB = []
            tempR = []
            tempG = []
           
    img_final = cv2.merge((r_img,g_img,b_img))
    result.setPillImage(Image.fromarray(cv2.cvtColor(img_final,cv2.COLOR_RGB2BGR)))  
    image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img_final,cv2.COLOR_RGB2BGR)))
    result.addProccedImage(image)

def averageFilter(leftSetting,image,result):
    img1 = image.getOimagePil()
    kernel = decideKernel(leftSetting)
    data = utils.pillToCv2(img1)
    filter = np.ones((kernel,kernel),dtype=np.uint8) * 1/(kernel**2)
    img = utils.convolve2DRGB(data,filter)
    result.setPillImage(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_RGB2BGR)))  
    image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_RGB2BGR)))
    result.addProccedImage(image)

def gaussianFilter(leftSetting,image,result):
    img1 = image.getOimagePil()
    kernel = decideKernel(leftSetting)
    sigma = leftSetting.getSigma().get()
    if sigma == 0:
        sigma = 1
    data = utils.pillToCv2(img1)
    filter = utils.gaussian_kernel(sigma,kernel)
    img = utils.convolve2DRGB(data,filter)
    result.setPillImage(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_RGB2BGR)))  
    image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_RGB2BGR)))
    result.addProccedImage(image)
    
def laplacianFilter(leftSetting,image,result):
    img1 = image.getOimagePil()
    kernel_size = decideKernel(leftSetting)
    data = utils.pillToCv2(img1)
    data = cv2.GaussianBlur(data, (3, 3), 0)
    # grey = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
    # cvLaplacian = cv2.Laplacian(grey,cv2.CV_64F, ksize=3)
    # cv2.imshow("cv Laplacian",np.uint8(np.absolute(cvLaplacian)))
    filter = utils.LaplacianKernel(kernel_size)
    img_final = utils.convolve2DRGB_float64(data, filter)
    img_final = np.uint8(np.absolute(img_final))
    img = cv2.cvtColor(img_final, cv2.COLOR_RGB2GRAY)
    result.setPillImage(Image.fromarray(img))  
    image = ImageTk.PhotoImage(Image.fromarray(img))
    result.addProccedImage(image)

def contourDetectionSobel(leftSetting,image,result):
    img1 = image.getOimagePil()
    kernel_size = decideKernel(leftSetting)
    data = utils.pillToCv2(img1)
    data = cv2.GaussianBlur(data, (3, 3), 0)
    # cvsobel = cv2.Sobel(data,cv2.CV_64F,1,0,ksize=3)+cv2.Sobel(data,cv2.CV_64F,0,1,ksize=3)
    # cv2.imshow("cv2 sobel",cv2.cvtColor(np.uint8(np.absolute(cvsobel)),cv2.COLOR_RGB2GRAY))
    (sobel_x,sobel_y) = utils.sobelKernel(kernel_size)
    img_final = utils.convolve2DRGB_float64(data,sobel_x) + utils.convolve2DRGB_float64(data,sobel_y)
    img_final = cv2.cvtColor(np.uint8(np.absolute(img_final)),cv2.COLOR_RGB2GRAY)
    result.setPillImage(Image.fromarray(img_final))  
    image = ImageTk.PhotoImage(Image.fromarray(img_final))
    result.addProccedImage(image)

    

def contourDetectionGradient(leftSetting,image,result):
    img1 = image.getOimagePil()
    kernel = decideKernel(leftSetting)
    data = utils.pillToCv2(img1)
    data = cv2.GaussianBlur(data, (3, 3), 0)
    img_final = utils.convolve2DRGB_float64(data,np.array([[-1,0,1]])) + utils.convolve2DRGB_float64(data,np.array([[-1,0,1]]).T)
    img_final = cv2.cvtColor(np.uint8(np.absolute(img_final)),cv2.COLOR_RGB2GRAY)
    result.setPillImage(Image.fromarray(img_final))  
    image = ImageTk.PhotoImage(Image.fromarray(img_final))
    result.addProccedImage(image)
def contourDetectionPrewitt(leftSetting,image,result):
    img1 = image.getOimagePil()
    kernel_size = decideKernel(leftSetting)
    data = utils.pillToCv2(img1)
    data = cv2.GaussianBlur(data, (3, 3), 0)
    (perwitt_x,perwitt_y) = utils.perwittKernel(kernel_size)
    img_final = utils.convolve2DRGB_float64(data,perwitt_x) + utils.convolve2DRGB_float64(data,perwitt_y)
    img_final = cv2.cvtColor(np.uint8(np.absolute(img_final)),cv2.COLOR_RGB2GRAY)
    result.setPillImage(Image.fromarray(img_final))  
    image = ImageTk.PhotoImage(Image.fromarray(img_final))
    result.addProccedImage(image)

def contourDetectionRoberts(leftSetting,image,result):
    img1 = image.getOimagePil()
    kernel_size = decideKernel(leftSetting)
    data = utils.pillToCv2(img1)
    data = cv2.GaussianBlur(data, (3, 3), 0)
    (robert_x,robert_y) = utils.robertKernel(kernel_size)
    img_final = utils.convolve2DRGB_float64(data,robert_x) + utils.convolve2DRGB_float64(data,robert_y)
    img_final = cv2.cvtColor(np.uint8(np.absolute(img_final)),cv2.COLOR_RGB2GRAY)
    result.setPillImage(Image.fromarray(img_final))  
    image = ImageTk.PhotoImage(Image.fromarray(img_final))
    result.addProccedImage(image)
    
def contourDetectionLaplacian_gaussian(leftSetting,image,result):
    img1 = image.getOimagePil()
    kernel_size = decideKernel(leftSetting)
    data = utils.pillToCv2(img1)
    data = cv2.GaussianBlur(data, (3, 3), 0)
    log_kernel = utils.logkernel(kernel_size)
    img_final = utils.convolve2DRGB_float64(data,log_kernel) 
    img_final = cv2.cvtColor(np.uint8(np.absolute(img_final)),cv2.COLOR_RGB2GRAY)
    result.setPillImage(Image.fromarray(img_final))  
    image = ImageTk.PhotoImage(Image.fromarray(img_final))
    result.addProccedImage(image)
    
def contourDetectionKirsch(leftSetting,image,result):
    img1 = image.getOimagePil()
    #kernel = decideKernel(leftSetting)
    data = utils.pillToCv2(img1)
    data = cv2.GaussianBlur(data, (3, 3), 0)
    gray =cv2.cvtColor(data,cv2.COLOR_RGB2GRAY)
    kernelG1 = np.array([[ 5,  5,  5],
                         [-3,  0, -3],
                         [-3, -3, -3]], dtype=np.float32)
    kernelG2 = np.array([[ 5,  5, -3],
                         [ 5,  0, -3],
                         [-3, -3, -3]], dtype=np.float32)
    kernelG3 = np.array([[ 5, -3, -3],
                         [ 5,  0, -3],
                         [ 5, -3, -3]], dtype=np.float32)
    kernelG4 = np.array([[-3, -3, -3],
                         [ 5,  0, -3],
                         [ 5,  5, -3]], dtype=np.float32)
    kernelG5 = np.array([[-3, -3, -3],
                         [-3,  0, -3],
                         [ 5,  5,  5]], dtype=np.float32)
    kernelG6 = np.array([[-3, -3, -3],
                         [-3,  0,  5],
                         [-3,  5,  5]], dtype=np.float32)
    kernelG7 = np.array([[-3, -3,  5],
                         [-3,  0,  5],
                         [-3, -3,  5]], dtype=np.float32)
    kernelG8 = np.array([[-3,  5,  5],
                         [-3,  0,  5],
                         [-3, -3, -3]], dtype=np.float32)

    g1 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG1), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g2 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG2), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g3 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG3), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g4 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG4), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g5 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG5), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g6 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG6), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g7 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG7), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g8 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG8), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    img_final = cv2.max(
        g1, cv2.max(
            g2, cv2.max(
                g3, cv2.max(
                    g4, cv2.max(
                        g5, cv2.max(
                            g6, cv2.max(
                                g7, g8
                            )
                        )
                    )
                )
            )
        )
    )
    img_final = img_final - 100
    ############### binarization with thresholding ################
    # for i in range(len(img_final)):
    #       for j in range(len(img_final[0])):
            
    #            if img_final[i,j] < 157:
    #                 img_final[i,j] = 0
    #            else:
    #                 img_final[i,j] = 255
    ##############################################################  

    result.setPillImage(Image.fromarray(img_final))  
    image = ImageTk.PhotoImage(Image.fromarray(img_final))
    result.addProccedImage(image)

def contourDetectionCanny(leftSetting,image,result):
    img1 = image.getOimagePil()
    kernel = decideKernel(leftSetting)
    data = utils.pillToCv2(img1)
    # conversion of image to grayscale
    img = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    weak_th = None
    strong_th = None
    # Noise reduction step
    img = cv2.GaussianBlur(img, (5, 5), 1.4)
       
    # Calculating the gradients
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)
      
    # Conversion of Cartesian coordinates to polar 
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees = True)
       
    # setting the minimum and maximum thresholds 
    # for double thresholding
    mag_max = np.max(mag)
    if not weak_th:weak_th = mag_max * 0.1
    if not strong_th:strong_th = mag_max * 0.5
      
    # getting the dimensions of the input image  
    height, width = img.shape
       
    # Looping through every pixel of the grayscale 
    # image
    for i_x in range(width):
        for i_y in range(height):
               
            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang)
               
            # selecting the neighbours of the target pixel
            # according to the gradient direction
            # In the x axis direction
            if grad_ang<= 22.5:
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y
              
            # top right (diagnol-1) direction
            elif grad_ang>22.5 and grad_ang<=(22.5 + 45):
                neighb_1_x, neighb_1_y = i_x-1, i_y-1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
              
            # In y-axis direction
            elif grad_ang>(22.5 + 45) and grad_ang<=(22.5 + 90):
                neighb_1_x, neighb_1_y = i_x, i_y-1
                neighb_2_x, neighb_2_y = i_x, i_y + 1
              
            # top left (diagnol-2) direction
            elif grad_ang>(22.5 + 90) and grad_ang<=(22.5 + 135):
                neighb_1_x, neighb_1_y = i_x-1, i_y + 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y-1
              
            # Now it restarts the cycle
            elif grad_ang>(22.5 + 135) and grad_ang<=(22.5 + 180):
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y
               
            # Non-maximum suppression step
            if width>neighb_1_x>= 0 and height>neighb_1_y>= 0:
                if mag[i_y, i_x]<mag[neighb_1_y, neighb_1_x]:
                    mag[i_y, i_x]= 0
                    continue
   
            if width>neighb_2_x>= 0 and height>neighb_2_y>= 0:
                if mag[i_y, i_x]<mag[neighb_2_y, neighb_2_x]:
                    mag[i_y, i_x]= 0
   
    weak_ids = np.zeros_like(img)
    strong_ids = np.zeros_like(img)              
    ids = np.zeros_like(img)
       
    # double thresholding step
    for i_x in range(width):
        for i_y in range(height):
              
            grad_mag = mag[i_y, i_x]
              
            if grad_mag<weak_th:
                mag[i_y, i_x]= 0
            elif strong_th>grad_mag>= weak_th:
                ids[i_y, i_x]= 1
            else:
                ids[i_y, i_x]= 2
       
       
    # finally returning the magnitude of
    # gradients of edges
    
    result.setPillImage(Image.fromarray(mag)) 
    image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(np.uint8(np.absolute(mag)), cv2.COLOR_BGR2RGB)))
    result.addProccedImage(image)

def  spatiatFilterCMD(leftSetting,image,result):
     selectedFilter = leftSetting.getFilterGoal()

     if selectedFilter == "Median Filter":
        medianFilter(leftSetting,image,result)

     if selectedFilter == "Average Filter":
          averageFilter(leftSetting,image,result)
       
     if selectedFilter == "Gaussian Filter":
          gaussianFilter(leftSetting,image,result)
        
    
     if selectedFilter == "Contour detection-Laplacian":
          laplacianFilter(leftSetting,image,result)
     
     if selectedFilter == "Contour detection-Sobel":
        contourDetectionSobel(leftSetting,image,result)
     
     if selectedFilter == "Contour detection-Gradient":
          contourDetectionGradient(leftSetting,image,result)
     

     if selectedFilter == "Contour detection-Prewitt":
          contourDetectionPrewitt(leftSetting,image,result)

     if selectedFilter == "Contour detection-Roberts":
          contourDetectionRoberts(leftSetting,image,result)

     if selectedFilter == "Contour detection-Laplacian-Gaussian":
          contourDetectionLaplacian_gaussian(leftSetting,image,result)

     if selectedFilter == "Contour detection-Kirsch":
          contourDetectionKirsch(leftSetting,image,result)
    
     if selectedFilter == "Contour detection-Canny":
          contourDetectionCanny(leftSetting,image,result)

   

   