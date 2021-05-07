
import cv2
import numpy as np
import math
from scipy import signal
from PIL import Image

def pillToCv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
def resizeImage(img):
    width, height = img.size
    if width != 450 and height != 360:
        return img.resize((450, 360))
    return img

def cv2ToPill(img):
    return Image.fromarray(img)     
        
def is_grey_scale(img):
    cvImg = pillToCv2(img)
    _, _, ch =cvImg.shape
    if ch<3:
        return True 
    else: 
        # just to make  sure
        w, h, _ = cvImg.shape
        b ,g, r = cv2.split(cvImg)
        for i in range(w):
            for j in range(h):
                r_pixel , g_pixel, b_pixel = r[i,j],g[i,j],b[i,j]
                if not g_pixel == b_pixel == r_pixel: 
                    return False 
                    
    return True 

def convolve2DRGB(data,filter,padding=0,strides=1):
    h, w, _ = data.shape
    r,g,b = cv2.split(data)
    
    r_img = np.zeros((h,w),dtype=np.uint8)
    b_img = np.zeros((h,w),dtype=np.uint8)
    g_img = np.zeros((h,w),dtype=np.uint8)
    
    xKernShape = filter.shape[0]
    yKernShape = filter.shape[1]
    xImgShape = data.shape[0]
    yImgShape = data.shape[1]
    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((data.shape[0] + padding*2, data.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = data
    else:
        imagePaddedr = r
        imagePaddedg = g
        imagePaddedb = b
    # Iterate through image
    for y in range(data.shape[1]):
        # Exit Convolution
        if y > data.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(data.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > data.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        r_img[x, y] = (filter * imagePaddedr[x: x + xKernShape, y: y + yKernShape]).sum()
                        b_img[x, y] = (filter * imagePaddedb[x: x + xKernShape, y: y + yKernShape]).sum()
                        g_img[x, y] = (filter * imagePaddedg[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    final_img = cv2.merge((r_img,g_img,b_img))
    return final_img

def convolve2DRGB_float64(data,filter,padding=0,strides=1):
    h, w, ch = data.shape
    r,g,b = cv2.split(data)
    
    r_img = np.zeros((h,w),dtype=np.float64)
    b_img = np.zeros((h,w),dtype=np.float64)
    g_img = np.zeros((h,w),dtype=np.float64)
    
    xKernShape = filter.shape[0]
    yKernShape = filter.shape[1]
    xImgShape = data.shape[0]
    yImgShape = data.shape[1]
    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((data.shape[0] + padding*2, data.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = data
    else:
        imagePaddedr = r
        imagePaddedg = g
        imagePaddedb = b
    # Iterate through image
    for y in range(data.shape[1]):
        # Exit Convolution
        if y > data.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(data.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > data.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        r_img[x, y] = (filter * imagePaddedr[x: x + xKernShape, y: y + yKernShape]).sum()
                        b_img[x, y] = (filter * imagePaddedb[x: x + xKernShape, y: y + yKernShape]).sum()
                        g_img[x, y] = (filter * imagePaddedg[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    final_img = cv2.merge((r_img,g_img,b_img))
    return final_img

def gaussian_kernel(sigma, size):
    mu = np.floor([size / 2, size / 2])
    size = int(size)
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(-(0.5/(sigma*sigma)) * (np.square(i-mu[0]) + 
            np.square(j-mu[0]))) / np.sqrt(2*math.pi*sigma*sigma)

    kernel = kernel/np.sum(kernel)
    return kernel

def convolve2DGREY(image, mask):
	width = image.shape[1]
	height = image.shape[0]
	w_range = int(math.floor(mask.shape[0]/2))

	res_image = np.zeros((height, width),dtype=np.uint8)

	# Iterate over every pixel that can be covered by the mask
	for i in range(w_range,width-w_range):
		for j in range(w_range,height-w_range):
			# Then convolute with the mask 
			for k in range(-w_range,w_range):
				for h in range(-w_range,w_range):
					res_image[j, i] += mask[w_range+h,w_range+k]*image[j+h,i+k]
	return res_image

def LaplacianKernel(kernel_size):
    if kernel_size == 3:
        return np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    else:
        return np.array([[-1,-3,-4,-3,-1],[-3,0,6,0,-3],[-4,6,20,6,-4],[-3,0,6,0,-3],[-1,-3,-4,-3,-1]])

def sobelKernel(kernel_size):
    a = np.array([ 1, 2, 1])
    aa = np.array([ 1, 2, 1])[np.newaxis]
    b = np.array([1, 0, -1])
    bb = np.array([1, 0, -1])[np.newaxis]
    sob3x3_x = aa.T*b
    sob3x3_y = bb.T*a
    if kernel_size == 3:
        return (sob3x3_x,sob3x3_y)
    elif kernel_size == 5:
        sob5x5_x = signal.convolve2d(aa.T*a, sob3x3_x)
        sob5x5_y = signal.convolve2d(aa.T*a, sob3x3_y)
        return (sob5x5_x,sob5x5_y)
    else:
        sob5x5_x = signal.convolve2d(aa.T*a, sob3x3_x)
        sob5x5_y = signal.convolve2d(aa.T*a, sob3x3_y)

        sob7x7_x = signal.convolve2d(aa.T*a, sob5x5_x)
        sob7x7_y = signal.convolve2d(aa.T*a, sob5x5_y)

        sob9x9_x = signal.convolve2d(aa.T*a, sob7x7_x)
        sob9x9_y = signal.convolve2d(aa.T*a, sob7x7_y)
        return (sob9x9_x,sob9x9_y)
def perwittKernel(kernel_size):
    # only 3x3 and 5x5  kernel
    if kernel_size == 3:

        a = np.array([[ 1, 1, 1],[ 1, 1, 1],[ 1, 1, 1]])
        aa = np.array([ -1, 0, 1])[np.newaxis]
        return (a*aa.T,a*aa)
    else:
        px = np.array([[2,2,2,2,2],[1,1,1,1,1],[0,0,0,0,0],[-1,-1,-1,-1,-1],[-2,-2,-2,-2,-2]])
        py = np.array([[2,1,0,-1,-2],[2,1,0,-1,-2],[2,1,0,-1,-2],[2,1,0,-1,-2],[2,1,0,-1,-2]])
        return (px,py)
def robertKernel(kernel_size):
    roberts_cross_v = np.array( [[ 0, 0, 0 ],
                             [ 0, 1, 0 ],
                             [ 0, 0,-1 ]] )

    roberts_cross_h = np.array( [[ 0, 0, 0 ],
                             [ 0, 0, 1 ],
                             [ 0,-1, 0 ]] )
    return (roberts_cross_v,roberts_cross_h)

################################# laplacian of gaussion #############
def logkernel(kernel_size):
    if kernel_size == 3:

        log = np.array([[1,1,1],[1,-8,1],[1,1,1]])
        return log
    elif kernel_size == 5:
        log = np.array([[0,0,1,0,0],[0,1,2,1,0],[1,2,-16,2,1],[0,1,2,1,0],[0,0,1,0,0]])
        return log
    else:
        #for a Gaussian sigma = 1.4
        log = np.array([[0,1,1,2,2,2,1,1,0],[1,2,4,5,5,5,4,2,1],
                [1,4,5,3,0,3,5,4,1],[2,5,3,-12,-24,-12,3,5,2],
                [2,5,0,-24,-40,-24,0,5,2],[2,5,3,-12,-24,-12,3,5,2],
                [1,4,5,3,0,3,5,4,1],[1,2,4,5,5,5,4,2,1],[0,1,1,2,2,2,1,1,0]])
        return log

def decideKernel(leftSetting):
    kernel  = leftSetting.getKernel()
    if kernel=="3X3":
        return 3
    elif kernel =="5X5":
        return 5
    elif kernel=="9X9":
        return 9

def distance(point1,point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

###########################################################################

def convert_binary(image_src, thresh_val):
    color_1 = 255
    color_2 = 0
    initial_conv = np.where((image_src <= thresh_val), image_src, color_1)
    final_conv = np.where((initial_conv > thresh_val), initial_conv, color_2)
    return final_conv



############################ erosion - deltation operation#############
def add_padding(image, padding, value):
    return cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=value)

def operation(data, kernel,operation=None):
    kernel_size = kernel.shape[0]
    padding = int(kernel_size // 2)
    if operation:
        img_operated = data
        dilation_flag = False
        erosion_flag = False
        padding_value = 0          
        if operation == "erosion": 
            padding_value = 255
            erosion_flag = True
        elif operation == "dilation":
            dilation_flag = True

        padded = add_padding(data, padding, padding_value)  
        vertical_window = padded.shape[0] - kernel.shape[0] #final vertical window position
        horizontal_window = padded.shape[1] - kernel.shape[1] #final horizontal window position

        #start with vertical window at 0 position
        vertical_pos = 0 + kernel.shape[0]//2

        #sliding the window vertically
        while vertical_pos <= vertical_window:
            horizontal_pos = 0 + kernel.shape[1]//2

            #sliding the window horizontally
            while horizontal_pos <= horizontal_window:
                
                 # the window to be operated on 
                rstart = vertical_pos - kernel.shape[0]//2
                rend = vertical_pos + kernel.shape[0]//2
                    
                cstart = horizontal_pos - kernel.shape[1]//2
                cend = horizontal_pos + kernel.shape[1]//2
                    
                if operation == "erosion" and erosion_flag:
                    win = img_operated[rstart:rend+1,cstart:cend+1]
                    minv = minElementInWindow(window=win) 
                    img_operated[vertical_pos, horizontal_pos] = minv
    

                    #if operation is dilation and we find a match, then break the first 'for' loop 
                if operation == "dilation" and dilation_flag:       
                    win = img_operated[rstart:rend+1,cstart:cend+1]
                    minv = maxElementInWindow(window=win) 
                    img_operated[vertical_pos, horizontal_pos] = minv
    
                #increase the horizontal window position
                horizontal_pos += 1

            #increase the vertical window position
            vertical_pos += 1
        return img_operated

def maxElementInWindow(window=None):
    window = np.array(window)
    array_1d = window.flatten()
    return np.max(array_1d)
def minElementInWindow(window=None):
    window = np.array(window)
    array_1d = window.flatten()
    min = array_1d.min()
    return min

#######################################  corner detectors ########################
def reduceReduis(radius):
    if radius == 3:
        return 2
    elif radius == 5:
        return 3
    else:
        return 5
def susanNucleus(radius):
    kernel = np.zeros((2*radius+1, 2*radius+1) ,np.uint8)
    y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel[mask] = 1
    kernel[0,radius-1:kernel.shape[1]-radius+1] = 1
    kernel[kernel.shape[0]-1,radius-1:kernel.shape[1]-radius+1]= 1
    kernel[radius-1:kernel.shape[0]-radius+1,0] = 1
    kernel[radius-1:kernel.shape[0]-radius+1,kernel.shape[1]-1] = 1
    return kernel

def susanSum(kernel):
    return kernel.sum()

def susanTreshold(window):
    alpha = 0.02
    window = np.array(window)
    maxf = window.flatten().max()
    minf = window.flatten().min()
    return minf+alpha*(maxf-minf)
