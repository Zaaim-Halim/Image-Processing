from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import utils
import random
import cv2

################################### Modal #########################
class MyDialog:
    def __init__(self, parent, title=None,labeltext=""):
        self.top = Toplevel(parent)
        parent.eval(f'tk::PlaceWindow {str(self.top)} center')
        self.top.transient(parent)
        self.top.grab_set()
        if len(title) > 0: self.top.title(title)
     
        self.var = IntVar()
        self.treshold = Scale(self.top,from_=-100,tickinterval=50, to=100, orient=HORIZONTAL,variable =self.var, length=200, resolution=1)
        self.treshold.grid(column= 0,sticky="W",padx=10,row=1)
       
        self.treshold.bind("<Return>", self.ok)
        self.treshold.bind("<Escape>", self.cancel)
        
        b = Button(self.top, text="OK",relief=RIDGE,bd=0.6,width=6,command=self.ok)
        b.grid(row=2,column=0,padx=20,pady=4)
 
    def ok(self, event=None):
        self.top.destroy()
 
    def cancel(self, event=None):
        self.top.destroy()

    def getVar(self):
         return self.var.get()
 
def dialogo(root):
     d = MyDialog(root, "Brightness", " ")
     root.wait_window(d.top)
     return d.getVar()
     
######################################## TO GERY ##################
def toGreyScal(image):
     img = utils.pillToCv2(image)
     if not utils.is_grey_scale(image):
          grayImage = np.zeros(img.shape)
          R = np.array(img[:, :, 0])
          G = np.array(img[:, :, 1])
          B = np.array(img[:, :, 2])

          R = (R *.299)
          G = (G *.587)
          B = (B *.114)

          Avg = (R+G+B)
          grayImage = img
          for i in range(3):
               grayImage[:,:,i] = Avg
          return Image.fromarray(grayImage)
     else:
          return image

###############################################################

def  colorInversion(leftSetting,image,result):
     img = image.getOimagePil()
     pixels = utils.pillToCv2(img)
     for i in range(pixels.shape[0]):
          for j in range(pixels.shape[1]):
               x,y,z = pixels[i,j][0],pixels[i,j][1],pixels[i,j][2]
               x,y,z = abs(x-255), abs(y-255), abs(z-255)
               pixels[i,j] = (x,y,z)
     img_final = cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR)
     result.setPillImage(Image.fromarray(img_final))
     image = ImageTk.PhotoImage(Image.fromarray(img_final))
     result.addProccedImage(image)

def  grayScaleImage(leftSetting,image,result):
     img1 = image.getOimagePil()
     img = utils.pillToCv2(img1)
     grayImage = np.zeros(img.shape)
     R = np.array(img[:, :, 0])
     G = np.array(img[:, :, 1])
     B = np.array(img[:, :, 2])

     R = (R *.299)
     G = (G *.587)
     B = (B *.114)

     Avg = (R+G+B)
     grayImage = img

     for i in range(3):
          grayImage[:,:,i] = Avg
     
     result.setPillImage(Image.fromarray(grayImage))
     image = ImageTk.PhotoImage(Image.fromarray(grayImage))
     result.addProccedImage(image)

def  binarizationByThresholding(leftSetting,image,result):
     threshold = int(leftSetting.getTreshold().get())
     img1 = image.getOimagePil()
     img = toGreyScal(img1)
     pixels = img.load()
     
     if utils.pillToCv2(img1).shape[2]==3: # we could just transform the image to grayscale using pill or cv2
          for i in range(img.size[0]):
               for j in range(img.size[1]):
                    x,y,z = pixels[i,j][0],pixels[i,j][1],pixels[i,j][2]
                    if x > threshold:
                         x = 255
                    else:
                         x = 0
                    if y > threshold:
                         y = 255
                    else:
                         y = 0
                    if z > threshold:
                         z = 255
                    else:
                         z = 0
                    pixels[i,j] = (x,y,z)
     else:
          for i in range(img.size[0]):
               for j in range(img.size[1]):
                    x = pixels[i,j]
                    if x > threshold:
                         x = 255
                    else:
                         x = 0
                    pixels[i,j] = x
                   
     result.setPillImage(img)
     image = ImageTk.PhotoImage(img)
     result.addProccedImage(image)

def gaussianNoise(leftSetting,image,result):
     img1 = image.getOimagePil()
     img = utils.pillToCv2(img1)
     row,col,ch= img.shape
     mean = 0
     var = 0.1
     sigma = var**0.6
     gauss = np.random.normal(mean,sigma,img.shape)
     gauss = gauss.reshape(row,col,ch)
     noisyImage = img + gauss
     result.setPillImage(Image.fromarray((noisyImage * 255).astype(np.uint8)))
     image = ImageTk.PhotoImage(Image.fromarray((noisyImage * 255).astype(np.uint8)))
     result.addProccedImage(image)

def  poivreAndSelNoise(leftSetting,image,result):
     img1 = image.getOimagePil()
     img = utils.pillToCv2(img1)
     # Getting the dimensions of the image
     row , col, ch = img.shape
      
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
     number_of_pixels = random.randint(300, 10000)
     for i in range(number_of_pixels):
        
        y_coord=random.randint(0, row - 1)
          
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
          
        # Color that pixel to white
        img[y_coord][x_coord] = 255
          
    # Randomly pick some pixel
     number_of_pixels = random.randint(300 , 10000)
     for i in range(number_of_pixels):
        
          y_coord=random.randint(0, row - 1)
          
          x_coord=random.randint(0, col - 1)
          
          img[y_coord][x_coord] = 0
     result.setPillImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))    
     image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
     result.addProccedImage(image)
def contrastenhancement(leftSetting,image,result):
     img1 = image.getOimagePil()
     img = utils.pillToCv2(img1)
     if not utils.is_grey_scale(img1):
          r_image, g_image, b_image = cv2.split(img)

          r_image_eq = cv2.equalizeHist(r_image)
          g_image_eq = cv2.equalizeHist(g_image)
          b_image_eq = cv2.equalizeHist(b_image)
          image_eq = cv2.merge((r_image_eq, g_image_eq, b_image_eq))
          
     else:
          image_eq = cv2.equalizeHist(img)
     result.setPillImage(Image.fromarray(cv2.cvtColor(image_eq, cv2.COLOR_BGR2RGB)))    
     image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(image_eq, cv2.COLOR_BGR2RGB)))
     result.addProccedImage(image)

def adjustBrightness(root,leftSetting,image,result):
     img1 = image.getOimagePil()
     img = utils.pillToCv2(img1)
     var = dialogo(root)
     img_final = img + var
     img_final = np.uint8(np.absolute(img_final))
     img_final = cv2.cvtColor(img_final,cv2.COLOR_RGB2BGR)
     result.setPillImage(Image.fromarray(img_final))
     image = ImageTk.PhotoImage(Image.fromarray(img_final))
     result.addProccedImage(image)
def  ElementaryTransformationCMD(root,leftSetting,image,result):
    
     selectedFilter = leftSetting.getFilterGoal()
     if selectedFilter == "Color inversion":
        colorInversion(leftSetting,image,result)

     if selectedFilter == "Gray Scale image":
          grayScaleImage(leftSetting,image,result)
       
     if selectedFilter == "Binarization by thresholding":
          binarizationByThresholding(leftSetting,image,result)
        
    
     if selectedFilter == "Gaussian-Noise":
          gaussianNoise(leftSetting,image,result)
     
     if selectedFilter == "Poivre&Sel-Noise":
          poivreAndSelNoise(leftSetting,image,result)
     
     if selectedFilter == "Contrast enhancement":
          contrastenhancement(leftSetting,image,result)
     
     if selectedFilter == "Adjust Image Brightness":
          adjustBrightness(root,leftSetting,image,result)
    

    