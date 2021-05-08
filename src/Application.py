from tkinter import *
from tkinter import ttk, filedialog
import tkinter.font as font
from Command import  *
from constants import *
from tkinter.filedialog import askopenfilename
from utils import *
from PIL import Image, ImageTk


""" GLOBAL OBJECTS """
initImage = "resources/images/lena1000p.jpg"
leftSetting = ""
image = ""
result = ""
rightSetting = ""
root = ""

#Load all images in the resources folder
loadAllImages()
#######################################
""" RESET THE UI """
def resetUI():
    root.destroy()
    app()

class ContainerPanel(PanedWindow):

    def __init__(self,topPanel=None,bottomPanel=None):
        super().__init__(orient=VERTICAL,bd=2)
        self.topPanel = topPanel
        self.bottomPanel = bottomPanel
        self.pack(fill=BOTH,expand=True)

    def addTopPanel(self,panel):
        self.topPanel = panel
        self.add(panel)
        self.topPanel.pack_propagate(0)
    def addBottomPanel(self,panel):
        self.bottomPanel = panel
        self.add(panel)
        self.bottomPanel.pack_propagate(0)
        
class settingPanel(PanedWindow):

    def __init__(self,master=None,hieght=260,lefFrame=None,rightFrame=None):
        super().__init__(master,orient=HORIZONTAL,bd=1,height=hieght)
        self.__rightFrame = rightFrame
        self.__lefFrame = lefFrame

    def getLeftFrame(self):
        return self.__lefFrame

    def getRightFrame(self):
        return self.__rightFrame
    def addLeftFrame(self,frame):
        self.__lefFrame = frame
        self.add(self.__lefFrame)

    def addRightFrame(self,frame):
        self.__rightFrame = frame
        self.add(self.__rightFrame)


class drawingPanel(PanedWindow):

    def __init__(self,master=None,imageFrame=None,resultFrame=None):
        super().__init__(master,orient=HORIZONTAL,bd=2)
        self.__imageFrame = imageFrame
        self.__resultFrame = resultFrame

    def getImageFrame(self):
        return self.__imageFrame

    def getResulttFrame(self):
        return self.__resultFrame

    def addImageFrame(self,frame):
        self.__imageFrame = frame
        self.add(self.__imageFrame)
        self.__imageFrame.pack_propagate(0) ### SAVED MY DAY !!!!!!!!!
    def addResultFrame(self,frame):
        self.__resultFrame = frame
        self.add(self.__resultFrame)
        self.__resultFrame.pack_propagate(0) ### SAVED MY DAY !!!!!!!!!
    
class imageFrame(LabelFrame):

    def __init__(self,master=None,title="",width=455):

        super().__init__(master,width=width,text=title,height=400)
        self.width = width
        self.master = master
        self.Oimage = Image.open(initImage)
        self.adjustImage(img=self.Oimage)
        self.OimagePil = self.Oimage
        self.Oimage = ImageTk.PhotoImage(self.Oimage)
        self.createCanvas()
        self.addOriginalImage(img=self.Oimage)

    def openAndAdjust(self,path):
        img = Image.open(path)
        img = resizeImage(img)
        self.setIomagePill(img)
        img = ImageTk.PhotoImage(img)
        self.Oimage = img
        self.addOriginalImage(img=self.Oimage)
    def setIomagePill(self,img):
        self.OimagePil = img

    def adjustImage(self,img=None):
         width, height = self.Oimage.size
         if width != 450 and height != 360:
             self.Oimage = self.Oimage.resize((450, 360))

    def createCanvas(self):
        """ canvas to display the original image """
        self.canvas = Canvas(self)
        self.canvas.pack(fill=BOTH,expand=2)

    def getWidth(self):
        return self.width
    def setWidth(self,width):
        self.width = width

    def addOriginalImage(self,img=None):
        """ add and display the original image to the canvas """
        self.Oimage = img
        self.canvas.create_image(0,0,anchor=NW,image=self.Oimage)

    def getOimage(self):
        return self.Oimage
    def getOimagePil(self):
        return self.OimagePil
class resultFrame(LabelFrame):
    def __init__(self,master=None,title="",width=455):

        super().__init__(master,width=width,text=title,height=400)
        self.__width = width
        self.pillImage = ""
        self.RImage = Image.open(initImage)
        self.adjustImage(img=self.RImage)
        self.RImage = ImageTk.PhotoImage(self.RImage)

        self.createCanvasR()
        self.addProccedImage(img=self.RImage)
    def setPillImage(self,im):
        self.pillImage = im.copy()
    def getPillImage(self):
        return self.pillImage
    def adjustImage(self,img=None):
        width, height = self.RImage.size
        if width != 450 and height != 360:

            self.RImage = self.RImage.resize((450, 360))


    def createCanvasR(self):
        """ canvas to deisplay the proccessed image """
        self.canvasR = Canvas(self)
        self.canvasR.pack(fill=BOTH,expand=2)

    def getWidth(self):
        return self.__width
    def addProccedImage(self,img=None):
        """ add and display the processed image """
        self.RImage = img
        self.canvasR.create_image(0,0,anchor=NW,image=self.RImage)

 #lef and right setting pannel************
class LeftPanel(LabelFrame):
     def __init__(self,master=None,width=650):
        super().__init__(master,width=width)
        self.width = width
        self.master = master
        self.FLAG = False
        self.create_widgets()

###############  CALLBACK TO OPEN FILE DDYNAMICALLY #################
     def OpenFile(self):
        self.FLAG = True
        self.Dimage = askopenfilename(initialdir="/",
                           filetypes =(("Image File", "*.png"),("All Files","*.*")),
                           title = "Choose a file."
                           )
        global image
        img = Image.open(self.Dimage)
        img = resizeImage(img)
        image.setIomagePill(img)
        img = ImageTk.PhotoImage(img)
        image.addOriginalImage(img=img)
#####################################################################

     def decideFilter(self,event=None):
         selectedV = self.getFilterType()
         if selectedV == "Elementary Transformation":
              self.comboFilterGoal["values"] = ELTRANSFORMATION
              self.comboFilterGoal.current(0)
         elif selectedV == "Spatial domain Filters":
            self.comboFilterGoal["values"] = FILTERSPATIAL
            self.comboFilterGoal.current(0)
         elif selectedV == "Frequency domain filters":
            self.comboFilterGoal["values"] = FILTRAGEFREQUENTIAL
            self.comboFilterGoal.current(0)
         elif selectedV == "Mathematical morphology":
            self.comboFilterGoal["values"] = MORPHOLOGIEM
            self.comboFilterGoal.current(0)
         elif selectedV == "Points of interest":
            self.comboFilterGoal["values"] = INTERESTP
            self.comboFilterGoal.current(0)


     def create_widgets(self):
        rFont = font.Font(size=12,family="Times New Roman",weight='bold')
        self.l1 = Label(self,text="Choose Image"+"       ",fg="#39409A",font=rFont,width=13)
        self.l2 = Label(self,text="Filter type"+"             ",fg="#39409A",font=rFont,width=13)
        self.l3 = Label(self,text="Choose Kernel "+"     ",fg="#39409A",font=rFont,width=13)
        self.l4 = Label(self,text="Filter/Operation    ",fg="#39409A",font=rFont,width=13)
        self.l5 = Label(self,text="Sigma        "+"            ",fg="#39409A",font=rFont,width=13)
        self.l6 = Label(self,text="Threshold       "+"       ",fg="#39409A",font=rFont,width=13)

        self.l1.grid(row=4,column=0,sticky="E")
        self.l2.grid(row=8,column=0)
        self.l3.grid(row=16,column=0)
        self.l4.grid(row=12,column=0)
        self.l5.grid(row=20,column=0)
        self.l6.grid(row=24,column=0)

        ################### ENTRIES ############################

        bFont = font.Font(size=11,family="Times New Roman")
        self.comboImageFile = ttk.Combobox(self,width=30,values=IMAGES,font=bFont)
        self.comboImageFile.bind('<<ComboboxSelected>>', self.updateImageFile)
        self.comboImageFile.grid(column=8,sticky=W,pady=4 ,padx=9, row=4)
        self.comboImageFile.current(0)

        self.comboImageFileD = Button(self,width=25,text="open image",font=bFont,relief=RIDGE,bd=0.5,command=self.OpenFile)
        self.comboImageFileD.grid(column=8,sticky=E,pady=4 ,padx=9, row=4)

        self.ll = Label(self,text="or",fg="#39409A",font=bFont)
        self.ll.grid(column=8,sticky=S,pady=4 ,padx=9, row=4)


        self.comboFilterGoal = ttk.Combobox(self,width=80,values=["--"],font=bFont)
        self.comboFilterGoal.grid(column=8,padx=10,pady=4, row=12)
        self.comboFilterGoal.current(0)

        self.comboFilterType = ttk.Combobox(self,width=80, values=FILTERSTYPE,font=bFont)
        self.comboFilterType.grid(column=8,padx=10,pady=4 ,row=8)
        self.comboFilterType.current(1)
        self.comboFilterType.bind('<<ComboboxSelected>>', self.decideFilter)


        self.comboKernel = ttk.Combobox(self,width=80, values=KERNEL,font=bFont)
        self.comboKernel.grid(column=8, padx=10,pady=4 ,row=16)
        self.comboKernel.current(0)


        self.sigma = IntVar()
        self.E1 = Entry(self,width=82, bd =1,textvariable=self.sigma,font=bFont)
        self.E1.grid(column=8,padx=10, row=20)

        self.var = DoubleVar()
        self.treshold = Scale(self ,from_=0,tickinterval=50, to=256, orient=HORIZONTAL,variable =self.var, length=256, resolution=1)
        self.treshold.grid(column= 8,sticky="W",padx=10,row=24)

        #activate or disactivate treshhold
        self.TVal  = IntVar()
        self.c1 = Checkbutton(self, text='Enable-Disable',variable=self.TVal, onvalue=1, offvalue=0,font=bFont)
        self.c1.grid(column=8,sticky="E",padx=10,row=24)

        self.Itype = StringVar()
        self.Itype.set("GREY")
        self.c2 = Radiobutton(self, text='Grey Image',variable=self.Itype,value=IMAGETYPE[0],font=bFont)
        self.c2.grid(column=8,row=25,padx=10,sticky="W")

        self.c3 = Radiobutton(self, text='RGB Image',variable=self.Itype,value=IMAGETYPE[1],font=bFont)
        self.c3.grid(column=8,row=25,padx=10,sticky="S")

        self.c4 =  Radiobutton(self, text='Binary Image',variable=self.Itype,value=IMAGETYPE[2],font=bFont)
        self.c4.grid(column=8,row=25,padx=10,sticky="E")

    ############### GETTERS AND SETTERS ############################

     def updateImageFile(self,event=None):
         global image
         url = "resources/images/"+self.comboImageFile.get()
         image.openAndAdjust(url)

     def getImagefile(self):
        if self.FLAG:
            return self.Dimage
        else:
            self.FLAG = False
            return self.comboImageFile.get()

     def getFilterType(self):
         return self.comboFilterType.get()
     def getTreshold(self):
         return self.var
     def getTresholdState(self):
         return self.TVal
     def getSigma(self):
         return self.sigma
     def getFilterGoal(self):
         return self.comboFilterGoal.get()
     def getImageType(self):
         return self.Itype
     def getKernel(self):
         return self.comboKernel.get()

class RightPanel(LabelFrame):
     def __init__(self,master=None,title="",width=150):
        super().__init__(master,width=width)
        self.width = width
        self.master = master
        self.create_widgets()

     def create_widgets(self):
        self.run = Button(self,text="Run",relief=RIDGE,bg="#12A53E",bd=0.5,fg="#fff",width=20,command=self.doRun)
        rFont = font.Font(size=10,family="Helvetica",weight='bold')
        self.run["font"] = rFont
        self.run.grid(column=0,pady=10, row=0,sticky="W")

        self.histograme = Button(self,text="Histogramme",relief=RIDGE,bd=0.5,bg="#0FA7F3",fg="#fff",width=20,command=self.doHistograme)
        self.histograme["font"] = rFont
        self.histograme.grid(column=0,pady=10, row=1,sticky="W")

        self.runOnCv2 = Button(self,text="Open with OpenCv",font=rFont,relief=RIDGE,bd=0.5,bg="#F6B910",fg="#fff",width=20,command=self.doOpenCv)
        self.runOnCv2.grid(column=0, row=2,pady=10, sticky="W")

        self.saveImage = Button(self,text=" Save ",font=rFont,relief=RIDGE,bd=0.5,bg="#F27B49",fg="#fff",width=20,command=self.doSave)
        self.saveImage.grid(column=0, row=3,pady=10, sticky="W")

        self.img = PhotoImage(file="resources/reset_1_30x30.png")
        self.img1 = self.img.subsample(1,1)
        self.reset = Button(self,image=self.img1,relief=RIDGE,bd=0,width=30,font=rFont,command=resetUI)
        self.reset.grid(column=0, row=4,pady=8, sticky="S")

    ############################COMMANDS ########################
     def doRun(self):
        runCommandDispatcher(root,leftSetting,image,result)
     def doHistograme(self):
        processHistogramCommand(leftSetting,image,result)
     def doOpenCv(self):
        processOpenCvCommand(leftSetting,image,result)
     def doSave(self):
        processSaveCommand(leftSetting,image,result)
    ############################################################

#****************************************
class Application(Frame):
    def __init__(self, master=None,panel=None):
        super().__init__(master)
        self.master = master
        self.__panel = panel
        self.pack()

    def getPanel(self):
        return self.__panel

def app():

    global leftSetting
    global image
    global result
    global rightSetting
    global root
    root = Tk()
    icon = PhotoImage(file = "resources/icon.png")
    root.iconphoto(False, icon)
    root.resizable(False,False)
    app = Application(master=root)
    app.master.title("Image Processing : Supervisor : Pr. Hamid TAIRI - MIDVI - HALIM ZAAIM")
    app.master.minsize(910,650)
    app.master.maxsize(910,650)

    #create frame
    container = ContainerPanel()
    drawing = drawingPanel(master=container)
    setting = settingPanel(master=container)
    container.addTopPanel(setting)
    container.addBottomPanel(drawing)

    image = imageFrame(master=drawing,title="Original Image")
    result = resultFrame(master=drawing,title="Processed Image")
    drawing.addImageFrame(image)
    drawing.addResultFrame(result)

    #setting frame
    leftSetting = LeftPanel(master=setting)
    rightSetting = RightPanel(master=setting)
    setting.addLeftFrame(leftSetting)
    setting.addRightFrame(rightSetting)

    app.mainloop()

if __name__ == '__main__':
    app()