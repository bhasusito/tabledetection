# -*- coding: utf-8 -*-
import os
import cv2
import datetime
import time
import shutil
from multiprocessing import Process
from threading import Thread
#from main import skewCorrection
#from main import deNoise
#from TableExtraction import preprocess
#from TableExtraction_v2_2_0 import processtable
import TableExtraction_v2_2_0 as te
#from TableExtraction import PDF2Png_gs
#from FolderSeparation import Process_main
import sys
import MultiTable as mt

def Create_Process(ImageA,threadval,PageInc):
	#print("Length",len(ImageA))
	cmpthreads=[]
	#print ("Thread ", PageInc)
	strttime = time.time()
	
	for i in range(0,threadval):
		t=Thread(target=File_Process,args=(ImageA[i+PageInc],))
		cmpthreads.append(t)
		t.start()
	
	for i in range(0,threadval):
		cmpthreads[i].join()
		
	#print ("Thread End Time :: ", time.time()-strttime)


def getImages(Path):
    images = []    
    xtnList = [".jpg", ".JPG", ".png" , ".TIFF"]
    files = os.listdir(Path)
    for i,eachfile in enumerate(files):
        fileXtn = os.path.splitext(eachfile)[1]
        if fileXtn in xtnList:
            imgpath = os.path.join(DIRECTORY, eachfile)
            images.append(imgpath)
        
    return images
            
'''
def Imgappend(PathA):
    Image=[]
    #print(os.listdir(PathA))
    imgCount = 0
    for path, subdirs, files in os.walk(PathA):
          for name in files:
            if (name.endswith(".png")) or (name.endswith(".jpg")) or (name.endswith(".JPG"))or (name.endswith(".tif")) or (name.endswith(".tiff")) or (name.endswith(".TIFF")):
            	imgCount += 1
    imgCount +=1
    for path, subdirs, files in os.walk(PathA):
	    for i in range(1,imgCount):
	           for name in files:            	
            	       removeExtn = name[:-4]
            	       getnum = int(removeExtn[-6:])
            	       #print(getnum)
            	       if(i == getnum):
            	         ctimage=os.path.join(path, name)
            	         Image.append(ctimage)
            	         break
    #print("Image sorting",Image)
    return Image
'''

def File_Process(Inp_Img):
    DIRECTORY,eachfile=os.path.split(Inp_Img)
    imgpath = os.path.join(DIRECTORY, Inp_Img)
    fileXtn = os.path.splitext(eachfile)[1]
    imgname = os.path.splitext(eachfile)[0]
    rawImg = cv2.imread(imgpath)
    outFolder = os.path.join(DIRECTORY, Inp_Img[:-4])
    if os.path.exists(outFolder):
        shutil.rmtree(outFolder)
    os.mkdir(outFolder)
    outPath = os.path.join(outFolder, eachfile)
    Tables, ImageTables = mt.tableMain(rawImg.copy(), imgpath, outPath)
    tableDrawn = mt.test(eachfile,Tables, ImageTables, rawImg.copy(), outFolder)
    tablesaveName = imgname + "_out.jpg"
    tabledrawnSavepath = os.path.join(outFolder, tablesaveName)
    cv2.imwrite(tabledrawnSavepath, tableDrawn)
    print("------table extraction completed--------\n")
    
    
def File_Process_old(Inp_Img):
#    #src = cv2.imread(Inp_Img)
#    #print(src)
#    #input("")
#    #img = preprocess(Inp_Img)
#    #grayImage = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
#    #h, w = grayImage.shape[:2]
    DIRECTORY,eachfile=os.path.split(Inp_Img)
#    #imgpath = DIRECTORY + "/"+eachfile[:-4]
#    outFolder = os.path.join(DIRECTORY, eachfile[:-4])
#    if os.path.exists(outFolder):
#     shutil.rmtree(outFolder)
#    os.mkdir(outFolder)
#    outPath = os.path.join(outFolder, eachfile)
#    shutil.copyfile(Inp_Img, outPath)
#    #result = processtable(grayImage,src,outPath)
#    #input("")
    imgpath = os.path.join(DIRECTORY, Inp_Img)
    rawImg = cv2.imread(imgpath)
    outFolder = os.path.join(DIRECTORY, Inp_Img[:-4])
    if os.path.exists(outFolder):
        shutil.rmtree(outFolder)
    os.mkdir(outFolder)
    outPath = os.path.join(outFolder, eachfile)
    
    '''Noise removal'''
    #denoisedSrc = deNoise(imgpath, outFolder)
    #denoisedSrc = cv2.imread(imgpath)
#    print('denoised shape', denoisedSrc.shape)
    '''Skew correction'''
    #skewC = skewCorrection(denoisedSrc)
    ''' Table detection '''
    #tablecorners(grayImage,outPath)
    #result = processtable(skewC, outPath, rawImg)    
    result = te.processtable(rawImg, outPath, rawImg)   
    
    return
def Thread_Initiation(ImageFiles):
    a1 = datetime.datetime.now()
    print("func start time",a1)
    threadval=5
    Quot=int(len(ImageFiles)/threadval)
    Remind = (len(ImageFiles)%threadval)
    m_bMultiThreads = True
    m_bMultiProcess = False
    
    if m_bMultiThreads == True:
            if m_bMultiProcess == True:
                    cmpProcess=[]
                    PageInc = 0
                    strttime = time.time()
                    for j in range(0, Quot):
                            proc = Process(target=Create_Process, args=(ImageFiles,threadval,PageInc))
                            cmpProcess.append(proc)
                            proc.start()
                            PageInc = threadval*(j+1)
                    for proc in cmpProcess:
                            proc.join()
                    print ("Process End Time :: ", time.time()-strttime)
            else:
                    PageInc = 0
                    for j in range(0, Quot):
                            cmpthreads=[]
                            strttime = time.time()
                            print ("Thread ", j+1)
                            print ("Start Time :: ", strttime)
                            for i in range(0,threadval):
                                    #print("ImageFiles",ImageFiles[i+PageInc])
                                    t=Thread(target=File_Process,args=(ImageFiles[i+PageInc],))
                                    cmpthreads.append(t)
                    #print("Complete append",threads)
                            for i in range(0,threadval):
                                    cmpthreads[i].start()
                                    
                            for i in range(0,threadval):
                                    cmpthreads[i].join()
                            PageInc = threadval*(j+1)
                            print ("End Time :: ", time.time()-strttime)
                            print("PageInc",PageInc)
            ################
            cmpthreads=[]
            strttime = time.time()
            for i in range(0,Remind):
                t=Thread(target=File_Process,args=(ImageFiles[i+PageInc],))
                cmpthreads.append(t)
                #print("Complete append",threads)
            for i in range(0,Remind):
                cmpthreads[i].start()
                    
            for i in range(0,Remind):
                cmpthreads[i].join()
            print ("End Time :: ", time.time()-strttime)

    else:
	    for i in range(0, len(ImageFiles)):
	        File_Process(ImageFiles[i])
                    
    b1 = datetime.datetime.now()
    print("func End time",b1)
    print("Comapre func time",b1-a1)

if __name__ == "__main__":
    DIRECTORY= r'C:\ICDAR\Code\WhitespaceIntegrated\test'
    #DIRECTORY=sys.argv[1]
    print(DIRECTORY)
    ImageFiles=os.listdir(DIRECTORY)
    print(ImageFiles)
    ImageName=[]
#    for tt in range(len(ImageFiles)):
#        ImageName.append(os.path.join(DIRECTORY,ImageFiles[tt]))
#    Thread_Initiation(ImageName)
    
    images = getImages(DIRECTORY)
    #print(images)
    Thread_Initiation(images)
   