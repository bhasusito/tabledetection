# -*- coding: utf-8 -*-
import TableExtraction_v2_2_0 as tb #te
import cv2
#import Cell_Detection_tableV1_1 as cd
#import CellDetectionbr as cd
import TableRoiExtractionV3 as tr
import numpy as np
import  os, shutil
import xml.etree.ElementTree as ET
from geometry import Rect
import time
from colorama import Fore, Back, Style,init
    
def stats2rect(ca):
    x = ca[0]
    y = ca[1]
    w = ca[2]
    h = ca[3]
    return x, y, w, h

def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

def noise_removal(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img,80,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    kernel = np.ones((7,7), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours.sort(key=lambda x:get_contour_precedence(x, dilated.shape[1])) 
    for i in range(len(contours)):
        cnt = contours[i]
        [x,y,w,h] = cv2.boundingRect(cnt)
        imgBcrop=img[y:h+y,x:w+x]
        if(h>10) :
            _, blackAndWhite = cv2.threshold(imgBcrop, 127, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((1,1), np.uint8)
            dilated = cv2.dilate(blackAndWhite, kernel, iterations=1)
            nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, None, None, None, 4, cv2.CV_32S)
            if len(labels)>200:
                sizes = stats[1:, -1] 
                img2 = np.zeros((labels.shape), np.uint8)
                for i in range(0, nlabels - 1):
                    if sizes[i] >= 15:   
                        img2[labels == i + 1] = 255
                res = cv2.bitwise_not(img2)
                img[y:h+y,x:w+x] = res
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def test(imgName, tables, ImageTables, rawImg, outpath):
    tablesDetected = []
    
    #print(ImageTables)
    #print(tables)
    if ImageTables != None or len(tables) > 0:
        for i, eachtable in enumerate(ImageTables):
            #print("table object: ",eachtable)
            if not eachtable:
                continue
            if eachtable != None:
                   
                        print("eachtable.tablecoordinates:",eachtable.tablecoordinates)
                        if len(eachtable.tablecoordinates)> 0:
                            tX = eachtable.tablecoordinates[0]; tY = eachtable.tablecoordinates[1]; 
                            tW = eachtable.tablecoordinates[2]; tH = eachtable.tablecoordinates[3];
                            tablesDetected.append([tX, tY, tW, tH])
                            cv2.rectangle(rawImg, (tX, tY), (tW + tX, tH+tY), (255,0,0), 3)
        
        if tables != None:
            for i, rowCells in enumerate(tables):
                for cells in rowCells : 
                    for cell in cells :
                        cv2.rectangle(rawImg, (cell[0],cell[1]), (cell[2],cell[3]), (0, 0, 255), 2)
    saveXmlPath = outpath + '\\' + imgName + '.xml'
    #saveXmlPath = outpath + '\\' + os.path.splitext(imgName)[0] + '.xml'
    print('xml out path', saveXmlPath)          
    CreateXml_TrackA(imgName, saveXmlPath, tablesDetected, tables)
    return rawImg

def CreateXml_TrackA(imgName, saveXmlPath, tablesDetected, Tables):
     versionElement  = ET.Element("xml",  version="1.0", encoding="UTF-8")
     root = ET.Element("document",  filename=imgName)
     
     
     for i,tableRoi in enumerate(tablesDetected):
         x,y,w,h = stats2rect(tableRoi)
         table = ET.SubElement(root,"table")
         r = x + w; b = y + h
         #[x,y] [x,b] [r,b] [r,y]
         xy = str(x) + "," + str(y)
         xb = str(  x) + "," + str(b)
         rb = str(r) + "," + str(b)
         ry = str(r) + "," + str(y)
         #createPoints = str(x) + "," +str(y)+ "," +str(w)+ "," +str(h)
         createPoints = xy + " " + ry + " " + rb  + " " + xb
         coords  = ET.SubElement(table, "Coords ",points= createPoints)
         
#     for i, rowCells in enumerate(Tables):
         for j, cells in enumerate(Tables[i]):
             for k, cell1 in enumerate(cells):
                  cell = ET.SubElement(table, "cell")#,end-col= k, start-col= k, end-row= j, start-row= j)
                  cell.set('end-col', str(k))
                  cell.set('start-col', str(k))
                  cell.set('end-row', str(j))
                  cell.set('start-row', str(j))
                  x,y,w,h = stats2rect(cell1)
                 # r = x + w; b = y + h
                  #[x,y] [x,b] [r,b] [r,y]
#                  xy = str(x) + "," + str(y)
#                  xb = str(x) + "," + str(b)
#                  rb = str(r) + "," + str(b)
#                  ry = str(r) + "," + str(y)
                  xy = str(x) + "," + str(y)
                  xb = str(w) + "," + str(y)
                  rb = str(w) + "," + str(h)
                  ry = str(x) + "," + str(h) 
                  createPoints = xy + " " + xb + " " + rb  + " " + ry
                  cellCoords  = ET.SubElement(cell, "Coords ",points= createPoints)
          
     tree = ET.ElementTree(versionElement)
     tree = ET.ElementTree(root)
     tree.write(saveXmlPath,encoding="utf-8",xml_declaration=True,method='xml')

def readCoordinatesFromXml(xmlFilePath):
    tree = ET.parse(xmlFilePath)
    docRoot = tree.getroot()
    gt_RoiList = []
    try:
        if(docRoot.tag == "document"):
            for table in docRoot:
                if(table.tag == "table"):
                    for coord in table:
                        if(coord.tag == "Coords"):
                            tableRois = coord.attrib['points'].split()
                            xy = tableRois[0].split(',')
                            x = int(xy[0]) ; y = int(xy[1])
                            rb = tableRois[2].split(',')
                            r = int(rb[0]); b = int(rb[1])
                            gt_RoiList.append([x,y,r,b])
        return gt_RoiList
    except IOError:
        print("Failed to save txt file.")
        
def linebasedTableDetection(srcImg):
    grayImage = cv2.cvtColor(srcImg,cv2.COLOR_BGR2GRAY)
#    tablecoord = cd.multicell(grayImage)
#    tablecoord, lineOutImg = cd.LineBasedTable(grayImage) #cd1.fillimage(grayImage)
    tablecoord,Rowcellcoord,multicell,colorimg=cd.LineBasedTable(grayImage)
#    cv2.imwrite('colorimg.png', colorimg)
    return tablecoord,Rowcellcoord,multicell

def overlapCheck(tableRect, wordRect):
    tX, tY, tW, tH  = tableRect
    r1 = Rect(tX, tY, tW, tH)
    x, y, w, h = wordRect
    r2 = Rect(x, y, w, h)
    if r1.is_point_inside_rect(r2.center):
#        print(True)
        return True
    else:
#        print(False)
        return False                    

def labelDocRegions(ImageTables, imgpath):
#    imgpath = r'C:\ICDAR\Code\WhiteSpace\updated_space'
    head, tail = os.path.split(imgpath)
    filename =  os.path.splitext(tail)[0]
    imgpath =  os.path.join(head, filename + "_morph.png")
    wordsImg = cv2.imread(imgpath, 0)
    r, c = wordsImg.shape
    wordsRect = tb.morphCloseBlobs(wordsImg)
    labelledImg = np.zeros((r, c, 3), np.uint8)
    tableBlobs = []
#    for eachtable in ImageTables:
#        print(eachtable.tablecoordinates)
#        for words in wordsRect:
#            x, y, w, h = words
#            if overlapCheck(eachtable.tablecoordinates, words):
#                tableBlobs.append(words)
#            cv2.rectangle(labelledImg, (x, y), (w+x, h+y), (0, 0, 255), -1)
#
#    for rect in tableBlobs:
#        x, y, w, h = rect
#        cv2.rectangle(labelledImg, (x, y), (w+x, h+y), (70, 132, 48), -1)
    for eachtable in ImageTables:
         x, y, w, h = eachtable.tablecoordinates
         cv2.rectangle(labelledImg, (x, y), (w+x, h+y), (0, 0, 255), -1)
        
    savename = os.path.join(head, filename + 'labelledImg.png')
    cv2.imwrite(savename, labelledImg)
    return

def tableMain(srcImg, imgpath, debugOutPath):
    img = srcImg.copy()
    print("debugOutPath",debugOutPath)
    RoiList = []
    xmlFilePath = os.path.splitext(imgpath)[0] + '.txt'
    print(xmlFilePath)    
    if(os.path.exists(xmlFilePath)): #B2
        RoiList = getRoiFromTextFile(xmlFilePath)
        #RoiList = readCoordinatesFromXml(xmlFilePath)
    print(RoiList)
    denoiseSrc = noise_removal(img)
    
    ImageTables = []
    Tables = []
    if not RoiList :
        print("Table extraction: structured based.")
        Tables, ImageTables = tb.processtable(srcImg,debugOutPath, srcImg)
    else:
        print("Table extraction: cell based.")
        for i, tablecoordinate in enumerate(RoiList):
            x, y, r, b = stats2rect(tablecoordinate)
            r = x + r
            b = b + y
            roi = denoiseSrc[y:b, x:r]
            tables, ImageTable  = tr.processtable(x, y, roi)
            if ImageTable != None and len(tables)>0:
                ImageTables.append(ImageTable)
                Tables.append(tables)
            else:
                emptyTable = tr.Table()
                emptyTable.settablecoordinates([x, y, r-x, b-y ])
                ImageTables.append(emptyTable)
                Tables.append([[[x, y, r, b ]]])
    
    labelDocRegions(ImageTables, debugOutPath)
#    cv2.imwrite('labelledImg.png', labelledImg)
    return Tables, ImageTables

def makeDir(dirpath):
    dirpath = os.path.join(dirpath, "results")
    if  os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)
    return dirpath
 
def makeOutDir(dirpath,Filename):
    dirpath = os.path.join(dirpath, Filename)
    if  os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)
    return dirpath

def getRoiFromTextFile(txtFilePath):
    roiList = []
    try:
        rfile = open(txtFilePath) #,encoding="utf8")
        for i, word in enumerate(rfile):
           word = word.rstrip('\n')
           tableRoi = word.split(':')
           x, y, w, h = stats2rect(tableRoi)
           res = [int(x), int(y), int(w), int(h)]                      
           print('roi values', res)
           roiList.append(res)
        return roiList
    except:
        print("Something went wrong, while reading coordinates: {}".format(txtFilePath))
    finally:
        rfile.close()       
        return roiList       


if __name__ == "__main__":
    processStart = time.time()
    DIRECTORY = r'C:\Bhaskar\Programming\17-09-2019\WhitespaceIntegrated\test_results'
#    DIRECTORY = r'C:\POC\TableDetection\Python\Test'
    files = os.listdir(DIRECTORY)
    xtnList = [".jpg", ".JPG", ".png" , ".TIFF"]
    print("Total files:{}".format(len(files)))
    for i,eachfile in enumerate(files):
        fileXtn = os.path.splitext(eachfile)[1]
        imgname = os.path.splitext(eachfile)[0]
        if fileXtn in xtnList:
            print("Filename: {}".format(eachfile))
            imgpath = os.path.join(DIRECTORY, eachfile)
            srcImg = cv2.imread(imgpath)
            outFolder = os.path.join(DIRECTORY, imgname)
            if os.path.exists(outFolder):
                shutil.rmtree(outFolder)
            os.mkdir(outFolder)
            outPath = os.path.join(outFolder, eachfile)
            try:
                start = time.time()
                Tables, ImageTables = tableMain(srcImg.copy(), imgpath, outPath)
                print('Time taken', time. time()-start)
                tableDrawn = test(eachfile,Tables, ImageTables, srcImg.copy(), outFolder)
                tablesaveName = imgname + "_out.jpg"
                tabledrawnSavepath = os.path.join(outFolder, tablesaveName)
                cv2.imwrite(tabledrawnSavepath, tableDrawn)
            except:
                continue
            print("------table extraction completed--------\n")
        print("------ Process Completed ---------")
    
    timeLog = 'Workflow time taaken: ' + str(int(time. time()-processStart)/60) 
    print(Fore.RED + timeLog)
    fileLog = 'Total files: ' + str(len(files))
    print(Fore.GREEN + fileLog)
    print(Style.RESET_ALL)  