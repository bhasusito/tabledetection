# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:40:42 2018

@author: menugulab
"""

import pytesseract
import cv2
from geometry import Rect
from geometry import Point


#%%

def cell2Rect(cell):
    rowx =  cell[0] ; rowy = cell[1];
    rowW =  cell[2] - cell[0]
    rowH =  cell[3] - cell[1] 
    return rowx, rowy ,rowW ,rowH

def box2Rect(box,h):
    vx =int(box[1]); vy = h - int(box[4]); 
    vw = int(box[3]) - int(box[1])
    vh = int(box[4])  - int(box[2])
    return vx, vy, vw, vh

def blob2Rect(box):
    vx =int(box[1]); vy = int(box[2]); 
    vw = int(box[3]); vh = int(box[4])
    return vx, vy, vw, vh

def isIntersect(r1,r2,pointr2):
    resIntersect =  r1.overlaps_with(r2)
    respoint = r1.is_point_inside_rect(pointr2)
    return resIntersect, respoint


def tableOCR(img,tables):
    config = ('-l eng --oem 1 --psm 6')
    text1 = pytesseract.image_to_boxes(img,config=config)
#    text1 = pytesseract.image_to_data(img,config=config)
#    text2 = pytesseract.image_to_data(img,config=config)
#    print("---------")
    #print(text1)
    
#    print("---------")
    h, w = img.shape[:2]
    tableOCR = []
    #print("------ Table OCR Start ------")
    for i, rowCells in enumerate(tables):
        #print("Table:{0}".format(i))
        tabletext = []       
        for cells in rowCells :
            rowWords = []
            for cell in cells :
                rowx, rowy ,rowW ,rowH = cell2Rect(cell) 
                if rowW > 0 and rowH > 0 :
                    #print(rowx, rowy ,rowW ,rowH)
                    r1 = Rect(rowx, rowy ,rowW ,rowH) 
                    img = cv2.rectangle(img, (cell[0],cell[1]), (cell[2],cell[3]), (255, 0, 0), 2)
                    cellvalue = ""        
                    for box in text1.splitlines():
                        box = box.split(' ')            
                        if len(box) > 1:
                            vx,vy,vw,vh = box2Rect(box,h)
                            if vw > 1 and vh > 1 :
                                r2 = Rect(vx,vy,vw,vh)
                                midpointx = int(vw/2); midpointy = int(vh/2)
                                pointr2 = Point(vx  + midpointx , vy + midpointy)
                                resIntersect, respoint = isIntersect(r1,r2,pointr2)
                                if resIntersect and respoint:            
                                    img = cv2.rectangle(img, (int(box[1]), h - int(box[2])), (int(box[3]), h - int(box[4])), (0, 0, 255), 2)
                                    cellvalue = cellvalue + box[0]
                                    #print(box)
                    #print("------------")
                    rowWords.append(cellvalue)
                else:
                    img = cv2.rectangle(img, (cell[0],cell[1]), (cell[2],cell[3]), (0, 255, 0), 3)
                    rowWords.append("")
            tabletext.append(rowWords)
        #print("rows, cols: {0} x {1}".format(len(tabletext),len(tabletext[0])))
        #print("cols:{0}".format(len(tabletext[0])))
        tableOCR.append(tabletext)
    #print("------ Table OCR End ------")
    #print(tableOCR)
    print("Table ocr processing...")
    save_stats = "C:\\Geico\\TableExtraction\\Mar4\\Test\\box.jpg"
#    cv2.imwrite(save_stats, img)
    return tableOCR

#%%
    
def table_to_ocr(img,tables):
    h, w = img.shape[:2]
    tableOCR = []
    
    config = ('-l eng --oem 1 --psm 6')
    #text1 = pytesseract.image_to_boxes(img,config=config)
    text1 = pytesseract.image_to_data(img,config=config)
    
    #print(text1)
    text1 = text1.split('\n')
    ocrWords = []
#    print(text1)
    for data in text1:
        d = data.split('\t')
        if len(d) < 12 : 
            word = " "
        else:
            word = d[11]
        ocrWords.append([d[6], d[7], d[8], d[9], word])
#        print(d[6], d[7], d[8], d[9], d[11])
#        print(len(d))  
        
    if ocrWords[0][0] == 'left':
        del ocrWords[0]
            
    #print("------ Table OCR Start ------")
    for i, rowCells in enumerate(tables):
        tabletext = []       
        for cells in rowCells :
            rowWords = []
            for cell in cells :
                rowx, rowy ,rowW ,rowH = cell2Rect(cell) 
                if rowW > 0 and rowH > 0 :
                    r1 = Rect(rowx, rowy ,rowW ,rowH) 
                    img = cv2.rectangle(img, (cell[0],cell[1]), (cell[2],cell[3]), (255, 0, 0), 2)
                    cellvalue = ""        
                    for box in ocrWords:
                        if len(box) > 1:
                            vx = int(box[0]); vy = int(box[1]); vw  = int(box[2]); vh = int(box[3]);
                            if vw > 1 and vh > 1 and vw < w and vh < h:
                                r2 = Rect(vx,vy,vw,vh)
                                midpointx = int(vw/2); midpointy = int(vh/2)
                                pointr2 = Point(vx  + midpointx , vy + midpointy)
                                resIntersect, respoint = isIntersect(r1,r2,pointr2)
                                if resIntersect and respoint:            
                                    img = cv2.rectangle(img, (int(box[0]), int(box[1])), ((int(box[0]) + int(box[2])), (int(box[1]) + int(box[3]))), (0, 0, 255), 2)
                                    cellvalue = cellvalue + box[4] + " "
                                    #print(box)
                    rowWords.append(cellvalue)
                    #print(cellvalue)
                else:
                    img = cv2.rectangle(img, (cell[0],cell[1]), (cell[2],cell[3]), (0, 255, 0), 3)
                    rowWords.append("")
            tabletext.append(rowWords)
        tableOCR.append(tabletext)
    print("Table ocr processing...")
    #save_stats = "C:\\Geico\\TableExtraction\\Mar4\\Test\\box.jpg"
    #cv2.imwrite(save_stats, img)
    return tableOCR

#%%
    
def getTableHeaders(tables,txtFilePath):
    firstrowData = []
    blobs_rect = read_file_coordinates(txtFilePath)
    
    firstrowNum = 0;
    #tableOCR = []
    for i, rowCells in enumerate(tables):
        #tableheader = []       
        for j, cells in enumerate(rowCells):
            if j == firstrowNum:
                rowWords = []
                for cell in cells :
                    rowx, rowy ,rowW ,rowH = cell2Rect(cell) 
                    if rowW > 0 and rowH > 0 :
                        r1 = Rect(rowx, rowy ,rowW ,rowH) 
                        cellvalue = ""        
                        for box in blobs_rect:
                            if len(box) > 1:
                                vx,vy,vw,vh = blob2Rect(box)
                                if vw > 1 and vh > 1 :
                                    r2 = Rect(vx,vy,vw,vh)
                                    midpointx = int(vw/2); midpointy = int(vh/2)
                                    pointr2 = Point(vx  + midpointx , vy + midpointy)
                                    resIntersect, respoint = isIntersect(r1,r2,pointr2)
                                    if resIntersect and respoint:            
                                        cellvalue = cellvalue + box[0] # + " "
                        rowWords.append(cellvalue)
                    else:
                        rowWords.append("")
                #tableheader.append(rowWords)
                firstrowData.append(rowWords)
                break;
        #firstrowData.append(tableheader)
    return firstrowData
#%%
def googleOCR(img,tables,txtFilePath):
    
    blobs_rect = read_file_coordinates(txtFilePath)
    #print(blobs_rect)
    
#    print("---------")
    h, w = img.shape[:2]
    tableOCR = []
    #print("------ Table OCR Start ------")
    for i, rowCells in enumerate(tables):
        #print("Table:{0}".format(i))
        tabletext = []       
        for cells in rowCells :
            rowWords = []
            for cell in cells :
                rowx, rowy ,rowW ,rowH = cell2Rect(cell) 
                if rowW > 0 and rowH > 0 :
                    #print(rowx, rowy ,rowW ,rowH)
                    r1 = Rect(rowx, rowy ,rowW ,rowH) 
                    img = cv2.rectangle(img, (cell[0],cell[1]), (cell[2],cell[3]), (255, 0, 0), 2)
                    cellvalue = ""        
                    for box in blobs_rect:
                        #box = box.split(' ')            
                        if len(box) > 1:
                            vx,vy,vw,vh = blob2Rect(box)
                            #vx,vy,vw,vh = box2Rect(box,h)
                            if vw > 1 and vh > 1 :
                                r2 = Rect(vx,vy,vw,vh)
                                midpointx = int(vw/2); midpointy = int(vh/2)
                                pointr2 = Point(vx  + midpointx , vy + midpointy)
                                resIntersect, respoint = isIntersect(r1,r2,pointr2)
                                if resIntersect and respoint:            
                                    img = cv2.rectangle(img, (int(box[1]), h - int(box[2])), (int(box[3]), h - int(box[4])), (0, 0, 255), 2)
                                    cellvalue = cellvalue + box[0]  + " "
                                    #print(cellvalue)
                    #print("------------")
                    rowWords.append(cellvalue)
                else:
                    img = cv2.rectangle(img, (cell[0],cell[1]), (cell[2],cell[3]), (0, 255, 0), 3)
                    rowWords.append("")
            tabletext.append(rowWords)
        #print("rows, cols: {0} x {1}".format(len(tabletext),len(tabletext[0])))
        #print("cols:{0}".format(len(tabletext[0])))
        tableOCR.append(tabletext)
    #print("------ Table OCR End ------")
    #print(tableOCR)
    print("Table ocr processing...")
    save_stats = "C:\\Bhaskar_backup\\Bhaskar\\POC\\LayoutSeparation\\Gaiko\\Tabledetection\\OCRtest\\box.jpg"
    #   cv2.imwrite(save_stats, img)
    return tableOCR


def read_file_coordinates(txt_file_path):
        ''' Read coordinates file '''
        filepath = txt_file_path
        try:
            rfile = open(filepath) #,encoding="utf8")
            blobs_rect = []
            for i, word in enumerate(rfile):
                 word = word.rstrip()
                 box = word.split(' ')
                 #print(box)
                 blobs_rect.append(box)
            #print("Total blob rectangles: {}".format(len(blobs_rect)))
            rfile.close()
            return blobs_rect
        except IOError:
            print("Something went wrong, while reading coordinates file: {}".format(filepath))
        finally:
            rfile.close()
            
            

#%%
def main():
    filepath = "C:\\Bhaskar_backup\\Bhaskar\\POC\\LayoutSeparation\\Gaiko\\Tabledetection\\WriteToExcel\\CLM%200522403630101041-000013.png"
    #tesseract_exe = r'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'
    #pytesseract.pytesseract.tesseract_cmd = tesseract_exe
    n = filepath.rfind('\\')
    print(filepath[n+1:])
    print(filepath[:-4])
    img = cv2.imread(filepath)
    h, w = img.shape[:2]
    print(h, w)
    config = ('-l eng --oem 1 --psm 6') 
    text = pytesseract.image_to_string(img,config=config)    
    cells = [[477, 1157, 695, 1203], [695, 1157, 955, 1203], [955, 1157, 1238, 1203], [1238, 1157, 1569, 1203], [1569, 1157, 1812, 1203], [1812, 1157, 2093, 1203]]
    text1 = pytesseract.image_to_boxes(img,config=config)
    text2 = pytesseract.image_to_data(img,config=config)
    print("---------")
    print(text2)
    print("---------")
    rowWords = []
    for cell in cells :     
        rowx =  cell[0] ; rowy =cell[1];
        rowW =  cell[2] - cell[0]
        rowH =  cell[3] - cell[1]    
        r1 = Rect(rowx, rowy ,rowW ,rowH) 
        img = cv2.rectangle(img, (cell[0],cell[1]), (cell[2],cell[3]), (255, 0, 0), 2)
        cellvalue = ""        
        for box in text1.splitlines():
            box = box.split(' ')
            #img = cv2.rectangle(img, (int(box[1]), h - int(box[2])), (int(box[3]), h - int(box[4])), (0, 255, 0), 2)            
            if len(box) > 1:                        
                vx =int(box[1]); vy = h - int(box[4]); 
                vw = int(box[3]) - int(box[1])
                vh = int(box[4])  - int(box[2])
                if vw > 1 and vh > 1 :
                    #print(vx,vy,vw,vh)
                    r2 = Rect(vx,vy,vw,vh)
                    midpointx = int(vw/2); midpointy = int(vh/2)
                    pointr2 = Point(vx  + midpointx , vy + midpointy) 
                    res =  r1.overlaps_with(r2)                 
                    if res:
                        respoint = r1.is_point_inside_rect(pointr2)
                        if respoint:                        
                            img = cv2.rectangle(img, (int(box[1]), h - int(box[2])), (int(box[3]), h - int(box[4])), (0, 0, 255), 2)
                            cellvalue = cellvalue + box[0]
#                else:
#                    print(box)
    
        #print("Cellvalue :{0}".format(cellvalue))
        rowWords.append(cellvalue)
    print(rowWords)
    save_stats = "C:\\Bhaskar_backup\\Bhaskar\\POC\\LayoutSeparation\\Gaiko\\Tabledetection\\WriteToExcel\\box.jpg"
    cv2.imwrite(save_stats, img)


#    for i,box in enumerate(text1):
#        print(box)
#        if len(box) > 4:
#            
#            print(len(box))
#            vx = box[1]; vy = box[2];
#            vw = box[3]- box[1]
#            vh = box[4]  - box[2] 
#            r2 = Rect(vx,vy,vw,vh)
#            res =  r1.overlaps_with(r2)                 
#            if res:
#                print(box)
        
    #print(text1)
    #print(pytesseract.image_to_string(img,config=config))


#%%
   
def GetTables():
    tables = []
    
    rowcells = []
    table1r1 = [[282,18,471,150],[960,79,1582,141]]
    rowcells.append(table1r1)
    tables.append(rowcells)
    
#    rowcells = []
#    table1r1 = [[477, 1119, 695, 1157], [695, 1119, 955, 1157], [955, 1119, 1238, 1157], [1238, 1119, 1569, 1157], [1569, 1119, 1812, 1157], [1812, 1119, 2093, 1157]]
#    rowcells.append(table1r1)
#    table1r2 = [[477, 1157, 695, 1203], [695, 1157, 955, 1203], [955, 1157, 1238, 1203], [1238, 1157, 1569, 1203], [1569, 1157, 1812, 1203], [1812, 1157, 2093, 1203]]
#    rowcells.append(table1r2)
#    table1r3 = [[477, 1203, 695, 1243], [695, 1203, 955, 1243], [955, 1203, 1238, 1243], [1238, 1203, 1569, 1243], [1569, 1203, 1812, 1243], [1812, 1203, 2093, 1243]]
#    rowcells.append(table1r3)
#    tables.append(rowcells)
#
#    rowcells = []
#    table2r1 = [[1159, 801, 1260, 842], [1260, 801, 1607, 842], [1607, 801, 1918, 842], [1918, 801, 2257, 842]]
#    rowcells.append(table2r1)
#    table2r2 = [[1159, 842, 1260, 880], [1260, 842, 1607, 880], [1607, 842, 1918, 880], [1918, 842, 2257, 880]]
#    rowcells.append(table2r2)
    #tables.append(rowcells)
    
    return tables
#%%
if __name__=="__main__":
    print("Hello world")
    print(pytesseract.get_tesseract_version())
    selfversion = str(pytesseract.get_tesseract_version())
    selfversion = str(selfversion[1:4])
    print(selfversion)
    if selfversion < '3.05':
        print("image data version ")
        
    
#    filepath = "C:\\Bhaskar_backup\\Bhaskar\\POC\\LayoutSeparation\\Gaiko\\Tabledetection\\WriteToExcel\\CLM%200522403630101041-000013.png"    
    filepath = "C:\\Geico\\TableExtraction\\Mar4\\Test\\ocrIssue.png"
    img = cv2.imread(filepath)
    
    tables = GetTables()
    table_to_ocr(img,tables)
    
#    config = ('-l eng --oem 1 --psm 6')
#    ocr = pytesseract.image_to_string(img,config=config)
#    print(ocr)
#    
#    print("---------")
#    text1 = pytesseract.image_to_data(img,config=config)
#    print(text1)
#    print("----")
#    
#    text1 = text1.split('\n')
#    for data in text1:
#        d = data.split('\t')
#        print(d[6],d[7],d[8],d[9],d[11])
#        #print(d)
#        print(len(d))
#        print("----")
#    
#
#    linedata = '5	1	1	1	1	1	73	32	149	23	93	2/15/2017'        
#    e = linedata.split('	')
#    print(e[6],e[7],e[8],e[9],e[11])
#    
    
#    text1 = pytesseract.image_to_boxes(img,config=config)
#    #print(text1)
#    print("-------------")
    #tables = GetTables()
    #txtFilePath = "C:\\Bhaskar_backup\\Bhaskar\\POC\\LayoutSeparation\\Gaiko\\Tabledetection\\Feb10th\\OCR\\pagedpi_300-176.png.txt"
    #results = googleOCR(img,tables,txtFilePath)
    #print(results)
    #tableOCR = tableOCR(img,tables)  
   #main()

