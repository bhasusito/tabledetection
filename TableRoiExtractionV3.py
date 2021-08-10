import cv2
import numpy as np
from geometry import Rect
from operator import itemgetter
import os, shutil, glob
#import Cell_Detection_table as cd

class Table():  
    tablecoordinates = []
    tablerows = []
    tablecols = []
    hrPoints = []
    vrPoints = []
    tableHeaders = []
    def __init__(self):
        self.tablerows = []
        self.tablecols = []
        self.tablecoordinates = []
        self.hrPoints = []
        self.vrPoints = []
        self.tableHeaders = []
    def settablecoordinates(self,eachtable):
        self.tablecoordinates.append(eachtable[0])
        self.tablecoordinates.append(eachtable[1])
        self.tablecoordinates.append(eachtable[2])
        self.tablecoordinates.append(eachtable[3])
        
    def settablerows(self, eachtablerow):
        self.tablerows.append(eachtablerow)
    
    def settablecols(self,eachtablecol):
        self.tablecols.append(eachtablecol)
        
    def sethrPoints(self,hrpoint):
        self.hrPoints.append(hrpoint)
        
    def setvrPoints(self,vrpoint):
        self.vrPoints.append(vrpoint)

    def settableHeaders(self,firstRowData):
        for eachrow in firstRowData:
            self.tableHeaders.append(eachrow)
            
            
def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

def rect2Rect(rect):
    return Rect(rect[0], rect[1], rect[2], rect[3])

def binarize(inpImg):
    gry = cv2.cvtColor(inpImg, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(~gry, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return binary
def noise_removal(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img,80,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    kernel = np.ones((7,7), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    _, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
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



def stats2rect(ca):
    x = ca[0]
    y = ca[1]
    w = ca[2]
    h = ca[3]
    return x, y, w, h

def img2Rect(binary):
    rects = []
    _, contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h>13:
            rects.append([x, y, w, h])
    return rects

#def img2words(binary):
#    kSize = 20
#    kernel = np.ones((1, kSize), np.uint8)
#    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
#    _, contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#    words = np.zeros(binary.shape, np.uint8)
#    for contour in contours:
#        x, y, w, h = cv2.boundingRect(contour)
#        if h < 50 :
##            cv2.fillPoly(words, pts =[contour], color=255)
#            cv2.rectangle(words, (x, y), (x+w, y+h), 255, cv2.FILLED)
#    return words
    
def img2words(binary):
    kSize = 20
    kernel = np.ones((1, kSize), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    _, contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    words = np.zeros(binary.shape, np.uint8)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < 50:
#            cv2.fillPoly(words, pts =[contour], color=255)
            cv2.rectangle(words, (x, y), (x+w, y+h), 255, cv2.FILLED)
    return words


def funf(ca, cb):    
    if ca[1] <= (cb[1] + cb[3]) and (ca[1] + ca[3]) >= cb[1]:        
        return True
    return False

def lineblobfillfun(thresh, ca, cb):
    if funf(ca,cb):
        x = ca[0] + ca[2]
        y = ca[1]
        w = cb[0] - x
        h = ca[3]
        cv2.rectangle(thresh, (x, y), (x+w, y+h), 255, cv2.FILLED)
    return thresh

def borderNoiseRemoval(binary, border):
    h, w = binary.shape
    cv2.rectangle(binary, (0,0), (w, border), 0, cv2.FILLED)
    cv2.rectangle(binary, (0, 0), (border, h), 0, cv2.FILLED)
    cv2.rectangle(binary, (0, h-border), (w, h), 0, cv2.FILLED)
    cv2.rectangle(binary, (w-border, 0), (w, h), 0, cv2.FILLED)           
    return binary

def findLines(wordsImg, wordsRects):
    for i, ca in enumerate(wordsRects):
        r1 = rect2Rect(ca)
        for j, cb in enumerate(wordsRects):
            if i != j:
                r2 = rect2Rect(cb)
                res = r2.overlaps_on_y_axis_with(r1)
                if res:
                    wordsImg = lineblobfillfun(wordsImg, ca, cb)
    return wordsImg

def setTableRows(linesRect, srcImg):
#    cv2.imwrite('srcImg.png', srcImg)
#    print('lines...', linesRect)
#    print('total line', len(linesRect))

    if len(linesRect) == 0:
        return None
        
    doc_table = Table()
    if len(linesRect[0]) > 1:
        tx = min(linesRect, key=itemgetter(0))[0]
        ty = linesRect[0][1]
        rList = [line[0]+line[2] for line in linesRect]
        tr = max(rList)
        tb = linesRect[-1][1] + linesRect[-1][3]
        doc_table.settablecoordinates([tx, ty, tr-tx, tb-ty])
        linesRect.sort(key=lambda x:x[1])
#        print('lines...', len(linesRect))
        for row in linesRect:
            rx, ry, rw, rh = stats2rect(row)
            doc_table.settablerows([rx, ry, rw, rh ])
#        print('table rows...', len(doc_table.tablerows))
    elif len(linesRect[0]) == 1:
        lx, ly, lw, lh = stats2rect(linesRect[0])
        doc_table.settablecoordinates([lx, ly, lw, lh])
   
    return doc_table

def blobs_in_Box(hostRect, allblobs):
    count = 0
    blobs = []
    for x, y, w, h in allblobs:
        r2 = Rect(x, y, w, h)
        res = hostRect.overlaps_with(r2)
        if res:
            count += 1
            blobs.append([x, y, w, h])
    return blobs, count

def horizontal_gap_bw_Boxes(blobs):
    blobCount = len(blobs)
    gapRect = []
    for i in range(1, blobCount):
        x, y, w, h = blobs[i-1]
        x1, y1, w1, h1 = blobs[i]
        gx = x+w+1
        gy = y
        gw = (x1-1) - gx
        gh = h
        if gw > 5:
            gapRect.append([gx, gy, gw, gh])
    return gapRect

    
def wordGapsInBox(doc_table, wordsRects):
    wordGap = []
    for j, row in enumerate(doc_table.tablerows):
        r1 = rect2Rect(row)
        row_blobs, count = blobs_in_Box(r1, wordsRects)
        if count > 1:
            row_blobs.sort(key=lambda x: x)
            gapRect = horizontal_gap_bw_Boxes(row_blobs)
            if len(gapRect) > 0:
                wordGap.append([gapRect, count, j]) ## insert row_blobs
    return wordGap

def createTable(x, y, rows, cols, wordsImg, wordRects, srcImg):
    cells = srcImg.copy()
    wordRects.sort(key=lambda x:x)
    tx = wordRects[0][0]
    wordRects.sort(key=lambda x:x[2], reverse = True)
    tr = wordRects[0][2]
    wordRects.sort(key=lambda x:x[1])
    ty = wordRects[0][1]
    wordRects.sort(key=lambda x:x[1], reverse = True)
    tb = wordRects[0][3]
    cv2.rectangle(srcImg, (tx, ty), (tr, tb), (255, 0, 0), 3)
    
    doc_table = Table()
    doc_table.settablecoordinates([x+tx, y+ty, (tr-tx), (tb-ty)]) 
#    print(rows)
    rows.sort(key=lambda x:x[1])
    cols.sort(key=lambda x:x[0])
    if len(rows) < 2 or len(cols) < 2:
        return None, [[]], srcImg
#    print(rows)
#    print(cols)
    doc_table.setvrPoints(y+ty)
    for i in range(0, len(rows)-1):
        bottom_TELa = rows[i][3]
        top_TELa1 = rows[i+1][1]
        gap = 0
        gap = round((top_TELa1 - bottom_TELa)/2)
        hr = gap + rows[i][3] 
#        cv2.rectangle(srcImg, (tx, hr), (tr, hr), (255,191,0),2)
        doc_table.settablerows([x+tx, y+hr, x+tr, y+hr])                
        doc_table.setvrPoints(y+hr)           
#        doc_table.setvrPoints(tb)
#        doc_table.sethrPoints(x+tx)
    doc_table.setvrPoints(y+tb)
    
    doc_table.sethrPoints(x+tx)
    for i in range(0, len(cols)-1):
        right_TELa = cols[i][2]
        left_TELa1 = cols[i+1][0] 
        gap = 0                               
        gap = round((left_TELa1 - right_TELa)/2)
        vx = cols[i][2] + gap
#        cv2.rectangle(srcImg, (vx, ty), (vx, tb), (255,0,0),2)
        doc_table.settablecols([x+vx, y+ty, x+vx, y+tb])                
        doc_table.sethrPoints(x+vx)
#        doc_table.sethrPoints(x+tr)
    doc_table.sethrPoints(x+tr)
     
#    cv2.imwrite('out.png', srcImg)
    vrLen = len(doc_table.vrPoints)
    hrLen = len(doc_table.hrPoints)
    
    rowCells = []
    
    for i,vr in enumerate(doc_table.vrPoints):
        if i != vrLen-1:
            cols = []                
            for j,hr in enumerate(doc_table.hrPoints):
                    if i < vrLen-1 and j < hrLen-1:
                            cell = []
                            cell = [doc_table.hrPoints[j], doc_table.vrPoints[i], doc_table.hrPoints[j+1], doc_table.vrPoints[i+1]]        				
                            cols.append(cell)
                            cv2.rectangle(cells, (cell[0], cell[1]), (cell[2], cell[3]), (0,0,255), 2)
            rowCells.append(cols)
#    cv2.imwrite('cells.png', cells)
    return doc_table, rowCells, srcImg
def createCells(x, y, doc_table, wordsRect, srcImg):
    cells = srcImg.copy()
    cellsout = srcImg.copy()
    rowsImg = srcImg.copy()
    
    tx, ty, tw, th = stats2rect(doc_table.tablecoordinates)
#    print("table coordinates:",[tx, ty, tw, th])
    cv2.rectangle(rowsImg, (tx, ty), (tw +tx, th +ty), (0,0,255), 2)
    doc_table1 = Table()
    doc_table1.settablecoordinates([x+tx, y+ty, tw, th])
    tx, ty, tw, th = stats2rect(doc_table1.tablecoordinates)
    doc_table1.sethrPoints(tx)
    
    # Find cols split
    rowcols = []
    for i, row in enumerate(doc_table.tablerows):
        rowcol = []
        cell = []
        r1 = rect2Rect(row)
        wordblobs, count = blobs_in_Box(r1, wordsRect)
        if count == 1:
            cell.append(wordblobs[0])
            doc_table1.sethrPoints(x + wordblobs[0][0] + wordblobs[0][2])
            rowcol.append(tx)
            #rowcol.append(wordblobs[0][0] + wordblobs[0][2])
            rowcol.append(tx+tw)
        else:
            wordblobs.sort(key=lambda x:x)
            for j in range(1, count):
                w1 = wordblobs[j-1]
                w2 = wordblobs[j]
                midpt = round((w2[0] - (w1[0]+w1[2]))/2)
                midpt += w1[0]+w1[2]
                vx = x + midpt
                doc_table1.sethrPoints(vx)
                rowcol.append(tx)
                rowcol.append(vx)
                rowcol.append(tx+tw)
                wy = x + wordblobs[j][1]
                wh = x + wordblobs[j][3] + wy
                cv2.rectangle(rowsImg, (vx, wy), (vx, wh), (255,0,0),2)
                doc_table1.settablecols([vx, ty, vx, (wh)])
        rowcols.append(rowcol)
    doc_table1.sethrPoints(tx+tw)
    
    # Find row split
    for i, cell in enumerate(doc_table.tablerows):
        cv2.rectangle(cells, (cell[0], cell[1]), (cell[0]+cell[2], cell[1]+cell[3]), (0,0,255), 2)
#        cv2.imwrite('rows1.png', cells)
        
    doc_table.tablerows.sort(key=lambda x:x[1])
    
    doc_table1.setvrPoints(ty)
    for i in range(0, len(doc_table.tablerows)-1):
        bottom_TELa = doc_table.tablerows[i][1]+doc_table.tablerows[i][3]
        top_TELa1 = doc_table.tablerows[i+1][1]
        gap = 0
        gap = round((top_TELa1 - bottom_TELa)/2)
        hr =y+ gap + doc_table.tablerows[i][1] + doc_table.tablerows[i][3] 
#        cv2.rectangle(rowsImg, (tx, hr), (tx+tw, hr), (255,191,0),2)
        #cv2.rectangle(rowsImg, (x+tx, ty+hr), (x+tx+tw, ty+hr), (0,0,255),2)
        #print("before: ",doc_table.tablerows[i])
        #print("after: ",[tx, hr, tx+tw, hr])
        doc_table1.settablerows([tx, hr, tx+tw, hr])  
#        cv2.imwrite('rows.png', rowsImg)              
        doc_table1.setvrPoints(hr)
    doc_table1.setvrPoints(ty+th)
    
    
#    print('vrpoints=', doc_table1.vrPoints)
#    print('hrpoints=', doc_table1.hrPoints)
#    print("---- after row formation -----")
#    print('rows', doc_table1.tablerows)
#    print('cols', doc_table1.tablecols)
#    for i, cell in enumerate(doc_table1.tablerows):
#        cv2.rectangle(afterrowsImg, (cell[0], cell[1]), (cell[2], cell[3]), (0,0,255), 2)
#        cv2.imwrite('rows2.png', afterrowsImg)
#    print("---- after row formation end -----")
    
   
    vrLen = len(doc_table1.vrPoints)
    hrLen = len(doc_table1.hrPoints)           
    rowCells = []
    for i,vr in enumerate(doc_table1.vrPoints):
        if i != vrLen-1:
            cols = []
            rr = rowcols[i].copy()
            for j,rc in enumerate(rr):
                if j != len(rr)-1:
                    cell = []
                    cell = [rc, vr, rr[j+1], doc_table1.vrPoints[i+1]]
#                    print('cell blobs..', cell)
                    cv2.rectangle(cellsout, (cell[0], cell[1]), (cell[2], cell[3]), (0,0,255), 2)
                    cols.append(cell)
            rowCells.append(cols)
#            cv2.imwrite('cellsout.png', cellsout)
            

    return rowCells, doc_table1
            
            

def processtable(x, y, roiImg):
   
#    outimg = roiImg.copy()
#    deNImg = noise_removal(roiImg)
    binary = binarize(roiImg)
    binary = borderNoiseRemoval(binary, 10)
    wordsImg = img2words(binary)

    wordsRects = img2Rect(wordsImg)
    linesImg = findLines(wordsImg, wordsRects)
    linesRect = img2Rect(linesImg)
    linesRect.sort(key=lambda x: x[1])
    doc_table = setTableRows( linesRect, roiImg.copy())
    if not doc_table == None:
        tableCells,doc_table1  = createCells(x, y, doc_table, wordsRects, roiImg.copy())
        return tableCells, doc_table1
    else:
        return [[]], None

def makeDir(dirpath):
    dirpath = os.path.join(dirpath, "results")
    if  os.path.exists(dirpath): #and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)
    return dirpath
        


def test(tables, ImageTables, rawImg):
    for i, eachtable in enumerate(ImageTables):
        tX = eachtable.tablecoordinates[0]; tY = eachtable.tablecoordinates[1]; 
        tW = eachtable.tablecoordinates[2]; tH = eachtable.tablecoordinates[3];
        cv2.rectangle(rawImg, (tX, tY), (tW + tX, tH+tY), (0,0,255), 3)
 
    for i, rowCells in enumerate(tables):
        cnt = 0
        for cells in rowCells :
            for cell in cells :
                cnt +=1
                cv2.rectangle(rawImg, (cell[0],cell[1]), (cell[2],cell[3]), (0, 255, 0), 2)
#    print('no of cells....', cnt)
    return rawImg

def linebasedTableDetection(srcImg):
    grayImage = cv2.cvtColor(srcImg,cv2.COLOR_BGR2GRAY)
    tablecoord, lineOutImg = cd.LineBasedTable(grayImage) #cd1.fillimage(grayImage)
    return tablecoord

if __name__ == "__main__":
#    for (root,dirs,files) in os.walk(r'C:\Confident\Apr8\dataset'): 
    DIRECTORY = r'C:\Confident\tableStr\Apr11\images'
#    DIRECTORY = root
    outpath = makeDir(DIRECTORY)
    images = glob.glob(os.path.join(DIRECTORY , "*.png"))
    for i, imgpath in enumerate(images):
        srcImg = cv2.imread(imgpath)
        img = srcImg.copy()
        tablecoordinates = linebasedTableDetection(srcImg.copy())
        imgname = os.path.split(imgpath)[1]
        print("------table extraction started--------")
        print('filename', imgname)
        if(len(tablecoordinates) > 1):
#            print("Table extraction: Line based.")
#            print('roi coordinates', tablecoordinates)
            denoiseSrc = noise_removal(img)
            ImageTables = []
            Tables = []
            for j, tablecoordinate in enumerate(tablecoordinates):
                x, y, r, b = stats2rect(tablecoordinate)
                roi = denoiseSrc[y:b, x:r]
#                cv2.imwrite(str(j)+'.png', roi)
#                ImageTable, tables, outImg  = tr.processtable(x, y, roi, [x, y, r, b])
                tables, ImageTable = processtable(x, y, roi)
                if ImageTable != None and len(tables)>0:
                    ImageTables.append(ImageTable)
                    Tables.append(tables)
                else:
                    emptyTable = Table()
                    emptyTable.settablecoordinates([x, y, r-x, b-y ])
                    ImageTables.append(emptyTable)
                    Tables.append([[[x, y, r, b ]]])
            tableDrawn = test(Tables, ImageTables, srcImg)
            cv2.imwrite(outpath+'\\'+imgname[:-4]+ '_'+str(j)+'.png', tableDrawn)
        
        print("------table extraction completed--------\n")
     
    print("------ Process Completed ---------")