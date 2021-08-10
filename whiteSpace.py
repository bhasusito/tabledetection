# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import shutil
from geometry import Rect
from LineDetection import Line_Removal_fn
from operator import itemgetter
import matplotlib.pyplot as plt

isDebug = False

def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

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

def binarize(inpImg):
    gry = cv2.cvtColor(inpImg, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(~gry, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return binary

def distancefunD(ca, cb):
    d = cb[0] - (ca[0]+ca[2])
    return d

def stats2rect(ca):
    x = ca[0]
    y = ca[1]
    w = ca[2]
    h = ca[3]
    return x, y, w, h
def second_Hump(hist):
    length = len(hist)
    flag = 0
    first_Hump = max(hist)
    if hist[0] == first_Hump:
        flag = 1
    peak = []
    for i in range(1,length-1):
        if (hist[i] > hist[i-1] and hist[i] > hist[i+1]):
            peak.append(hist[i])
#    print(len(peak))
    if(len(peak) > 2):
        if flag:
            second_hump = max(peak)
        else:
            peak.sort()
            second_hump = peak[len(peak)-2]  
        new_hist = [float(i) for i in hist]
        hump_pos = new_hist.index(second_hump)
        try:
            while new_hist[hump_pos] > 0:
                hump_pos += 1
            hump_pos -= 1
        except:
            pass
        return second_hump, hump_pos
    else:
        return 10,0
def connectedComp_Analysis(linesRemovedImg):
    '''connected component analysis'''
    connectivity = 8  # You need to choose 4 or 8 for connectivity type
    stats = cv2.connectedComponentsWithStats(~linesRemovedImg, connectivity, cv2.CV_32S)[2]
    cc_stats = list(stats)
    return cc_stats

def getWordsGap1(cc_stats):
     statsLines = []
     for i, ca in enumerate(cc_stats): 
        blobsRow = [] 
        x, y, w, h = stats2rect(ca)
        if w>2 and h< 80:
            blobsRow.append(ca)
            for j, cb in enumerate(cc_stats):
                if j != i:
                     funfres = funf(ca, cb)
                     if funfres:
                         blobsRow.append(cb)
            if blobsRow:
                statsLines.append(blobsRow)
         
     return statsLines

def getWordsGap(cc_stats):
    try:
        compB = []
        compA = []
        alldcomp = []
        hrsize = 0
        for i, ca in enumerate(cc_stats):        
            x, y, w, h = stats2rect(ca) 
            if w > 2 and h < 80:#and h>15:
                lcx = []       
    #            cv2.rectangle(src, (x, y), (x+w, y+h), (255,0,0), 2)
                for j, cb in enumerate(cc_stats):
                        if i != j:
                            funfres = funf(ca, cb)
                            if funfres:
                                d = distancefunD(ca, cb)
                                if funfres and cb[0] > (ca[0]+ca[2]):
                                     lcx.append([d, j])                                                       
                if len(lcx) > 1:
                    idx = min(lcx, key=itemgetter(0))                
                    alldcomp.append(idx[0])
                    compB.append(cc_stats[idx[1]])
                    compA.append(ca)
#        print(alldcomp)
        alldcomp = list(filter(lambda x: x < 60, alldcomp))   
        num_bins = max(alldcomp)
        h1, bins, patches = plt.hist(alldcomp, bins = num_bins, facecolor='green', alpha=0.5)
        secondHump, hrsize= second_Hump(h1)
        hrsize = bins[int(hrsize)]
    except:
        return 12
    return int(hrsize)

def img2words(binary):
    
    cc_stats = connectedComp_Analysis(binary)
    #######################
#    statsLines = getWordsGap1(cc_stats)
#    testImg = np.zeros((binary.shape), np.uint8)
#    for line in statsLines[0]:
#        for ca in line:
#            x,y,w,h,r = stats2rect(ca)
#            cv2.rectangle(testImg, (x, y), (x+w, y+h), 255, cv2.FILLED)
    #############################
    kSize = getWordsGap(cc_stats)
    print('white space hrsize:', kSize)
#    kSize = 15
    kernel = np.ones((1, kSize), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    saveImage(outpath, binary, '_whiteSpace_drawn.jpg')
    contours,hiwspace = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    words = np.zeros(binary.shape, np.uint8)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
#        if h < 50 :
#            cv2.fillPoly(words, pts =[contour], color=255)
        cv2.rectangle(words, (x, y), (x+w, y+h), 255, -1)
    return words, binary

def img2Rect(binary):
    rects = []
    contours, hieRect = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rects.append([x, y, w, h])
    return rects

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

def formLines(wordsImg, wordsRects):
    for i, ca in enumerate(wordsRects):
        r1 = Rect(ca[0], ca[1], ca[2], ca[3])
        for j, cb in enumerate(wordsRects):
            if i != j:
                r2 = Rect(cb[0], cb[1], cb[2], cb[3])
                res = r2.overlaps_on_y_axis_with(r1)
                if res:
                    wordsImg = lineblobfillfun(wordsImg, ca, cb)
    return wordsImg

def blobs_in_Box(r1, allblobs):
    count = 0
    blobs = []
    for x, y, w, h in allblobs:
        r2 = Rect(x, y, w, h)
#        res = hostRect.overlaps_with(r2)
        if r1.overlaps_with(r2) or r1.is_point_inside_rect(r2.center) or r2.is_point_inside_rect(r1.center):
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

def wordGapsInBox(linesRect, wordsRect):
    linesRect.sort(key=lambda x: x[1])
    wordGap = []
    for line in linesRect:
        r1 = Rect(line[0], line[1], line[2], line[3])
        row_blobs, count = blobs_in_Box(r1, wordsRect)
        if count > 1:
            row_blobs.sort(key = lambda x: x)
            gapRect = horizontal_gap_bw_Boxes(row_blobs)
            if len(gapRect) > 0:
                wordGap.append([gapRect])
    return wordGap

def borderNoiseRemoval(binary, border):
    h, w = binary.shape
    cv2.rectangle(binary, (0,0), (w, border), 0, cv2.FILLED)
    cv2.rectangle(binary, (0, 0), (border, h), 0, cv2.FILLED)
    cv2.rectangle(binary, (0, h-border), (w, h), 0, cv2.FILLED)
    cv2.rectangle(binary, (w-border, 0), (w, h), 0, cv2.FILLED)           
    return binary  

def saveImage(outpath, outImage, savename):
	if isDebug:
		head, filename =  os.path.split(outpath)
		filenamewithoutXtn = os.path.splitext(filename)[0]
		savepath = os.path.join(head ,filenamewithoutXtn + savename)
		cv2.imwrite(savepath, outImage)

	
def process(srcImg, outpath):
    binary = binarize(srcImg)
    binary = borderNoiseRemoval(binary, 60)
    binary,linesRemovalImg = Line_Removal_fn(binary)
    print('outpath', outpath)
    saveImage(outpath, binary, '_binary.jpg')
    wordsImg, morph = img2words(binary)
    saveImage(outpath, morph, '_morph.jpg')
    kernel = np.ones((4, 1), np.uint8)
    wordsImg = cv2.erode(wordsImg, kernel, iterations = 2)
    saveImage(outpath, wordsImg, '_wordsImg.jpg')
    wordsRect = img2Rect(wordsImg)
    linesImg = formLines(wordsImg, wordsRect)
    saveImage(outpath, linesImg, '_linesImg.jpg')
    linesRect = img2Rect(linesImg)
    wordGapRect =  wordGapsInBox(linesRect, wordsRect)
    whitespace = np.zeros((srcImg.shape), dtype = np.uint8)
    for spaceInline in wordGapRect:
        for eachspace in spaceInline:
            for x, y, w, h in eachspace:
#                cv2.rectangle(srcImg, (x, y), (x+w, y+h), (255, 255, 0), -1)
                cv2.rectangle(whitespace, (x, y), (x+w, y+h), (255, 255, 255), -1)
    kernel = np.ones((4, 1), np.uint8)
    whitespace = cv2.dilate(whitespace, kernel, iterations = 2)
    saveImage(outpath, whitespace, '_whiteSpace_drawn.jpg')
#    saveImage(outpath, srcImg, '_whiteSpaceColor_drawn.jpg')
    return whitespace

def makeDir(dirpath, dirname):
    dirpath = os.path.join(dirpath, dirname)
    if  os.path.exists(dirpath): #and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)
    return dirpath


if __name__ == '__main__':
    DIRECTORY = r'C:\Bhaskar\Programming\17-09-2019\WhitespaceIntegrated\test'
    files = os.listdir(DIRECTORY)
    xtnList = [".jpg", ".JPG", ".png" , ".TIFF"]
    print("Total files:{}".format(len(files)))
    for i,eachfile in enumerate(files):
        fname =  os.path.splitext(eachfile)[0]
        fileXtn = os.path.splitext(eachfile)[1]
        if fileXtn in xtnList:
            print("Filename: {}".format(eachfile))
            imgpath = os.path.join(DIRECTORY, eachfile)
            srcImg = cv2.imread(imgpath)
            outFolder = os.path.join(DIRECTORY,fname)
            if os.path.exists(outFolder):
                shutil.rmtree(outFolder)
            os.mkdir(outFolder)
            
            process(srcImg, outFolder)
            