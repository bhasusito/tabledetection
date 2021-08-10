    
import cv2
#import numpy as np


def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]




def rowcoldetect(tablexywh,colorimg,Horizontalimg,verticalimg,blank_image):

        #tabledatacoord=[]
        rowdraw=colorimg.copy()
        height,width=blank_image.shape
        #blank_image1 = np.zeros((height,width), np.uint8)
        #cv2.imwrite("beforcoordtext.png",colorimg)        
        NumofHorizontalline=cv2.bitwise_and(~blank_image,Horizontalimg)
        NumofVerticalline=cv2.bitwise_and(~blank_image,verticalimg)
        HVCombine=NumofHorizontalline+NumofVerticalline
#        cv2.imwrite("filledimg.png",blank_image)
#        cv2.imwrite("NumofVerticalline.png",NumofVerticalline)
#        cv2.imwrite("NumofHorizontalline.png",NumofHorizontalline)
        Horizontalcontours = cv2.findContours(NumofHorizontalline,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1]
        Horizontalcontours.sort(key=lambda x:get_contour_precedence(x, NumofHorizontalline.shape[1]))
        Verticalcontours = cv2.findContours(NumofVerticalline,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1]
        Verticalcontours.sort(key=lambda x:get_contour_precedence(x, NumofVerticalline.shape[1]))
        HContour=[]
        VContour=[]
        for hcontour in Horizontalcontours:
                hx, hy, hw, hz = cv2.boundingRect(hcontour)
                #print("HWidth",hw,"--",width)
                if width-10<hw and width+10>hw and hw > 5 and hz >5 :
                 cv2.rectangle(colorimg,(hx,hy),(hx+hw,hy+hz),(255,0,0),3)
                 HContour.append([hx, hy, hw, hz])
        for vcontour in Verticalcontours:
                vx, vy, vw, vz = cv2.boundingRect(vcontour)
                if vw > 5 and vz > 5:
                    cv2.rectangle(colorimg,(vx,vy),(vx+vw,vy+vz),(0,0,255),3)
                    VContour.append([vx, vy, vw, vz])
        #cv2.imwrite("Rowcol.png",colorimg)
        print(len(Horizontalcontours),"--",len(HContour))
        if   int(len(Horizontalcontours)/2)  < len(HContour)  :
                Rowrect=[]
                scale2=20
                verticalsize = round(width / scale2)
                verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,verticalsize))
                rowdata=[]
                for hc in range(len(HContour)-1):
		  #print(HContour[hc])
                  hx, hy, hw, hz = HContour[hc]
                  hx1, hy1, hw1, hz1 = HContour[hc+1]
		  #cv2.rectangle(rowdraw,(hx,hy),(hx1+hw1,hy1-5),(255,0,0),3)
                  Rowrect.append([hx,hy,hx1+hw1,hy1])
                  Linecrop=HVCombine[hy:hy1,hx:hx1+hw1]
		  
                  Rvertical = cv2.erode(Linecrop, verticalStructure, iterations=1)
                  Rvertical = cv2.dilate(Rvertical, verticalStructure, iterations=1)
                  RowVerticalcontours = cv2.findContours(Rvertical,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1]
                  RowVerticalcontours.sort(key=lambda x:get_contour_precedence(x, Rvertical.shape[1]))
                  columndata=[]
                  for Colval in range(len(RowVerticalcontours)-1):
                   Cx, Cy, Cw, Cz = cv2.boundingRect(RowVerticalcontours[Colval])
                   Cx1, Cy1, Cw1, Cz1 = cv2.boundingRect(RowVerticalcontours[Colval+1])
		   #hx1, hy1, hw1, hz1 = HContour[hc+1]
                   cv2.rectangle(rowdraw,(hx+Cx,hy+ Cy),(hx1+Cx1-5,hy1+Cy1-10),(0,255,0),3)
                   columndata.append([hx+Cx,hy+Cy, hx1+Cx1, hy1+Cy1]) 
                  rowdata.append(columndata) 
                #cv2.imwrite("rowdraw1.png",rowdraw)
        else:
                rowdata=[]
                #input("")
        return rowdata

        #input("")
	





