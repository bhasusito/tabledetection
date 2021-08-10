import cv2

def binarize(inpImg):
    gry = cv2.cvtColor(inpImg, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(~gry, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return binary

def checkresize(width, height):
        if width > 500 and height > 500:
                return True
        else:
                return False
        
def Line_Removal_fn(src):
        rows, cols = src.shape[:2]
        if cols > 0 and rows > 0:
                if checkresize(cols, rows):
                        bw = src
                        horizontal = bw.copy()
                        vertical  = bw.copy()
                        bw_rows, bw_cols = bw.shape[:2]
                        if bw_cols >= bw_rows:
                                if bw_cols > 100 and bw_rows > 100:
                                        hor = round(bw_cols * 0.05)
                                        ver = round(bw_rows * 0.03)
                                else:
                                        fix = min(bw_rows, bw_cols)
                                        hor = fix
                                        ver = fix
                        else:
                                if bw_cols > 100 and bw_rows > 100:
                                        hor = round(bw_cols * 0.03)
                                        ver = round(bw_rows * 0.05)
                                else:
                                        fix = min(bw_rows, bw_cols)
                                        hor = fix
                                        ver = fix
                        horizontalsize = round(bw_cols / hor)
                        verticalsize = round(bw_rows / ver)
                else:
                        bw = src
                        horizontal = bw.copy()
                        vertical = bw.copy()
                        bw_rows, bw_cols = bw.shape[:2]
                        if bw_cols >= bw_rows:
                                if bw_cols > 100 and bw_rows > 100:
                                        hor = round((bw_cols * 3) / 100)
                                        ver = round((bw_rows * 2) / 100)
                                else:
                                        fix = min(bw_rows, bw_cols)
                                        hor = fix
                                        ver = fix
                        else:
                                if bw_cols > 100 and bw_rows > 100:
                                        hor = round((bw_cols * 2) / 100)
                                        ver = round((bw_rows * 3) / 100)
                                else:
                                        fix = min(bw_rows, bw_cols)
                                        hor = fix
                                        ver = fix
                        horizontalsize = hor
                        verticalsize = ver
                if horizontalsize == 0:
                        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                else:
                        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
                horizontal = cv2.erode(horizontal, horizontalStructure, 2)
                horizontal = cv2.dilate(horizontal, horizontalStructure, 2)
#                horizontalStructure.release()

                if verticalsize == 0:
                        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                else:
                        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
                vertical = cv2.erode(vertical, verticalStructure, 2)
                vertical = cv2.dilate(vertical, verticalStructure, 2)
#                verticalStructure.release()
#                join = horizontal + vertical
#                lineRmvd = src - join
#                twolines.append(join)
#                element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
#                horizontal = cv2.dilate(horizontal, horizontal, element)
#                vertical = cv2.dilate(vertical, vertical,element)
#                horizontal = bw & horizontal;
#                vertical = bw & vertical;
                join = horizontal + vertical
#                cv2.imwrite('lines.jpg', join)
                lineRmvd = src - join
        return lineRmvd,join
        
#if __name__ == "__main__":
#    
#    imgpath = r'C:\Confident\LineDetection_py\lines1.jpg'
#    srcImg = cv2.imread(imgpath)
#    srcImg = binarize(srcImg)    
#    lineRmvd = Line_Removal_fn(srcImg)
#    cv2.imwrite('text.jpg', lineRmvd)
#    openkernl = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#    opend = cv2.morphologyEx(lineRmvd, cv2.MORPH_OPEN, openkernl)
#    opend = cv2.cvtColor(opend, cv2.COLOR_GRAY2BGR)
#    cv2.imwrite('text1.jpg', opend)
                        
                                        

                                
	
