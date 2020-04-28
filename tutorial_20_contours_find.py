import cv2 as cv
import numpy as np


def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    # cv.destroyAllWindows()


def edge_demo(img):
    blurred = cv.GaussianBlur(img,(3,3),0)
    gray = cv.cvtColor(blurred,cv.COLOR_BGR2GRAY)

    edge_output = cv.Canny(gray,50,150)
    return edge_output


"""
findContours(image, mode, method, contours=None, hierarchy=None, offset=None)
    mode:轮廓检索模式
        RETR_EXTERNAL ：只检索最外面的轮廓；
        RETR_LIST：检索所有的轮廓，并将其保存到一条链表当中；
        RETR_CCOMP：检索所有的轮廓，并将他们组织为两层：顶层是各部分的外部边界，第二层是空洞的边界;
        RETR_TREE：检索所有的轮廓，并重构嵌套轮廓的整个层次; 建议用这个
    method:轮廓逼近方法
        CHAIN_APPROX_NONE：以Freeman链码的方式输出轮廓，所有其他方法输出多边形（顶点的序列）。
        CHAIN_APPROX_SIMPLE:压缩水平的、垂直的和对角分割，也就是，函数只保留他们的终点部分。
    offset:每一个轮廓点的偏移量. 当轮廓是从图像 ROI 中提取出来的时候，使用偏移量有用，因为可以从整个图像上下文来对轮廓做分析. 
    返回值有三个，第一个是图像，第二个是轮廓，第三个是（轮廓的）层析结构。
        轮廓（第二个返回值）是一个 Python 列表，其中存储这图像中的所有轮廓。
        每一个轮廓都是一个 Numpy 数组，包含对象边界点（x，y）的坐标。

drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)        
函数 cv2.drawContours() 可以被用来绘制轮廓。它可以根据你提供的边界点绘制任何形状。
    参数image是原始图像，参数contours是使用findContours检测到的轮廓数据，每个轮廓以点向量的形式存储。
    参数contourIdx是轮廓的索引（在绘制独立轮廓是很有用，当设置为 -1 时绘制所有轮廓）。
    参数color是轮廓的颜色;
    参数thickness是轮廓的宽度，为-1时表示填充整个轮廓
"""


def contours_demo(img):
    blurred = cv.GaussianBlur(img,(3,3),0) #高斯模糊，消除噪声
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU) # 二值化
    cv_show("binary image",thresh)

    # cloneImage,contours,hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cloneImage,contours,hierarchy = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) # 只检索最外面的轮廓

    # 第3个参数-1表示画出所有轮廓，第5个参数为-1时表示线填充轮廓;为正数时,表示绘制的轮廓的宽度
    res = cv.drawContours(img,contours,-1,(0,0,255),2)
    cv_show('detect contours',res)

def contours_demo2(img):
    binary = edge_demo(img) # 使用canny边缘检测获取二值化图像
    cloneImage,contours,hierarchy = cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) # 只检索最外面的轮廓

    for i,contour in enumerate(contours):
        cv.drawContours(img, contours, i, (0, 0, 255), 2)
        # cv.drawContours(img, contours, i, (0, 0, 255), -1) # -1表示用红色填充轮廓

        print(i)
    cv_show("contours_demo2",img)


if __name__ == '__main__':
    # src = cv.imread("./images/circle.png")
    # cv_show('demo',src)
    # contours_demo(src)

    src = cv.imread("./images/blob.png")
    cv_show('demo', src)
    contours_demo2(src)
