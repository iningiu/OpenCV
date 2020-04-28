import cv2 as cv
import numpy as np


def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    # cv.destroyAllWindows()

def measure_object(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    print("threshold value:%s"%ret)
    cv_show('binary image',binary)

    outImage,contours,hierarchy = cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for i,contour in enumerate(contours):
        cv.drawContours(img,contours,i,(0,255,255),1) # 用黄色线条画出轮廓

        area = cv.contourArea(contour) # 计算轮廓面积
        print("contour area:", area)

        # 轮廓周长,第二参数可以用来指定对象的形状是闭合的（True）,还是打开的（一条曲线）
        perimeter = cv.arcLength(contour,True)
        print("contour perimeter:", perimeter)

        x,y,w,h = cv.boundingRect(contour) # 画出轮廓的外接矩形
        cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2) # 根据轮廓外接矩形返回数据，画出外接矩形

        rate = min(w, h) / max(w, h)  # 计算矩阵宽高比
        print("rectangle rate", rate)

        mm = cv.moments(contour)  # 求取轮廓的几何距
        print(type(mm))
        cx = mm['m10']/mm['m00'] # 计算出图像的重心
        cy = mm['m01']/mm['m00']
        cv.circle(img, (np.int(cx), np.int(cy)), 2, (0, 0, 255), -1)  # 根据几何距获取的中心点，画出中心圆
    cv_show("measure_object",img)

"""
approxCurve = approxPolyDP(curve, epsilon, closed, approxCurve=None) 轮廓逼近
第一个参数curve：输入的点集，直接使用轮廓点集contour
第二个参数epsilon：指定的精度，也即是原始曲线与近似曲线之间的最大距离。
第三个参数closed：若为true,则说明近似曲线是闭合的，反之，若为false，则断开。
返回值：输出的点集，当前点集是能最小包容指定点集的。画出来即是一个多边形；
"""
def contour_approx(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    print("threshold value: %s" % ret)
    cv_show("binary image", binary)

    outImage,contours,hierarchy = cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for i,contour in enumerate(contours):
        # cv.drawContours(img,contours,i,(0,0,255),2)
        epsilon = 0.01 * cv.arcLength(contour, True)
        approxCurve = cv.approxPolyDP(contour,epsilon,True)  # 4是与阈值的间隔大小，越小越易找出，True是是否找闭合图像
        print(type(approxCurve),approxCurve.shape)

        if approxCurve.shape[0] >= 7: # 圆 红色
            cv.drawContours(img,contours,i,(0,0,255),2) # 画出轮廓
            # cv.drawContours(img,[approxCurve],-1,(0,0,255),2) # 画出轮廓
        elif approxCurve.shape[0] == 4: # 矩形 黄色
            cv.drawContours(img,contours,i,(0,255,255),2)
            # cv.drawContours(img,[approxCurve],-1,(0,255,255),2) # 画出轮廓
        else: # 三角形 蓝色
            cv.drawContours(img,contours,i,(255,0,0),2)
    cv_show('contour_approx',img)


if __name__ == '__main__':
    img1 = cv.imread("./images/blob.png")
    cv_show('input_img',img1)
    # measure_object(img1)

    contour_approx(img1)
