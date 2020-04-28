import cv2 as cv
import numpy as np


def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    # cv.destroyAllWindows()

# 顶帽=原图像-开操作后图像,得到的是毛刺轮廓
def top_hat_demo(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv_show('binary',binary)
    # 这里的二值化图像就可以看作是原图像（注意：腐蚀膨胀是可以直接对彩色图像操作的）

    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    dst = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    cv_show("open img",dst) # 查看开操作后的图像

    dst = cv.morphologyEx(binary,cv.MORPH_TOPHAT,kernel)
    cv_show('top hot demo',dst) #查看顶帽图像

# 黑帽=闭操作图像-原图像
def black_hat_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv_show('binary', binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    dst = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    cv_show("close img", dst)  # 查看闭操作后的图像

    dst = cv.morphologyEx(binary, cv.MORPH_BLACKHAT, kernel)
    cv_show('black hat demo', dst)  # 查看黑帽图像

# 基本梯度（膨胀后的图像与腐蚀后的图像差值）
def basic_graditent_demo(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv_show('binary',binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    dst_dilate = cv.dilate(binary,kernel)
    dst_erode = cv.erode(binary,kernel)
    dst_gradient = cv.morphologyEx(binary,cv.MORPH_GRADIENT,kernel)
    dst = np.hstack((dst_dilate,dst_erode,dst_gradient))
    cv_show('basic_graditent',dst)

# 内部梯度（原图像减去腐蚀后的图像差值）
def internal_graditent_demo(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv_show('binary',binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT,(15,15))
    dst_erode = cv.erode(binary,kernel)
    cv_show('erode',dst_erode)
    dst = cv.subtract(binary,dst_erode)
    cv_show('internal_graditent',dst)

# 外部梯度（膨胀后图像与原图差值）
def external_graditent_demo(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv_show('binary',binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT,(15,15))
    dst_dilate = cv.dilate(binary,kernel)
    cv_show('dilate',dst_dilate)
    dst = cv.subtract(dst_dilate,binary)
    cv_show('external_graditent',dst)


if __name__ == '__main__':
    img1 = cv.imread('./images/dige.png')

    # top_hat_demo(img1)

    img2 = cv.imread("./images/black_hat.png")
    # black_hat_demo(img2)

    # basic_graditent_demo(img2)
    # internal_graditent_demo(img2)
    external_graditent_demo(img2)
