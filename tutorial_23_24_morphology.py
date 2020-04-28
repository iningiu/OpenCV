import cv2 as cv
import numpy as np

"""
开运算:先腐蚀再膨胀,它被用来去除噪声。
闭运算:先膨胀再腐蚀。它经常被用来填充前景物体中的小洞，或者前景物体上的小黑点。
这里我们用到的函数是 cv2.morphologyEx(src, op, kernel)。
    参数2 op 用来指定操作类型：cv.MORPH_OPEN--开运算;cv.MORPH_CLOSE--闭运算
    
开闭操作作用：
    1. 去除小的干扰块-开操作
    2. 填充闭合区间-闭操作
    3. 水平或垂直线提取,调整kernel的row，col值差异 -开操作
    比如：采用开操作，kernel为(1, 15),提取垂直线，kernel为(15, 1),提取水平线，
    4. 消除干扰线-开操作

其他形态学操作：
    顶帽：原图像与开操作之间的差值图像
    黑帽：比操作与原图像直接的差值图像
    形态学梯度：其实就是一幅图像膨胀与腐蚀的差别。 结果看上去就像前景物体的轮廓
    基本梯度：膨胀后图像减去腐蚀后图像得到的差值图像。
    内部梯度：用原图减去腐蚀图像得到的差值图像。
    外部梯度：膨胀后图像减去原图像得到的差值图像。
"""

def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    # cv.destroyAllWindows()

def camp(val1,val2):
    val = val1 + val2
    if val > 255:
        return 255
    if val < 0:
        return 0
    return val

"""
开操作（先腐蚀后膨胀）
特点：消除噪点，去除小的干扰块，而不影响原来的图像
"""
def open_demo(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    for i in range(10000): # 为灰度图增加一些噪点
        h = np.random.randint(0,gray.shape[0]-1)
        w = np.random.randint(0,gray.shape[1]-1)
        val = np.random.randint(0,255)
        gray[h,w] = camp(gray[h,w],val) # 随机取一个像素点,它的像素值加上一个随机数

    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv_show('binary',binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    dst = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel=kernel)
    cv_show('open demo',dst)

"""
闭操作（先膨胀后腐蚀）
特点：可以填充闭合区域
"""
def close_demo(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv_show('binary',binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT,(6,6))
    dst = cv.morphologyEx(binary,cv.MORPH_CLOSE,kernel=kernel)
    cv_show('close demo',dst)

# 利用开操作提取水平垂直线
def extract_line(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv_show('binary', binary)

    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 1)) # 提取水平线
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 15)) # 提取垂直线
    dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel=kernel)
    cv_show('extract line', dst)

# 利用开操作消除干扰线
def eliminate_line(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv_show('binary', binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # 因为干扰线很细，小于我们想要的字母，先腐蚀后膨胀对字母无影响，但是对于细线在腐蚀的时候就处理掉了
    dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel=kernel)
    cv_show('eliminate line', dst)

# 利用开操作提取指定形状：通过getStructuringElement将我们的内核形状设置为想要的形状
def extract_shape(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv_show('binary', binary)

    """
   有时我们需要构建一个椭圆形 / 圆形的核。为了实现这种要求，OpenCV提供了
   函数cv2.getStructuringElement()。你只需要告诉他你需要的核的形状和大小。
       矩形：MORPH_RECT;
       交叉形：MORPH_CORSS;
       椭圆形：MORPH_ELLIPSE;
       """
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel=kernel)
    cv_show('extract shape', dst)


if __name__ == '__main__':
    img = cv.imread('./images/dige.png')
    # cv_show("orginal img",img)

    img2 = cv.imread('./images/close.png')
    # cv_show("orginal img", img2)

    # open_demo(img)
    # close_demo(img2)

    # img3 = cv.imread("./images/extract_line.png")
    # cv_show('extract line', img3)
    # extract_line(img3)

    # img4 = cv.imread("./images/abcd.png")
    # cv_show('eliminate_line', img4)
    # eliminate_line(img4)

    img5 = cv.imread("./images/extract_shape.png")
    cv_show('extract_shape', img5)
    extract_shape(img5)
