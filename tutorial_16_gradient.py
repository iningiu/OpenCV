"""
博客：
  https://www.cnblogs.com/ssyfj/p/9274745.html
  https://zhaoxuhui.top/blog/2017/06/23/%E5%9F%BA%E4%BA%8EPython%E7%9A%84OpenCV%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%8615.html#%E4%B8%89%E5%9B%BE%E5%83%8F%E6%A2%AF%E5%BA%A6
"""
import cv2 as cv
import numpy as np


def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    # cv.destroyAllWindows()


def sobel_demo(img):
    # 获取x轴方向的梯度,对x求一阶导，一般图像都是256，CV_8U但是由于需要进行计算，为了避免溢出，所以我们选择CV_32F
    # grad_x = cv.Sobel(img,cv.CV_32F,1,0)
    # 获取y轴方向的梯度，对y求一阶导
    # grad_y = cv.Sobel(img,cv.CV_32F,0,1)

    # Scharr算子是Sobel算子的增强版，当一些边缘比较弱，使用Sobel提取不到时，可以用Scharr算子
    grad_x = cv.Scharr(img, cv.CV_32F, 1, 0)
    grad_y = cv.Scharr(img, cv.CV_32F, 0, 1)

    gradx = cv.convertScaleAbs(grad_x) # 由于算完的图像有正有负，将其转回原来的uint8形式,取绝对值
    grady = cv.convertScaleAbs(grad_y)

    # 计算两个图像的权值和，dst = src1*alpha + src2*beta + gamma
    gradxy = cv.addWeighted(gradx,0.5,grady,0.5,0)

    cv_show('gradx',gradx)
    cv_show('grady',grady)
    cv_show('gradxy',gradxy)


def laplace_demo(img):
    dst = cv.Laplacian(img,cv.CV_32F)
    lpls = cv.convertScaleAbs(dst)
    cv_show("laplace_demo",lpls)

def custom_laplace(image):
    # 以下算子与上面的Laplace_demo()是一样的
    # kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) # 使用4卷积核算子去处理（是Laplacian默认）
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]) # 使用8卷积核处理，增强了
    dst = cv.filter2D(image, cv.CV_32F, kernel=kernel)
    lpls = cv.convertScaleAbs(dst)
    cv_show("custom_laplace", lpls)


if __name__ == '__main__':
    img = cv.imread('./images/lena.jpg')
    cv_show('lean',img)

    sobel_demo(img)

    # laplace_demo(img)

    # custom_laplace(img)
