"""
学习博客：
    https://www.cnblogs.com/ssyfj/p/9273643.html；
    http://zhaoxuhui.top/blog/2017/05/30/%E5%9F%BA%E4%BA%8EPython%E7%9A%84OpenCV%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%8611.html
    
图像金字塔：用来进行图像缩放的
    高斯金字塔(Gaussian pyramid): 用于下采样，主要的图像金字塔
    拉普拉斯金字塔(Laplacian pyramid): 用来从金字塔低层图像重建上层未采样图像，在数字图像处理中也即是预测残差，可以对图像进行最大程度的还原，配合高斯金字塔
                                      一起使用。用高斯金字塔的每一层图像减去其上一层图像上采样并高斯卷积之后的预测图像，得到一系列的差值图像即为LP分解图像。
进行图像缩放可以用图像金字塔，也可以使用resize函数进行缩放，后者效果更好。                                      
"""

import cv2 as cv
import numpy as np

def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    # cv.destroyAllWindows()

"""
pyrDown(src, dst=None, dstsize=None, borderType=None)
    pyrDown函数先对图像进行高斯平滑，然后再进行下采样（将原图所有的偶数行和列去掉，图像尺寸变为原来的1/4）
    参数dstsize表示下采样之后的目标图像的大小，如果不指定，则默认为((src.cols+1)/2, (src.rows+1)/2))
"""
def pyramid_demo(img): # 高斯金字塔
    level = 3 # 金字塔层数
    temp = img.copy()
    pyramid_images = []

    for i in range(level):
        dst = cv.pyrDown(temp) # 下采样,每一层都缩小为原来的1/4
        pyramid_images.append(dst)
        cv_show('pyramid_down_'+str(i+1),dst)
        temp = dst.copy() # 作为下一层的输入
    return pyramid_images


"""
1.拉普拉斯金字塔使用的图片大小必须是2^n大小，或者是一个宽高相等的图片.
2.pyrUp(src, dst=None, dstsize=None, borderType=None)
    pyrUp函数先对图像进行升采样（将图像尺寸行和列方向增大一倍），然后再进行高斯平滑
    参数dstsize表示降采样之后的目标图像的大小，如果不指定，则默认为(src.cols*2, src.rows*2)
    borderType参数表示表示图像边界的处理方式
3.步骤：
    3.1 将图像在每个方向扩大为原来的两倍，新增的行和列以0填充
    3.2 使用先前同样的内核(乘以4)与放大后的图像卷积，获得 “新增像素”的近似值
"""
def laplace_demo(img):
    pyramid_images = pyramid_demo(img) #拉普拉斯需要用到高斯金字塔结果
    level = len(pyramid_images)

    for i in range(level-1,-1,-1): # 从塔尖开始,即最小的那张图片开始,2,1,0
        if i-1 < 0:
            expand = cv.pyrUp(pyramid_images[i],dstsize=img.shape[:2]) #先上采样
            lpls = cv.subtract(img,expand)  #使用高斯金字塔上一个减去当前上采样获取的结果，才是拉普拉斯金字塔
            cv_show('laplace_demo'+str(i),lpls)
        else:
            expand = cv.pyrUp(pyramid_images[i],dstsize=pyramid_images[i-1].shape[:2])
            lpls = cv.subtract(pyramid_images[i-1],expand)
            cv_show('laplace_demo'+str(i),lpls)


if __name__ == '__main__':
    img = cv.imread("./images/01.jpg")
    cv_show('lena',img)

    # pyramid_demo(img)

    laplace_demo(img)
