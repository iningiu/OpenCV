```
import cv2 as cv
import numpy as np

def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    # cv.destroyAllWindows()

"""
pyrDown(src, dst=None, dstsize=None, borderType=None)
    pyrDown函数先对图像进行高斯平滑，然后再进行降采样（将图像尺寸行和列方向缩减一半）
    参数dstsize表示降采样之后的目标图像的大小，如果不指定，则默认为((src.cols+1)/2, (src.rows+1)/2))
"""
def pyramid_demo(img): # 高斯金字塔
    level = 3 # 金字塔层数
    temp = img.copy()
    pyramid_images = []

    for i in range(level):
        dst = cv.pyrDown(temp) # 降采样,每一层都缩小为原来的1/4
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
```
