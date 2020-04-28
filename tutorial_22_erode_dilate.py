import cv2 as cv

"""
在OpenCV中实现形态学处理很简单，一般情况下对二值图像进行操作。
函数一般基本需要输入 两个参数：待处理图像和结构化元素(又称卷积核)用于决定操作性质。
两个基本的形态学操作是 腐蚀和膨胀。他们的变体构成了开运算、闭运算、梯度等等。
膨胀是将白色区域扩大，腐蚀是将黑色区域扩大。
"""



def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    # cv.destroyAllWindows()


"""
就像土壤侵蚀一样，这个操作会把前景物体的边界腐蚀掉（但是前景仍然 是白色）。
卷积核沿着图像滑动，如果与卷积核对应的原图像的所有像素值都是 1，
那么中心元素就保持原来的像素值，否则就变为零。
这会产生什么影响呢？根据卷积核的大小靠近前景的所有像素都会被腐蚀掉（变为 0），
所以前景物体会变小，整幅图像的白色区域会减少。
这对于去除白噪声很有用，也可以用来断开两个连在一块的物体等。
"""
def erode_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv_show("binary", binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5)) # 可以修改卷积核大小来增加腐蚀效果，越大腐蚀越强
    dst = cv.erode(binary,kernel=kernel,iterations=1) # 参数3是腐蚀次数，默认为1
    cv_show('erode_demo',dst)

"""
与腐蚀相反，与卷积核对应的原图像的像素值中只要有一个是 1，中心元 素的像素值就是 1。
所以这个操作会增加图像中的白色区域（前景）。一般在去 噪声时先用腐蚀再用膨胀。
因为腐蚀在去掉白噪声的同时，也会使前景对象变 小。所以我们再对他进行膨胀。
这时噪声已经被去除了，不会再回来了，但是 前景还在并会增加。
膨胀也可以用来连接两个分开的物体。
"""
def dilate_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv_show("binary", binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.dilate(binary, kernel=kernel)
    cv_show("dilate_demo", dst)


def BGRImg(img): # 可以不进行灰度处理，直接对彩色图像腐蚀，膨胀

    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    dst = cv.erode(img,kernel=kernel)
    cv_show('erode', dst)
    dst = cv.dilate(dst,kernel=kernel)
    cv_show('dilate', dst)


if __name__ == '__main__':
    img = cv.imread("./images/dige.png")
    cv_show('img',img)

    # erode_demo(img) # 毛刺消去
    # dilate_demo(img) # 毛刺变粗

    BGRImg(img)
