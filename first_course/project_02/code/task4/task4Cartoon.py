import cv2
num_down = 2
num_bilateral =7

img_rgb = cv2.imread("me.jpg")

# down sample image using Gaussian pyramid
colored_image = img_rgb
for _ in xrange(num_down):
    colored_image = cv2.pyrDown(colored_image)

# repeatedly apply small bilateral filter instead of
# applying one large filter
for _ in xrange(num_bilateral):
    colored_image = cv2.bilateralFilter(colored_image, d=9,sigmaColor=9,sigmaSpace=7)

# up sample image to original size
for _ in xrange(num_down):
    colored_image = cv2.pyrUp(colored_image)

# convert to grayscale and apply median blur
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
img_blur = cv2.medianBlur(img_gray, 7)

# detect and enhance edges
img_edge = cv2.adaptiveThreshold(img_blur, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blockSize=9,C=2)
# convert back to color, bit-AND with color image
img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
img_cartoon = cv2.bitwise_and(colored_image, img_edge)

# display
cv2.imshow("cartoon image", img_cartoon)
