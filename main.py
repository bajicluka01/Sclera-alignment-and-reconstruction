import cv2 as cv
import numpy as np
#import imutils

outputFolder = "output"

def displayImage(im, windowSize = [800, 600]):
    #resize image to fit the window
    #(h, w) = im.shape[:2]
    #r = windowSize[0] / float(w)
    #dim = (windowSize[1], int(h * r))
    #im = cv.resize(im, dim)
    #im = cv.resize(im, (windowSize[0], windowSize[1]), interpolation=cv.INTER_AREA)
    #im = imutils.resize(im, height=windowSize[1])

    cv.namedWindow("resized_window", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO) 
    cv.resizeWindow("resized_window", windowSize[0], windowSize[1])
    cv.imshow("resized_window", im)
    k = cv.waitKey(0) # Wait for a keystroke in the window
    if k == 's':
        return

def writeImage(im, out):
    cv.imwrite(out+"/test.png", im)


def findPoints(im):
    image=im.copy()
    im = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #thresh, im_bw = cv.threshold(im, 127, 255, cv.THRESH_BINARY)
    ret, thresh = cv.threshold(im, 127, 255, 0)
    #find contours
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    image = cv.drawContours(image, contours, -1, (0,255,0), 10)

    #find extremes
    cnt = contours[0]
    m = cv.moments(cnt)
    centroid_x = int(m['m10']/m['m00'])
    centroid_y = int(m['m01']/m['m00'])

    image = cv.circle(image, (centroid_x, centroid_y), 10, (0,0,255), -1)

    left = tuple(cnt[cnt[:,:,0].argmin()][0])
    right = tuple(cnt[cnt[:,:,0].argmax()][0])
    top = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottom = tuple(cnt[cnt[:,:,1].argmax()][0])

    image = cv.circle(image, left, 15, (255,0,0), -1)
    image = cv.circle(image, right, 15, (255,0,0), -1)
    image = cv.circle(image, top, 15, (255,0,0), -1)
    image = cv.circle(image, bottom, 15, (255,0,0), -1)

    return image


img = cv.imread("SBVPI/1/1L_s_1_sclera.png")
#displayImage(img)
img = findPoints(img)
displayImage(img)
writeImage(img, outputFolder)

