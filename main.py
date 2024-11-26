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
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    thresh, im_bw = cv.threshold(im, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(im_bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    im = cv.drawContours(im, contours, -1, (0,255,0), 3)
    return im


img = cv.imread("SBVPI/1/1L_s_1_sclera.png")
#displayImage(img)
img = findPoints(img)
displayImage(img)
writeImage(img, outputFolder)

