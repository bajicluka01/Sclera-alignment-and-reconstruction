import cv2 as cv
import numpy as np
import os
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

    #here we just take the longest contour (works fine on almost all images, but might need to be improved)
    #21L_r_4_sclera.png
    #14R_l_4_sclera.png ...

    contours = sorted(contours, key=len, reverse=True)
    if len(contours) != 0:
        cnt = contours[0]
        m = cv.moments(cnt)
        if m['m00'] != 0:
            centroid_x = int(m['m10']/m['m00'])
            centroid_y = int(m['m01']/m['m00'])
            image = cv.circle(image, (centroid_x, centroid_y), 10, (0,0,255), -1)
    else:
        print("zero!")
        return image

    left = tuple(cnt[cnt[:,:,0].argmin()][0])
    right = tuple(cnt[cnt[:,:,0].argmax()][0])
    top = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottom = tuple(cnt[cnt[:,:,1].argmax()][0])

    image = cv.circle(image, left, 15, (255,0,0), -1)
    image = cv.circle(image, right, 15, (255,0,0), -1)
    image = cv.circle(image, top, 15, (255,0,0), -1)
    image = cv.circle(image, bottom, 15, (255,0,0), -1)

    return image

def alignment(im):
    return 0

def normalization(im):
    return 0

def reconstruction(imgs):
    return 0

def read_images():
    imgs = []
    subjects = []
    dbfolder = "SBVPI/"
    for file in os.scandir(dbfolder):
        current = file.name
        subjects.append(current)
        for img in os.scandir(dbfolder+current+"/"):
            imgs.append(dbfolder+current+"/"+img.name)
    return subjects, imgs


ids, images = read_images()
scleras = [x for x in images if "sclera" in x]

#img = cv.imread("SBVPI/1/1L_s_1_sclera.png")
#displayImage(img)
#img = findPoints(img)

for scl in scleras:
    img = cv.imread(scl)
    img = findPoints(img)
    #print(scl)
    #displayImage(img)
writeImage(img, outputFolder)



