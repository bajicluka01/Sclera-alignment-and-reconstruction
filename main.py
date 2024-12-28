import cv2 as cv
import numpy as np
import os
from sklearn.decomposition import PCA
from scipy.spatial import distance
from model import Model
#import imutils

outputFolder = "output"

def displayImage(im, windowName, windowSize = [800, 600]):
    #to preserve ratio
    factor = im.shape[1] / windowSize[0]

    cv.namedWindow("resized window", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO) 
    #cv.resizeWindow("resized window", windowSize[0], windowSize[1])
    cv.resizeWindow("resized window", int(im.shape[1]/factor), int(im.shape[0]/factor))
    cv.imshow("resized window", im)
    k = cv.waitKey(0) # Wait for a keystroke in the window
    #a to exit
    if k == 97:
        exit()

def writeImage(im, out):
    cv.imwrite(out+"/test.png", im)

def findPoints(im):
    image = im.copy()
    im = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #thresh, im_bw = cv.threshold(im, 127, 255, cv.THRESH_BINARY)
    ret, thresh = cv.threshold(im, 127, 255, 0)
    #find contours
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    image = cv.drawContours(image, contours, -1, (0,255,0), 10)

    #find extremes

    #here we just take the longest contour (works fine on almost all images, but might need to be improved)
    #21L_r_4_sclera.png
    #14R_l_4_sclera.png 
    #36L_u_4_sclera.png
    contours = sorted(contours, key=len, reverse=True)
    center = (-1,-1)

    if len(contours) != 0:
        cnt = contours[0]
        m = cv.moments(cnt)
        if m['m00'] != 0:
            centroid_x = int(m['m10']/m['m00'])
            centroid_y = int(m['m01']/m['m00'])
            center = (centroid_x, centroid_y)
            image = cv.circle(image, center, 10, (0,0,255), -1)
        else:
            print("no centroid!")
            return image, [], []
    else:
        print("zero contours!")
        return image, [], []

    left = tuple(cnt[cnt[:,:,0].argmin()][0])
    right = tuple(cnt[cnt[:,:,0].argmax()][0])
    top = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottom = tuple(cnt[cnt[:,:,1].argmax()][0])

    pois = []
    pois.append(left)
    pois.append(right)
    pois.append(top)
    pois.append(bottom)

    image = cv.circle(image, left, 15, (255,0,0), -1)
    image = cv.circle(image, right, 15, (255,0,0), -1)
    image = cv.circle(image, top, 15, (255,0,0), -1)
    image = cv.circle(image, bottom, 15, (255,0,0), -1)

    #testing
    #pois = []
    #for pt in cnt:
    #    pois.append((pt[0][0], pt[0][1]))

    return image, center, pois

def alignment(im, center, pois):
    #pca = PCA(n_components=2).fit(pois)
    #print(len(pois))
    #angle = -45
    angle = 0
    if(len(pois) != 0):
        pca = PCA().fit(pois)
        #print(pca)
        angle1 = np.arctan2(*pca.components_[0])
        angle2 = np.arctan2(*pca.components_[1])
        #print(angle1, angle2)
    else: #set a meaningless center if it wasn't successfully calculated (only relevant for 3 images in SBVPI)
        center = (0,0)
    rotationMatrix = cv.getRotationMatrix2D(center, angle, 1.0)
    nrows, ncols, channels = im.shape
    rotated = cv.warpAffine(im, rotationMatrix, (ncols, nrows))
    return rotated

def normalization(im, center, pois):
    if len(pois) != 4:
        print("Incorrect amount of POIs")
        return []

    #todo: do the order more elegantly
    bb_topleft = (pois[0][0], pois[2][1])
    bb_bottomright = (pois[1][0], pois[3][1])

    cv.rectangle(im, bb_topleft, bb_bottomright, (0,0,255), 5)

    cropped = im[pois[2][1]:pois[3][1], pois[0][0]:pois[1][0]].copy()
    return cropped

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

def similarity(im1, im2, covarianceMatrix):
    dist = distance.mahalanobis(im1, im2, covarianceMatrix)
    return dist

#necessary for Windows multiprocessing
if __name__ == '__main__':

    images = ['SBVPI/1/1L_l_1_sclera.png', 'SBVPI/1/1L_r_1_sclera.png', 'SBVPI/1/1L_s_1_sclera.png', 'SBVPI/1/1L_u_2_sclera.png']
    model = Model(images)

    #TODO isolate vessel images in separate folder
    #teach the model to use all images within a class to create a new image (i.e. reconstruction)






    exit()


    ids, images = read_images()
    scleras = [x for x in images if "sclera" in x]

    #img = cv.imread("SBVPI/1/1L_s_1_sclera.png")
    #displayImage(img)
    #img = findPoints(img)

    vessels = [x for x in images if "vessels" in x]

    im1 = cv.imread(vessels[0])
    im1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    #im1 = np.matrix.flatten(im1)
    im2 = cv.imread(vessels[1])
    im2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
    im2 = np.matrix.flatten(im2)

    im1mean = im1.mean()
    im2mean = im2.mean()
    #covariance = np.sum((im1-im1mean) * (im1-im1mean)) / (len(im1) - 1)
    #im1 = np.matrix.flatten(im1)
    #im1 = np.matrix.flatten(im1)
    #covarianceMatrix = np.cov(im1, im1)
    #covarianceMatrix = np.multiply(im1, np.matrix.transpose(im1))
    #covarianceMatrix = np.multiply(im1, im1)
    #covarianceMatrix = np.matrix.transpose(im1).dot(im1)
    #im1 = np.matrix.flatten(im1)

    #print(covarianceMatrix.shape)
    #print(im1.shape)
    #print(np.linalg.inv(covarianceMatrix))
    #print(similarity(im1, im1, np.linalg.inv(covarianceMatrix)))
    #print(similarity(im1, im1, covarianceMatrix))

    for scl in scleras:
        img = cv.imread(scl)
        img, center, pois = findPoints(img)
        #print(scl)
        displayImage(img, scl)

        img = alignment(img, center, pois)
        normalized = normalization(img, center, pois)
        #displayImage(normalized, scl, (normalized.shape[1], normalized.shape[0]))
        displayImage(normalized, scl)

    #writeImage(img, outputFolder)
