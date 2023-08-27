import copy
import csv
import io
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from PyQt5 import QtGui, QtWidgets
from scipy import ndimage, optimize
from scipy.ndimage import filters

import ImageClass
from FindCenter import findCenter, getBiImList, getEllipse, getView, showIm
from myFigure import myFigure
from myFunc import a16a8, detect_peaks, getStrVal, pil16pil8
from myMath import fitCirc, fitLine


def getRing(dataList,tol,kernelSize,nIter, ringCenterY, ringSizePrev, meanSizeRate,\
             centralPlane, ringCenterXPrev):
    karnel = np.ones([kernelSize,kernelSize])
    karnel = 1.0*karnel/sum(karnel)
    dataListConv = []
    maximaList = []
    maxPointsX = []
    maxPointsY = []
    for data in dataList:
        data = np.array(filters.gaussian_filter(data, kernelSize), dtype=np.uint8)
        if np.sum(data) == 0:
            data = dataListConv[-1]
        maxima, x, y = detect_peaks(data)
        maximaList.append(maxima)
        dataListConv.append(data)
        maxPointsX.append(x)
        maxPointsY.append(y)
    pointsX, pointsY, pointsXerr, planes, ringCenterYEst, ringSizeEst, ringCenterXEst = estimateRing(dataListConv, centralPlane, ringCenterXPrev, tol, meanSizeRate, ringSizePrev)
    if len(pointsX)>0:
        for i in range(nIter-1):
            pointsZ = []
            planesZ = np.arange(planes.size)*zPixelScale #z positions of all planes
            largestSize, j = 0, 0
            for i in range(planes.size): #get points of only allowed planes
                if planes[i]>0:
                    pointsZ.append(planesZ[i])
                    pointsZ.append(planesZ[i])
                    if ringSizeEst[j]>largestSize:
                        largestSize = ringSizeEst[j]
                    j+=1
            if centralPlane==None: centralPlane=len(dataListConv)/2
            if ringCenterXPrev==None:
                imBiTmp = getBiImList([dataListConv[centralPlane]],1,Ham)[0][-1]
                eParams = getEllipse(imBiTmp)
                embMiddleX, embMiddleY = eParams[1] #Y is AP direction
                ringCenterXPrev=embMiddleX
            centerX, centerY, radius = sizeFilterFit(ringSizeEst, pointsX, pointsXerr, pointsZ,\
                                                     ringSizePrev, meanSizeRate, [ringCenterXPrev, centralPlane*zPixelScale])
            if radius > 0:
                ringSizeEst = np.zeros(len(dataListConv))
                planes = np.zeros(len(dataListConv))
                pointsX, pointsY, pointsXerr = [], [], []
                ringSize = np.zeros(len(dataListConv))
                for i in range(len(dataListConv)):
                    if abs(i*zPixelScale-centerY)<radius:
                        ringSize[i]=2*np.sqrt(radius**2-(i*zPixelScale-centerY)**2)
                        first, second, errF, errS = refineRing(dataListConv[i], tol, centerX, ringCenterYEst, ringSize[i])
                        if first!=None:
                            pointsX.append(first[1])
                            pointsY.append(first[0])       
                            pointsX.append(second[1])
                            pointsY.append(second[0])
                            pointsXerr.append(errF)
                            pointsXerr.append(errS)
                            planes[i]=1
                    else:
                        ringSize[i]=0
                tol = tol/1.5
            else:
                centerX, centerY = findMidbody(dataList,ringCenterY,tol,ringCenterXPrev)
                break
    else:
        radius = 0
        centerX, centerY = findMidbody(dataList,ringCenterY,tol,ringCenterXPrev)
    return pointsX, pointsY, pointsXerr, planes, ringCenterYEst

def estimateRing(dataListConv, centralPlane, ringCenterXPrev, tol, meanSizeRate, ringSizePrev):
    planes = np.zeros(len(dataListConv))
    pointsX, pointsY, pointsXerr = [], [], []
    ringCenterY = np.array([])
    ringCenterX = np.array([])
    ringSize = np.array([])
    if centralPlane == None: 
        k = 2*len(dataListConv)/4
    elif ringSizePrev>100:
        k = 3*max(0,min(centralPlane,len(dataListConv)-1))/4
    else:
        k = max(0,min(centralPlane,len(dataListConv)-1))
    frames1=range(k,-1,-1)
    frames2=range(k+1,len(dataListConv))
    if check: print('k={0}, first set {1} frames, second set {2} frames'.format(k, len(frames1), len(frames2)))
    for i in frames1:
        first, second, errFirst, errSecond = findRing(dataListConv[i], tol, ringSizePrev, ringCenterY, ringCenterX, ringSize, meanSizeRate, ringCenterXPrev)
        if first != None:
            ringCenterY = np.append(ringCenterY,(first[0]+second[0])/2)
            ringCenterX = np.append(ringCenterX,(first[1]+second[1])/2)
            ringSize = np.append(ringSize,abs(second[1]-first[1]))
            pointsX.append(first[1])
            pointsY.append(first[0])       
            pointsX.append(second[1])
            pointsY.append(second[0])
            pointsXerr.append(errFirst)
            pointsXerr.append(errSecond)
            planes[i]=1
        if check:#Check which maxima are found. All maxima in red, ring maxima in blue 
            maxima, maxPointsX, maxPointsY = detect_peaks(dataListConv[i])
            checkMaximaFound(dataListConv[i],[maxPointsX,maxPointsY],first,second, np.mean(ringCenterX),np.mean(ringCenterY))
    pointsX=pointsX[::-1]
    pointsY=pointsY[::-1]
    pointsXerr=pointsXerr[::-1]
    if ringCenterY.size>1: ringCenterY=ringCenterY[::-1]
    if ringSize.size>1: ringSize=ringSize[::-1]
    if ringCenterX.size>1: ringCenterX=ringCenterX[::-1]
    for i in frames2:
        first, second, errFirst, errSecond = findRing(dataListConv[i], tol, ringSizePrev, ringCenterY, ringCenterX, ringSize, meanSizeRate, ringCenterXPrev)
        if first != None:
            ringCenterY = np.append(ringCenterY,(first[0]+second[0])/2)
            ringCenterX = np.append(ringCenterX,(first[1]+second[1])/2)
            ringSize = np.append(ringSize,abs(second[1]-first[1]))
            pointsX.append(first[1])
            pointsY.append(first[0])       
            pointsX.append(second[1])
            pointsY.append(second[0])
            pointsXerr.append(errFirst)
            pointsXerr.append(errSecond)
            planes[i]=1
        if check:#Check which maxima are found. All maxima in red, ring maxima in blue 
            maxima, maxPointsX, maxPointsY = detect_peaks(dataListConv[i])
            checkMaximaFound(dataListConv[i],[maxPointsX,maxPointsY],first,second, np.mean(ringCenterX),np.mean(ringCenterY))
    return pointsX, pointsY, pointsXerr, planes, np.mean(ringCenterY), ringSize, ringCenterX

def findRing(data, tol, ringSizePrev, ringCenterY, ringCenterX,\
              ringSize, meanSizeRate,ringCenterXPrev): #finds two points that belong to the ring and satisfy criteria
        
    def popMaxima(maxima,data):
        pop = [np.where(data*maxima==np.max(data*maxima))[0][0],\
                    np.where(data*maxima==np.max(data*maxima))[1][0]]
        maxima[max(pop[0]-maxDist,0):min(pop[0]+maxDist,maxima.shape[0]),\
                   max(pop[1]-maxDist,0):min(pop[1]+maxDist,maxima.shape[1])]=0
        return pop, maxima
    
    def checkSizePrevUpper(point1, point2, ringSizePrev, meanSizeRate):
        if meanSizeRate!=0: return abs(point1[1]-point2[1])<ringSizePrev-0.5*meanSizeRate+tol
        else: return abs(point1[1]-point2[1])<ringSizePrev
    
    def checkSizePrevLower(point1, point2, ringSizePrev, meanSizeRate):
        if meanSizeRate!=0: return 0.8*ringSizePrev-1*meanSizeRate-2*tol<abs(point1[1]-point2[1])
        elif not lateStage: return 0.5*ringSizePrev<abs(point1[1]-point2[1])
        else:return abs(point1[1]-point2[1])>20
    
    def checkCenterPrev(point1, point2, ringCenterXPrev, ringSizePrev):
        return abs(ringCenterXPrev-(point1[1]+point2[1])/2)<0.5*ringSizePrev/2
    
    def checkSize(point1,point2, ringSize):
        return ringSize[-1]-4*tol<abs(point1[1]-point2[1])<ringSize[-1]+2*tol
    
    def checkCenterY(point1,ringCenterY):
        return abs(ringCenterY-point1[0])<3*tol
    
    def checkCenterX(point1, point2, ringCenterX, ringSizePrev):
        return abs(ringCenterX-(point1[1]+point2[1])/2)<min(2*tol,0.5*ringSizePrev)
    
    def checkYEmb(point1, embMiddleY):
        return embMiddleY-20<point1[0]<embMiddleY+100
    
    def checkSameY(point1, point2):
        return abs(point1[0]- point2[0]) < 4*tol
    
    def checkXEmb(point1, point2, embMiddleX):
        return not (abs(embMiddleX-point1[1])<30 or abs(embMiddleX-point2[1])<30)
    
    def checkPoints(point1, point2, ringCenterX, ringCenterY, ringSize, ringCenterXPrev, ringSizePrev, meanSizeRate, embMiddleX, embMiddleY, lateStage):
        if not checkSizePrevUpper(point1, point2, ringSizePrev, meanSizeRate):
            if check: print('large size relative to previous ring', point2)    
            return False
        if ringSize.size==0 and not checkSizePrevLower(point1, point2, ringSizePrev, meanSizeRate):
            if check: print('small size relative to previous ring', point2)    
            return False
        if not checkCenterPrev(point1, point2, ringCenterXPrev, ringSizePrev):
            if check: print('wrong center relative to previous ring', point2)    
            return False
        if ringSize.size>0 and not checkSize(point1, point2, ringSize):
            if check: print('wrong size', point2)    
            return False
        if ringCenterX.size>0 and not checkCenterX(point1, point2, np.mean(ringCenterX), ringSizePrev):
            if check: print('wrong center X', point2)    
            return False
        if not checkSameY(point1, point2):
            if check: print('not on on the same line', point2)    
            return False
        if not lateStage and not checkXEmb(point1, point2, embMiddleX):
            if check: print('too close to the embryo middle X', point2)    
            return False
        return True
    
    maxima, maxPointsX, maxPointsY = detect_peaks(data)
    orderedMaxima = []
    newMaxima = copy.deepcopy(maxima)
    count = 0
    imBiTmp = getBiImList([data],1,Ham)[0][-1]
    eParams = getEllipse(imBiTmp)
    embMiddleX, embMiddleY = eParams[1] #Y is AP direction
    if check:
        print('embMiddleX = ',embMiddleX, 'embMiddleY = ',embMiddleY)
        print('ringCenterY', ringCenterY)
        print('ringCenterX',ringCenterX,'ringSize',ringSize, 'rate',meanSizeRate)
        print('ringSizePrev',ringSizePrev,'centXPrev',ringCenterXPrev)
    if ringCenterXPrev == None: ringCenterXPrev = embMiddleX
    while True:
        tmp, newMaxima = popMaxima(newMaxima,data)
        if count > 100 or tmp[1] == 0:
            print('Not found')
            return None, None, None, None
        if check:
            print('max [y,x]',tmp,data[tmp[0],tmp[1]])
        if checkYEmb(tmp, embMiddleY):
            if ringCenterY.size==0 or checkCenterY(tmp, np.mean(ringCenterY)):
                orderedMaxima.append(tmp)
                if check: print('point added')
                for point in orderedMaxima[:-1]:
                    if checkPoints(orderedMaxima[-1],point, ringCenterX, ringCenterY, ringSize, ringCenterXPrev, ringSizePrev, meanSizeRate, embMiddleX, embMiddleY, lateStage):
                        first = point
                        second = orderedMaxima[-1]
                        firstErr = getError(first,data)
                        secondErr = getError(second, data)
                        return first, second, firstErr, secondErr
            elif check: print('far from previous center Y')
        elif check: print('far from embryo middle')
        count+=1
        
def sizeFilterFit(ringSize, pointsX, pointsXerr, pointsZ, ringSizePrev,meanSizeRate, ringCenterPrev = [0,0]): #determines which points to consider for the ring fit
    ''' finds points to use for ring fitting '''
    def getErr(v):
        i = max(0,int(v[0]))
        k = min(int(np.ceil(v[1])),len(pointsX)/2)
        if i + 2 < k:
            centerX, centerY, radius, errorbars, resid = fitCirc(pointsX[2*i:2*k],pointsZ[2*i:2*k],pointsXerr[2*i:2*k],(ringSizePrev-0.3*meanSizeRate)/2, aveR=True)
            if np.isnan(errorbars[1]) or radius*2>2*max(ringSize) or radius*2>1.2*ringSizePrev:
                return 10000.
            if ringCenterPrev!=[0,0] and np.sqrt((centerX-ringCenterPrev[0])**2+(centerY-ringCenterPrev[1])**2)>1.4*ringSizePrev/2-radius: #the center movement can't allow crossing of the ring with previous
                return 10000.
            return errorbars[0]*errorbars[1]*(resid/(k-i))**1.#*errorbars[1]
        else: return 10000.
    
    if ringCenterPrev[0] != 0: #check that all points are inside the previous ring
        indexes = []
        for i in range(len(pointsX)/2):
            if not (checkInRing([pointsX[2*i],pointsZ[2*i]], ringSizePrev, ringCenterPrev, tol = 0.4*ringSizePrev) and\
                    checkInRing([pointsX[2*i+1],pointsZ[2*i+1]], ringSizePrev, ringCenterPrev, tol = 0.4*ringSizePrev)):
                indexes.append(2*i)
                indexes.append(2*i+1)
        pointsXFull = pointsX
        pointsXerrFull = pointsXerr
        pointsZFull = pointsZ
        pointsX = list(np.delete(pointsX, indexes))
        pointsZ = list(np.delete(pointsZ, indexes))
        pointsXerr = list(np.delete(pointsXerr, indexes))
        if check:
            checkRingFigure([pointsXFull,pointsZFull,pointsXerrFull],[ringCenterPrev[0],ringCenterPrev[1],ringSizePrev/2],[pointsX,pointsZ,pointsXerr])
            print('outliers', indexes)       
    
    if len(pointsX) > 3: #remove outliers
        res = optimize.brute(getErr,[[0,len(pointsX)/2-1],[1,len(pointsX)/2]],Ns=len(pointsX)/2)
    else:
        res = [0,0]
    pointsBot = int(res[0])
    pointsTop = int(np.ceil(res[1]))
    pointsXFull = pointsX
    pointsXerrFull = pointsXerr
    pointsZFull = pointsZ
    pointsX = pointsX[2*pointsBot:2*pointsTop]
    pointsXerr = pointsXerr[2*pointsBot:2*pointsTop]
    pointsZ = pointsZ[2*pointsBot:2*pointsTop] #get points of only allowed planes
    if len(pointsX)>3:
            centerX, centerY, radius, errorbars = fitCirc(pointsX,pointsZ,pointsXerr,(ringSizePrev-0.3*meanSizeRate)/2)
    else:   
        radius = 0
        centerX = np.mean(pointsX)
        centerY = np.mean(pointsZ)
    if check:
        checkRingFigure([pointsXFull,pointsZFull,pointsXerrFull],[centerX,centerY,radius],[pointsX,pointsZ,pointsXerr])   
    return centerX, centerY, radius

def refineRing(data, tol, ringCenterX, ringCenterY, ringSize):
    maxima, maxPointsX, maxPointsY = detect_peaks(data)
    left = (ringCenterX-ringSize/2, ringCenterY)
    right = (ringCenterX+ringSize/2, ringCenterY)
    distL = np.sqrt((maxPointsX-left[0])**2+(maxPointsY-left[1])**2)
    distR = np.sqrt((maxPointsX-right[0])**2+(maxPointsY-right[1])**2)
    first = np.array([maxPointsX[np.argmin(distL)],maxPointsY[np.argmin(distL)]])[::-1]
    errFirst = getError(first,data)
    second = np.array([maxPointsX[np.argmin(distR)],maxPointsY[np.argmin(distR)]])[::-1]
    errSecond = getError(second,data)
    if check and first!= None:
        fig = plt.figure()
        ax = fig.add_subplot(111) #full
        ax.imshow(data)
        ax.autoscale(False)
        ax.plot(maxPointsX,maxPointsY, 'ro')
        ax.plot(first[1],first[0],'bo')
        ax.plot(second[1],second[0],'bo')
        ax.plot(ringCenterX,ringCenterY,'go')
        plt.show()
    return first, second, errFirst, errSecond
        
def getError(point, data): #calculate error using intensity values
    subArray = data[point[0],:]
    value = subArray[point[1]]
    #value thresh is estimated from image. No rigorous estimation
    thresh = 10
    err1 = subArray.size-point[1]
    for i in range(subArray.size-point[1]):
        if subArray[point[1]+i]<value-thresh:
            err1 = i
            break
        elif subArray[point[1]+i]>value:
            value = subArray[point[1]+i]
    err2 = point[1]
    value = subArray[point[1]]
    for k in range(point[1]):
        if subArray[point[1]-k]<value-thresh:
            err2 = k
            break
        elif subArray[point[1]-k]>value:
            value = subArray[point[1]-k]
    error = max(err1,err2)
    return error

def findMidbody(dataList,ringCenterY,tol, centerXIni):#finds midbody
    maxIntPos = []
    maxInt = []
    for data in dataList:
        area = data[int(ringCenterY-tol):int(ringCenterY+tol),\
                    int(centerXIni-tol):int(centerXIni+tol)]
        tmp, x, y = detect_peaks(area)
        maxIntPos.append([np.where(area*tmp == np.max(area*tmp))[0][0],\
                          np.where(area*tmp == np.max(area*tmp))[1][0]])
        maxInt.append(area[maxIntPos[-1][0],maxIntPos[-1][1]])
    midBodIndex = int(np.mean(np.where(maxInt==np.max(maxInt))[0]))
    centerY = midBodIndex*zPixelScale
    centerX = maxIntPos[midBodIndex][1] + int(centerXIni-tol)
    return centerX, centerY

def readFromCsv(fileName):
    csvFile = csv.reader(open(fileName, 'rb'), delimiter=' ')
    rows = [row for row in csvFile]
    embryoDiam, embryoDiamErr = [float(rows[1][i]) for i in [2,3]]
    angle = float(rows[2][2])
    embCenterX, embCenterXerr = [float(rows[3][i]) for i in [2,3]]
    embCenterY, embCenterYerr = [],[]
    flag = False
    for row in rows[5:]:
        if flag:
            timePointStart = int(row[0])
            break
        if row[0] == 'timePoint':flag=True
        if not flag:
            embCenterY.append(float(row[1]))
            embCenterYerr.append(float(row[2]))
    embCenterY = np.array(embCenterY)
    embCenterYerr = np.array(embCenterYerr)
    return timePointStart, angle, embCenterX, embCenterY, embryoDiam, embCenterXerr, embCenterYerr, embryoDiamErr

def removeChromosome(dataList, z=None):
    from myFunc import maxIntensProject
    kernelSize = 1
    karnel = np.ones([kernelSize,kernelSize])
    karnel = 1.0*karnel/sum(karnel)
    dataChRem=[]
    if z==None: z=len(dataList)
    for i in range(len(dataList)/z):
#         data = np.array(signal.convolve2d(maxIntensProject(dataList[i*z:(i+1)*z]),karnel,'same')/kernelSize,dtype = np.uint8)
        data = maxIntensProject(dataList[i*z:(i+1)*z])
        ind = getChromosome(data)
        for j in range(z):
            tmp = np.copy(dataList[i*z+j])
            tmp[ind]=0
            dataChRem.append(tmp)
    return dataChRem
            
def getChromosome(image):
    th=255
    nr_objects, nr_object_prev = 0, 0
    NObj = 4
    if check: showIm(image)
    flag=True
    while flag and th>0:
        th-=2
        tmp=np.zeros_like(image)
        tmp[np.where(image>th)]=255
        labeled,nr_objects = ndimage.label(tmp)
        if nr_object_prev>NObj and nr_objects!=nr_object_prev: flag=False
        nr_object_prev = nr_objects
    th+=2
    if th==0: th=255
    tmp=np.zeros_like(image)
    tmp[np.where(image>th)]=255
    labeled,nr_objects = ndimage.label(tmp)
    centers = np.array(ndimage.measurements.center_of_mass(tmp, labeled, range(1,nr_objects+1)))
    ct = np.zeros((nr_objects, 3))
    ct[:,0] = np.arange(1,nr_objects+1)
    ct[:,1:]=centers
    centers =ct[ct[:,2].argsort()]
    for i in range(nr_objects-1):
        d = abs(centers[i,2]-centers[i+1,2])
        dy = abs(centers[i,1]-centers[i+1,1])
        if d<20 and dy>70: break
    if check:
        print(centers,d,dy,i)
        showIm(tmp)
    tmp=np.zeros_like(image)
    if d<20 and dy>70:
        tmp[np.where(labeled==centers[i,0])]=255
        tmp[np.where(labeled==centers[i+1,0])]=255
    tmp[np.where(ndimage.morphology.binary_dilation(tmp, iterations=5))]=255 
    if check: showIm(tmp)
    labeled,nr_objects = ndimage.label(tmp)
    return np.where(labeled!=0)

def checkMaximaFound(data,maxPoints,first,second,ringCenterX=None,ringCenterY=None): #plots image with all maxima [red] and found ring [blue].
    ''' displays image with local maxima points and ring positions '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(data)
    ax.autoscale(False)
    ax.plot(maxPoints[0],maxPoints[1], 'ro')
    if first!=None:
        ax.plot(first[1],first[0],'bo')
        ax.plot(second[1],second[0],'bo')
    if ringCenterX!=None:
        ax.plot(ringCenterX,ringCenterY,'go')
    print('checkMaximaFound', first, second)
    plt.show()

def checkInRing(point, diam, center, tol = 0):
    ''' returns whether point is inside the ring with center (x,y) and d=diam '''
    if ((point[0]-center[0])**2 + (point[1]-center[1])**2) <= (diam+tol)**2/4. and abs(point[0]-center[0])<diam/2:
        return True
    else:
        return False

def checkRingFigure(set1,circ,set2=None):#plots ring fitted to the data
    centerX, centerY, radius = circ
    theta = np.arange(0, 2*np.pi, 0.1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hold(True)
    circX = centerX + radius*np.cos(theta)
    circY = centerY + radius*np.sin(theta)
    if len(set1)>2:
        xerr = set1[2]
        ax.errorbar(set1[0],set1[1], xerr = xerr, color = [1,1,0], fmt='o')
    else:
        ax.scatter(set1[0],set1[1], color = [1,1,0])
    if set2!=None:
        if len(set2)>2:
            xerr = set2[2]
            ax.errorbar(set2[0],set2[1], xerr = xerr, color = [0,1,0], fmt='o')
        else:
            ax.scatter(set2[0],set2[1], color = [0,1,0])
#    ax.autoscale(False)
    ax.plot(circX,circY,color=[0,0,1], linestyle='solid')
    ax.scatter(centerX,centerY,color=[0,0,1], marker='+')
    print('checkRingFigure')
    plt.show()

def reportFindCenter(timePointStart, angle, embCenterX, embCenterY, embryoDiam, embCenterXerr, embCenterYerr, embryoDiamErr):
    print('Start ring search at slide = {}'.format(timePointStart))
    print('Embryo angle = {}'.format(angle))
    print('Embryo center position in X = {}'.format(getStrVal(embCenterX,embCenterXerr)))
    print('Embryo center position in Z(slide) = {}'.format(embCenterY))
    print('Embryo diameter = {}'.format(getStrVal(embryoDiam, embryoDiamErr)))

def processEmbryo():
    global check,lateStage
    
    app = QtWidgets.QApplication(sys.argv)
    angle, embCenterX, embCenterY, embryoDiam, embCenterXerr, embCenterYerr, embryoDiamErr = findCenter(filename,nSlices,zPixelScale,flip,mahFlag, Ham)
    reportFindCenter(timePointStart, angle, embCenterX, embCenterY, embryoDiam, embCenterXerr, embCenterYerr, embryoDiamErr)
     
    x = np.arange(embCenterY.size)
    slope, inter, se, ie = fitLine(x,embCenterY,embCenterYerr)
    def embCenterYlin(time):
        return inter + slope* time, time*se+ie
     
    ringSizePrev = embryoDiam
    outStringIO = io.StringIO()

    outStringIO.write('embryoDiam = '+ str(embryoDiam)+' '+str(embryoDiamErr)+'\n')
    outStringIO.write('angle = '+str(int(angle))+'\n'+'embryoCenterX = '+str(embCenterX)+' '+str(embCenterXerr)+'\n')
    outStringIO.write('timePint, embryoCenterY, embCenterYerr'+'\n')

    for i in range(len(embCenterY)):
        outStringIO.write(str(i)+' ' + str(embCenterY[i])+' '+str(embCenterYerr[i])+'\n')

    im = Image.open(filename)
    im.seek((timePointStart)*nSlices)
    dataList = []
    imList = []
    ringSize = []
     
    #--------------------------------------------------------#
    try:
        while np.sum(np.asarray(im))>0:
            if np.max(np.asarray(im))>0:
                dataList.append(a16a8(np.asarray(im.rotate(angle, expand=1))))
                imList.append(pil16pil8(im.rotate(angle, expand=1)))
            else:
                dataList.append(dataList[-1])
                imList.append(imList[-1])
            im.seek(im.tell()+1)
    except EOFError:
        pass # end of sequence
    del im
    if len(imList)%nSlices>0: imList = imList[:-(len(imList)%nSlices)]
    if len(dataList)%nSlices>0: dataList = dataList[:-(len(dataList)%nSlices)]
     
    if chromosomeMarker: dataList = removeChromosome(dataList, nSlices)
    theta = np.arange(0, 2*np.pi, 0.1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hold(True)
    ax.set_aspect(1)
    nRings = len(dataList)/nSlices
    bottom, top = 0, nSlices
    ringCenterY = None
    sizeList, sizeListErr = [], []
    rates = np.array([0])
    meanSizeRate = 0
    centerY = []
    centerX = []
    ringPosList = []
    centralPlane = None
    ringCenterXPrev = None
    yLimits = [0,0]
    xLimits = [dataList[0].shape[1]/2,dataList[0].shape[1]/2]
    outStringIO.write('timePoint centerX errorX centerY errorY radius errorR\n')
    for i in range(len(dataList)/nSlices):
#         if i==5: check=True
        print('ring #', i,'bottom',bottom,'top',top,'center',centralPlane, 'meanSizeRate', meanSizeRate)
        if nRings>3: color = [2.7*(i%(nRings/3))/nRings,1.8*(i%(nRings/2))/nRings,0.9*(i%(nRings/1))/nRings]
        else: color = [0,0,0.9*(i%(nRings))/nRings]
        if ringSizePrev - 0.2*meanSizeRate> 0 and top > bottom+1:
            ''' get ring values '''
            pointsX, pointsY, pointsXerr, planes, ringPositionNew = getRing(dataList[i*nSlices+bottom:i*nSlices+top],min(tol,ringSizePrev/2),kernelSize,2,\
                    ringCenterY, ringSizePrev, min(meanSizeRate,rates[-1]), centralPlane,ringCenterXPrev)
            lateStage=True
             
            ''' calculate circle fit '''
            pointsZ = []
            if sum(planes)>1 and not np.isnan(ringPositionNew):
                bottom = bottom+int(np.where(planes>0)[0][0])
                top = bottom+int(sum(planes))
                for tmp in range(bottom,top):
                    pointsZ.append(tmp*zPixelScale)
                    pointsZ.append(tmp*zPixelScale)
                tmpX, tmpY, radius, errorbars = fitCirc(pointsX, pointsZ, pointsXerr, 1.1*ringSizePrev/2-0.*min(meanSizeRate,rates[-1])/2)
                print('radius found {0}, radius ini {1}'.format(radius,ringSizePrev/2))
                if radius>1.1*ringSizePrev/2-0.001:
                    top = bottom+1
                    radius=0.
                else: top = min(nSlices,top+1)
            else:
                tmpX, tmpY = findMidbody(dataList[i*nSlices+bottom:i*nSlices+top],ringCenterY, tol, centerX[-1])
                tmpY = tmpY + bottom*zPixelScale
                radius = 0
                top = bottom
                 
            ''' reset ring position in ap axis '''
            if not np.isnan(ringPositionNew):
                print('new y pos found',ringPositionNew, 'prev pos', ringCenterY,2*radius/embryoDiam)        
                if radius>10 and ringCenterY==None:
                    ringPosList = [ringPositionNew]
                    ringCenterY = ringPositionNew
                elif 2*radius/embryoDiam>0.8:
                    ringPosList.append(ringPositionNew)
                    ringCenterY=ringPositionNew
                elif 2*radius/embryoDiam<=0.8 and radius>10:
                    ringPosList.append(ringPositionNew)
                    ringCenterY=np.median(ringPosList)
                print('new y pos',ringCenterY)
            centerX.append(tmpX)
            centerY.append(tmpY)
            centralPlane = int(centerY[-1]/zPixelScale)-bottom
            ringCenterXPrev = int(centerX[-1])
            sizeList.append(2*radius)
            sizeListErr.append(errorbars[2]*2)
            if len(sizeList)>1:
                rates = np.array([sizeList[k]-sizeList[k+1] for k in range(len(sizeList)-1)])
            meanSizeRate = np.mean(rates)
            ringSizePrev = 2*radius
            if radius>0:
                circX = centerX[-1] + radius*np.cos(theta)
                circY = centerY[-1] + radius*np.sin(theta)
                ax.scatter(pointsX, pointsZ, color = color, marker='o')
                if xLimits[0]>min(pointsX): xLimits[0]=min(pointsX)
                if xLimits[1]<max(pointsX): xLimits[1]=max(pointsX)
                if yLimits[0]>min(circY): yLimits[0]=min(circY)
                if yLimits[1]<max(circY): yLimits[1]=max(circY)
                ax.plot(circX,circY,color=color, linestyle='solid')
        else:
            tmpX, tmpY = findMidbody(dataList[i*nSlices+bottom-3:i*nSlices+top+3],ringCenterY, tol, centerX[-1])
            tmpY = tmpY + (bottom-3)*zPixelScale
            centerX.append(tmpX)
            centerY.append(tmpY)
            sizeList.append(0)
        if sizeList[-1]>0:
#             ''' FOR FLOW MAPS (z adjusted to keep on the surface) subtract each slide center '''
            if embryoCenterDrift == 'independent':outStringIO.write(str(timePointStart+i) + ' ' +\
                              str((centerX[-1]-embCenterX)*2/embryoDiam) + ' ' + str((errorbars[0]+embCenterXerr)*2/embryoDiam) + ' ' +\
                              str((centerY[-1]-embCenterY[timePointStart+i])*2/embryoDiam) + ' ' + str((errorbars[1]+embCenterYerr[timePointStart+i])*2/embryoDiam) + ' ' +\
                              str(sizeList[-1]/embryoDiam) + ' ' + str(errorbars[2]*2/embryoDiam) + '\n')
#             ''' ONLY FOR RING MAP (no z adjustment during imaging)! subtract line fit center '''
#             ''' NOTE: no error propagation from radius because it is very small <1%'''
            elif embryoCenterDrift == 'linear':outStringIO.write(str(timePointStart+i) + ' ' +\
                              str((centerX[-1]-embCenterX)*2/embryoDiam) + ' ' + str((errorbars[0]+embCenterXerr)*2/embryoDiam) + ' ' +\
                              str((centerY[-1]-embCenterYlin(timePointStart+i)[0])*2/embryoDiam) + ' ' + str((errorbars[1]+embCenterYlin(timePointStart+i)[1])*2/embryoDiam) + ' ' +\
                              str(sizeList[-1]/embryoDiam) + ' ' + str(errorbars[2]*2/embryoDiam) + '\n')
#             '''  subtract median value '''
            elif embryoCenterDrift == 'median':
                outStringIO.write(str(timePointStart+i) + ' ' +\
                              str((centerX[-1]-embCenterX)*2/embryoDiam) + ' ' + str((errorbars[0]+embCenterXerr)*2/embryoDiam) + ' ' +\
                              str((centerY[-1]-np.median(embCenterY))*2/embryoDiam) + ' ' + str((errorbars[1])*2/embryoDiam) + ' ' +\
                              str(sizeList[-1]/embryoDiam) + ' ' + str(errorbars[2]*2/embryoDiam) + '\n')
     
    f = open(fileOut, 'w')
    f.write('ringCenterY = '+str(ringCenterY)+'\n')
    f.write(outStringIO.getvalue())
    f.close()
    if timePointStart==0:
        f = open(fileOut[:-4]+'_0.csv', 'w')
        f.write('ringCenterY = '+str(ringCenterY)+'\n')
        f.write(outStringIO.getvalue())
        f.close()
    plt.ylim(yLimits[0]-30,yLimits[1]+30)
    plt.xlim(xLimits[0]-30,xLimits[1]+30)
    plt.ylabel('Z, pixels')
    plt.xlabel('X, pixels')
     
    sizeList = np.array(sizeList)
    sizeListErr = np.array(sizeListErr)
    ringSizes = sizeList[np.flatnonzero(sizeList)]
    ringSizesErr = sizeListErr[np.flatnonzero(sizeList)]
    x=np.arange(ringSizes.size)
    fig2 = myFigure()
    fig2.errorbar(x,ringSizes,ringSizesErr, color='k')
    fig2.ylabel('Ring Diameter, pixel')
    fig2.xlabel('time, s')
     
    dotRad = 3
     
    ''' draw embryo edge detection '''
    start, end = getView(dataList[2*nSlices/3+0*nSlices], Ham)
    for i in range(len(imList)):
        imDraw = ImageDraw.Draw(imList[i])
        imDraw.rectangle([0, start, imList[0].size[0], end], outline="blue")
        z = i%nSlices
        if ((embryoDiam/2)**2-(z*zPixelScale-embCenterY[timePointStart+i/nSlices])**2)>0:
            ringSize=np.sqrt((embryoDiam/2)**2-(z*zPixelScale-embCenterY[timePointStart+i/nSlices])**2)
            imDraw.line((embCenterX-ringSize,start,embCenterX-ringSize,end), "white")
            imDraw.line((embCenterX+ringSize,start,embCenterX+ringSize,end), "white")
        del imDraw
         
    for i in range(len(imList)):
        imDraw = ImageDraw.Draw(imList[i])
        z = (i%nSlices)*zPixelScale
        if sizeList[i/nSlices]>0:
            if ((sizeList[i/nSlices]/2)**2-(z-centerY[i/nSlices])**2)>0:
                ringSize=np.sqrt((sizeList[i/nSlices]/2)**2-(z-centerY[i/nSlices])**2)
                imDraw.ellipse((centerX[i/nSlices]-ringSize-dotRad,ringCenterY-dotRad,centerX[i/nSlices]-ringSize+dotRad,ringCenterY+dotRad), fill="white")
                imDraw.ellipse((centerX[i/nSlices]+ringSize-dotRad,ringCenterY-dotRad,centerX[i/nSlices]+ringSize+dotRad,ringCenterY+dotRad), fill="white")
        elif z == int(centerY[i/nSlices]):
            imDraw.ellipse((centerX[i/nSlices]-2,ringCenterY-2,centerX[i/nSlices]+2,ringCenterY+2), fill="blue")
        del imDraw
    imageWindow2 = ImageClass.ImageWindow(fileName = 'images', imSeq = imList)
    del imList
    imageWindow2.show()
    plt.show()
    sys.exit(app.exec_())  

if __name__ == '__main__':
    series = '01' #prefix in the file name, i.e. *series*Ring.tif (input images)
    timePointStart = 1  #slide number to start ring detection. Make sure that intensity in the equatorial band is higher than at the surrounding cortex
    flip = False #Check that anterior side is at the top, otherwise change to True
    folder = '' #folder name with the images
    filename = folder+series+'Ring.tif' #file name of the images with embryo z stacks
    nSlices = 30 #number of z slices in a stack
    zPixelScale = 4. # number of pixels in the camera per 1um (ie 1um = 250nm x 4 pixels)
    tol = 8 # number of pixels allowed for ring position misalignment
    chromosomeMarker = False #if chromosome marker is present, the algorithm will try to detect and remove them before processing
    lateStage=False #if the ring is half way closed or more at the first slide, change to True
    kernelSize = 5 # the size of the gaussian filter before processing (has to be odd)
    embryoCenterDrift = 'linear' #defines how the center of the embryo in z is calculated
    #                             'independent': calculate center for each timepoint independently (use when the depth is manually adjusted through out imaging)
    #                             'linear': the center depth is calculated from a linear fit to detected individual centers (use when a steady drift is observed, like imaging on agarose pads)
    #                             'median': calculates median position for the center (use only when there is no drift of the embryo center)
     
    #------------------------------Change these parameters only if you know what you are doing------------------------------------------------
    fileOut = folder+series+'Ring.csv'  #file name to output the data (important to have the same for other programs to work)
    maxDist = 5 # minimum distance between local intensity maxima when searching for ring local intensity peaks
    check = False #to check the process of searching and fitting step by step, change to True
    Ham=zPixelScale==10
    mahFlag = False
    processEmbryo()