'''
Created on Feb 14, 2012

@author: renat
'''
import cv2, fitEllipse2, fitEllipse
from myMath import weightedAveStd, fitCirc, getSubIndex
from myFunc import a16a8, cropRotate
import matplotlib, Image, mahotas
import matplotlib.pyplot as plt
from scipy.ndimage import filters
import numpy as np


def showIm(im):
    cv2.imshow('img', im)
    cv2.waitKey()
    cv2.destroyAllWindows()

def getBoxSizeDiff(angle, im, noise, flag=False):
    imBi = im.rotate(angle, expand=1).point(lambda i: i>= noise and 255)
    imArray = np.asarray(imBi)
    boxSizeY = np.where(imArray==255)[0][-1] - np.where(imArray==255)[0][0]
    maximumIndex = np.argmax(np.where(imArray==255)[1])
    minimumIndex = np.argmin(np.where(imArray==255)[1])
    boxSizeX = np.where(imArray==255)[1][maximumIndex] - np.where(imArray==255)[1][minimumIndex]
    boxSizeDiff = boxSizeX - boxSizeY
    if flag:
        return boxSizeX, boxSizeY
    return boxSizeDiff

def getNoise(im):
    noise1 = 0
    noise0 = 255
    while noise0 > noise1+1:
        noise = (noise1+noise0)/2
        imBi = np.uint8(im>=noise)*255
        if sum([imBi[i][0] for i in range(imBi.shape[0])])+\
           sum([imBi[i][-1] for i in range(imBi.shape[0])])>0:
            noise1 = noise
        else:
            noise0 = noise
    return noise0

def getMaskArea(im, noise):
    return getMask(im,noise)[1]

def getMask(im, noise):
    imt = np.uint8(im>=noise)*255
    labeled, nr_objects = mahotas.label(imt)
    sizes = mahotas.labeled.labeled_size(labeled)
    if sizes.size>1:
        mask = 255*np.uint8(labeled==np.argmax(sizes[1:])+1)
        return mask, np.max(sizes[1:])
    else:
        return np.zeros_like(imt), 0

def getNoiseByArea(im ,areaBest = 120000, noise=None):
    nStep = 5
    if not noise: noise = 120
    area = getMaskArea(im, noise)
    if area < areaBest:
        while area < areaBest and noise<255:
            noise -= nStep
            area = getMaskArea(im, noise)
        areaPlus = area
        areaMinus = getMaskArea(im, noise + nStep)
        err = 100.*(abs(areaBest - areaPlus)+abs(areaMinus - areaBest))/areaBest
        if abs(areaPlus-areaBest) < abs(areaMinus-areaBest): return noise, err
        else: return noise + nStep, err
    else:
        while area > areaBest and noise>0:
            noise += nStep
            area = getMaskArea(im, noise)
        areaMinus = area
        areaPlus = getMaskArea(im, noise - nStep)
        err = 100.*(abs(areaBest - areaPlus)+abs(areaMinus - areaBest))/areaBest
        if abs(areaPlus-areaBest) < abs(areaMinus-areaBest): return noise - nStep, err
        else: return noise, err
    return noise, err

def getNoiseByAreaDiff(im , Ham, areaBest=20000, noise=70):
    if Ham: areaBest, delta = 120000, 50000 #Hamamazu
    else: areaBest, delta = areaBest, 5000  #QEM
    if areaBest<10000: areaBest=20000
    nStep = 2
    if np.max(im)==0:return 0, 0
    minArea = max(5000,areaBest - 1*delta)
    maxArea = areaBest + delta
    areaIni = getMaskArea(im, noise)
    if areaIni > minArea:
        nList = [noise]
        aList = [areaIni]
    else: nList, aList = [], []
    if areaIni < maxArea:
        area = areaIni
        noiseTmp = noise
        while area < maxArea and noiseTmp >= 0:
            noiseTmp -= nStep
            area = getMaskArea(im, noiseTmp)
            if area > minArea:
                nList.append(noiseTmp)
                aList.append(area)
        if len(nList)<2:
            nList.append(noiseTmp)
            aList.append(area)
    if areaIni > minArea:
        area = areaIni
        noiseTmp = noise
        while area > minArea and noiseTmp <= 256*256-1:
            noiseTmp += nStep
            area = getMaskArea(im, noiseTmp)
            if area < maxArea:
                nList.append(noiseTmp)
                aList.append(area)
        if len(nList)<2:
            nList.append(noiseTmp)
            aList.append(area)
    nList = np.sort(nList)[::-1]
    aList = np.sort(aList)
    aList = aList[np.where(aList<0.9*im.size)]
    nList = nList[np.where(aList<0.9*im.size)]
    if len(aList)<2:
        print('getNoiseByAreaDiff, short nList', nList, 'im max', np.max(im), 'areaIni={0}'.format(areaIni), areaBest)
        showIm(a16a8(im))
        if len(aList)>0:
            print('mask area={0}'.format(getMaskArea(im,nList[-1])))
    if len(nList)>1:    
        aDiff = np.abs(np.roll(aList,-1)-aList)#*((areaBest-aList)/1000.)**0.2
        aDiff[:-1] = filters.median_filter(aDiff[:-1], 5)
        ind = np.where(np.min(aDiff[:-1])==aDiff[:-1])[0][-1]
        if aList[ind]==0:ind = np.where(aList>0)[0][0]
        noise = nList[ind]
        err = aDiff[ind]/2./np.sqrt(30000*np.pi)
    else:
        return nList[0], 100
    return noise, err

def getBiImList(imList, nSlices, Ham, noise=120):
    imBiList = []
    errorsAll = []
    imBiTmp=[]
    errors=[]
    j=len(imList)/nSlices-1
    for k in range(1,nSlices+1): #threshold images in reverse
        noise, err = getNoiseByAreaDiff(imList[j*nSlices+(nSlices-k)], Ham, 15000, noise)
        mask = getMask(imList[j*nSlices+(nSlices-k)],noise)[0]
        imBiTmp.append(mask)
        errors.append(err)
    imBiTmp.reverse()
    errors.reverse()
    for j in range(len(imList)/nSlices):   
        imBiList += imBiTmp
        errorsAll += errors
    return imBiList, errors

def getBiImListMah(imList):
    imBiList = []
    for j in range(len(imList)):
        imMah = np.asarray(imList[j], dtype=np.uint32)
        noise = mahotas.thresholding.otsu(imMah)
        tmp = np.zeros_like(imMah)
        tmp[np.where(tmp>= noise)]=255
        imBiList.append(tmp)
    return imBiList

def removeOutliers(x,y,err):
    '''
    removes points very far from all others
    '''
    index = []
    i=0
    for point in x:
        if np.mean(x)-2*np.std(x) < point < np.mean(x)+2*np.std(x):
            pass
        else:
            index.append(i)
        i=i+1
    index.reverse()
    if len(index)>0:
        for i in index:
            x.pop(i)
            y.pop(i)
            err.pop(i)
    return x,y,err
    
def getEllipse(im):
    #Note: vertical ellipse
    imTmp = im.copy()
    contours = cv2.findContours(imTmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    tmp = contours[0]
    cont = np.array([p[0] for p in tmp])
    try: cPos, a, d, ang = fitEllipse2.fitellipse(cont,'linear')
    except RuntimeError: cPos, a, d, ang = (0,0), 0, 0, 0
    if a>d:
        a,d = d,a
        ang+= np.pi/2
    return ((a,d),cPos, ang)
    
def getEllipseMask(im, eParams):
    imTmp = np.zeros_like(im)
    ellipse = fitEllipse.create_ellipse(*eParams)
    ellipse = np.array([[[int(point[0]),int(point[1])]] for point in ellipse])
    bbox = np.array(cv2.boundingRect(ellipse))
    ''' fix box boundaries to be within image '''
    if bbox[0]<0:
        bbox[2]+=bbox[0]
        bbox[0]=0
    if bbox[1]<0:
        bbox[3]+=bbox[1]
        bbox[1]=0
    if bbox[0]+bbox[2]>=im.shape[1]:
        bbox[2] = im.shape[0]-bbox[0]-1
    if bbox[1]+bbox[3]>=im.shape[0]:
        bbox[3] = im.shape[1]-bbox[1]-1

    imTmp[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = 255
    for i in range(bbox[0],bbox[0]+bbox[2]):
        for j in range(bbox[1],bbox[1]+bbox[3]):
            if cv2.pointPolygonTest(ellipse,(i,j),False)<0: imTmp[j,i]=0
    return imTmp

def showEllipse(im, eParams):
    ellipse = fitEllipse.create_ellipse(*eParams)
    cv2.drawContours(im,[ellipse.astype(int)],-1, (100,0,0))
    cv2.circle(im,tuple(eParams[1].astype(int).tolist()),5,[100,0,0])
    showIm(im)
    
def fitSingleEmryo(images, errors, start, end, zPixelScale):
    '''
    fits an embryo from a single z stack.

    Parameters
    ----------
    images:         list of images in array format to fit the embryo.
                    The embryo in the images are oriented in Y direction with anterior part on top.
    errors:         list of errorbars, one for each image.
    start, end:     defines the region in the array to fit the embryo.
    zPixelScale:    distance in xy pixel sixes between z slices.

    Returns
    -------
    
    xCenter, zCenter, radius:    x and z position of the embryo center and its radius
    xSMerr, zSMerr, rSMerr:        standard error of mean of the position (position is the same in all measurements)
                                 and radius (radius varies, but we take mean, so error is how accurate we know mean)
    '''
    
    debug = False
    if debug:
        fig = matplotlib.pyplot.figure()
        ax = fig.add_subplot(111)
        ax.hold(True)
        theta = np.arange(0, 2*np.pi, 0.1*np.pi)
    
    centersX = []
    centersZ = []
    radiuses = []
    xCerr, zCerr, rerr = [], [], []
    for plane in range(start,end,10): #go through the region
        sliceList = [im[plane,:] for im in images]
        x1,x2,z1,z2, index = [],[],[],[],[]
        err1 = list(errors) #two points in each image, so errorbars are duplicated
        err2 = list(errors)
        for i in range(len(sliceList)):
            if sum(sliceList[i])>255*2: #in case black slide is present
                x1 = x1 + [np.where(sliceList[i]==255)[0][0]]
                x2 = x2 + [np.where(sliceList[i]==255)[0][-1]]
                z1 = z1 + [i*zPixelScale]
                z2 = z2 + [i*zPixelScale]
            else:
                index.append(i)
        index.reverse()
        if len(index)>0:
            for i in index:
                err1.pop(i)
                err2.pop(i)
        
        tmpX = x1+x2
        tmpZ = z1+z2
        errs = err1+err2
        try:
            centerX, centerZ, r, fitErrs = fitCirc(tmpX,tmpZ,errs)
            z = [centerX,centerZ]
            if debug: 
                ellipseX = z[0] + r*np.cos(theta)
                ellipseZ = z[1] + r*np.sin(theta)
                ax.plot(ellipseX,ellipseZ,color=[1.*(plane-start)/(end-start),0.8*(plane-start)/(end-start),0], linestyle='solid')
                ax.scatter(z[0],z[1],color=[1.*(plane-start)/(end-start),0.8*(plane-start)/(end-start),0], marker='+')
                ax.errorbar(tmpX,tmpZ,xerr=errs, color=[1.*(plane-start)/(end-start),0.8*(plane-start)/(end-start),0],fmt='o')
            
            centersX.append(z[0])
            centersZ.append(z[1])
            radiuses.append(r)
            xCerr.append(fitErrs[0])
            zCerr.append(fitErrs[1])
            rerr.append(fitErrs[2])
        except TypeError:
            print('skip slice')
    
    radius = np.median(radiuses)
    rstd = np.std(radiuses)
    while np.max(radiuses)>1.2*radius:
        centersX = np.delete(centersX, np.argmax(radiuses))
        centersZ = np.delete(centersZ, np.argmax(radiuses))
        xCerr = np.delete(xCerr, np.argmax(radiuses))
        zCerr = np.delete(zCerr, np.argmax(radiuses))
        rerr = np.delete(rerr, np.argmax(radiuses))
        radiuses = np.delete(radiuses, np.argmax(radiuses))
    xCenter, xstd = weightedAveStd(centersX, xCerr)
    xCenter = np.median(centersX)
    xSMerr = xstd / np.sqrt(len(centersX))
    zCenter, zstd = weightedAveStd(centersZ, zCerr)
    zCenter = np.median(centersZ)
    zSMerr = zstd / np.sqrt(len(centersZ))
    rSMerr = rstd / np.sqrt(len(radiuses))
    
    if debug:
#         fig.show()
        print('(xCenter, zCenter, radius), (xSMerr, zSMerr, rSMerr)',(xCenter, zCenter, radius), (xSMerr, zSMerr, rSMerr))
        plt.show()
    return (xCenter, zCenter, radius), (xSMerr, zSMerr, rSMerr)

def fitEmbryo(zStacks, errors, nSlices, start, end, zPixelScale):
    '''
    fits an embryo at different time points for each z stack.

    Parameters
    ----------
    zStacks:        list of images in array format to fit the embryo.
                    The embryo in the images are oriented in Y direction with anterior part on top.
    errors:         list of errorbars, one for each image.
    nSlices:        number of z slices per time point.
    start, end:     defines the region in the array to fit the embryo.
    zPixelScale:    distance in xy pixel sixes between z slices.

    Returns
    -------
    
    centersX, centersZ, radiuses:    lists of x and z positions of the embryo center and its radiuses for each time point
    cXerr, cZerr, rerr:              lists of standard error of mean of the positions (position is the same in all measurements)
                                     and standard deviation of the radius (radius varies, but we take mean, so error is confidence on mean)
    '''
    centersX = []
    centersZ = []
    radiuses = []
    cXerr, cZerr, rerr = [], [], []
    for k in range(len(zStacks)/nSlices): #k is a time point index
        images = zStacks[k*nSlices:(k+1)*nSlices]#[:2*nSlices/3]
        errs = errors[k*nSlices:(k+1)*nSlices]#[:2*nSlices/3]
        (xCenter, zCenter, radius), (xSMerr, zSMerr, rSMerr) = fitSingleEmryo(images, errs, start, end, zPixelScale)
        if radius<200 and zCenter>0:
            centersX.append(xCenter)
            centersZ.append(zCenter)
            radiuses.append(radius)
            cXerr.append(xSMerr)
            cZerr.append(zSMerr)
            rerr.append(rSMerr)
    centersX = np.array(centersX)
    centersZ = np.array(centersZ)
    radiuses = np.array(radiuses)
    cXerr = np.array(cXerr)
    cZerr = np.array(cZerr)
    rerr = np.array(rerr)
    return centersX, centersZ, radiuses, cXerr, cZerr, rerr
    
def getView(im,Ham=True):
    noise, err = getNoiseByAreaDiff(im,Ham)
    eParams = getEllipse(getMask(im,noise)[0])
    viewPos = int(eParams[1][1])
    start = viewPos - 50
    end =  viewPos - 0
    return start, end
    
def findCenter(filename,nSlices,zPixelScale,flip,mahFlag, Ham):
    
    '''
    Finds outline of the embryo and its center.

    Parameters
    ----------
    filename : name of the file to use
    nSlices : number of z slices per time point
    zPixelScale : distance in XY pixels between z planes
    flip : forced flip of the embryo 180 degrees
    mahFlag : use Mahotas thresholding algorithm instead of size dependent.
    Ham : Camera indicator. Hammamazu is true

    Returns
    -------
    angle : rotation angle of the embryo
    centerX, centersY, embryoDiam : position of the central axis in x direction, position of the central axis in z planes, average diameter of the embryo
    centerXerr, cYerr, embryoDiamErr : standard error in X direction, errors in Y direction, and standard error for the diameter.
    imageWindow : Qt window with drawn embryo and found outline
    '''
    
    kernelSize =1
    kernel = np.ones([kernelSize,kernelSize])
    kernel = 1.0*kernel
    im = Image.open(filename)
    im.seek(1*nSlices/4+0*nSlices)
    imConv = a16a8(filters.gaussian_filter(np.asarray(im),kernelSize))
    im = Image.fromarray(imConv)
    if mahFlag:
        noise = mahotas.thresholding.otsu(imConv)
    else:
        noise, err = getNoiseByAreaDiff(imConv, Ham)
    print('noise',noise)
    angle = getEllipse(getMask(imConv,noise)[0])[-1]*180/np.pi
    tmpIm = cropRotate((imConv, getEllipse(getMask(imConv,noise)[0]), False))
    if np.sum(tmpIm[:tmpIm.shape[0]/2,:])<np.sum(tmpIm[tmpIm.shape[0]/2:,:]):
        angle+=180
    angle+=180*flip
    print('angle=',angle)
    
    im = im.rotate(angle, expand=1)
    im.show()
    start, end = getView(np.asarray(im), Ham)
    im = Image.open(filename)
    imList,imPILList = [],[]
    arrayBiList = []
    errors = []
    print('step1, convolution')
    try:
        while np.sum(np.asarray(im))>0:
            data = a16a8(filters.gaussian_filter(np.asarray(im.rotate(angle, expand=1)),kernelSize))
            imPILList.append(Image.fromarray(data))
            imList.append(data)
            im.seek(im.tell()+1)
    except EOFError:
        pass # end of sequence
    del im, data
    if len(imList)%nSlices>0:
        imList = imList[:-(len(imList)%nSlices)]
        imPILList = imPILList[:-(len(imPILList)%nSlices)]
    
    print('step2, make binary images')
    for k in range(len(imList)/nSlices):
        if mahFlag:
            imBi = getBiImListMah(imList[k*nSlices:(k+1)*nSlices])
            err = np.ones(len(imList)).tolist()
        else:
            imBi, err = getBiImList(imList[k*nSlices:(k+1)*nSlices],nSlices,Ham, noise)
        arrayBiList = arrayBiList + imBi
        errors = errors + err
    del imBi
    
    print('step3, fit embryo')
    centersX, centersZ, radius, cXerr, cZerr, rerr = fitEmbryo(arrayBiList, errors, nSlices, start, end, zPixelScale)
    inds = getSubIndex(radius, 60, 80)
    embryoDiam, embryoDiamErr = weightedAveStd(radius[inds]*2, rerr[inds]*2)
    embryoDiam = int(embryoDiam)
    embryoDiamErr = embryoDiamErr/np.sqrt(radius.size)
    
    centerX, cXerr = weightedAveStd(centersX, cXerr)
    cXerr = cXerr/np.sqrt(centersX.size)
    centersZ = np.ones_like(centersZ)*np.mean(centersZ)
    cZerr = np.ones_like(centersZ)
    return angle, centerX, centersZ, embryoDiam, cXerr, cZerr, embryoDiamErr
    