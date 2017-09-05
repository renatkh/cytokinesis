'''
Created on Jun 15, 2014

@author: renat

Collection of various commonly used functions
'''

import numpy as np
from PIL import Image
import cv2
from myMath import getSig
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from PyQt4 import QtGui

def pil16pil8(im):
    return Image.fromarray(pil16a8(im))

def a16a8(im, imMin=None, imMax=None):
    im = im.astype(np.float)
    if np.max(im)>0:
        if imMax is None: imMax = np.max(im[np.where(im>0)])
        if imMin is None: imMin = np.min(im[np.where(im>0)])
        if imMax>0 and imMin==imMax: imMin=0
        if imMax>0 and imMax!=imMin: im[np.where(im>0)] = (im[np.where(im>0)]-imMin)/(imMax-imMin)
    im = np.uint8(255*im)
    return im
        
def pil16a8(im, minV=None, maxV=None):
    imtmp = np.asarray(im, dtype=np.float32)
    if minV is None: minV = np.min(imtmp[np.where(imtmp>0)])
    if maxV is None: maxV = np.max(imtmp)
    if maxV<=255: return np.uint8(imtmp)
    res = 255.*(imtmp-minV)/(maxV-minV)
    res[np.where(res<0)]=0
    return np.uint8(res)

def PIL2QImage(pilimage):
    w,h = pilimage.size
    if pilimage.mode != "RGBA":
        pilimage = pilimage.convert("RGBA")
    if w % 4 != 0:
        w = w + 4 - (w % 4)
        pilimage = pilimage.resize([w, h])
    w,h = pilimage.size
    data = pilimage.tobytes("raw", "RGBA")
    qimage = QtGui.QImage(data, w, h, QtGui.QImage.Format_ARGB32)
    return qimage

def PIL2QPixmap(pilimage):
    qimage = PIL2QImage(pilimage)
    pix = QtGui.QPixmap.fromImage(qimage)
    return pix

def getStrVal(x,xerr):
    import myMath
    if xerr is None or np.isnan(xerr) or np.isinf(xerr): return '{0}+/-inf'.format(x)
    elif xerr==0: return '{0}+/-0'.format(x)
    else: precis=max(0,max(getSig(x),getSig(xerr)))
    if x is None or np.isnan(x): return '{1}+/-{2}'.format(precis,x,xerr)
    else: return '{1:.{0}f}+/-{2:.{0}f}'.format(precis,np.round(x,precis),np.round(xerr,precis))

def argclose(val,a):
    ''' returns index of the closest element in the array '''
    return np.argmin(np.abs(a-val))


def cropRotate(tmp):
    ''' crops ellipse and rotates the image.
    INPUT:
    tmp: tuple of im, eParams and flip.
    im: image
    flip: flip image 180 degrees for ap orientation
    eParams: ellipse parameters in form of (a,b), center, angle
    a,b: ellipse size
    center: numpy array with center coordinates in the image
    angle: angular orientation of the ellipse
    
    OUTPUT:
    cropped rectangular image of the ellipse size with zeros outside of the ellipse
    '''
    
    im, eParams, flip = tmp
    (a,b), center, angle = eParams
    dy,dx=im.shape[0]/2,im.shape[1]/2
    center = (center[0]+dx,center[1]+dy)
    imLarge = np.zeros((im.shape[0]+2*dy,im.shape[1]+2*dx)).astype(im.dtype)
    imLarge[dy:im.shape[0]+dy,dx:im.shape[1]+dx]=im
    im = imLarge
    eParams = (a+5,b+3), center, angle #add extra pix To make sure that all of the embryo is included
    im = cropEllipse(im, eParams)
    im32 = np.float32(im)
    x,y = im.shape[1]/2 - center[0], im.shape[0]/2-center[1]
    mapy, mapx = np.mgrid[0:im.shape[0],0:im.shape[1]].astype(np.float32)
    mapx = mapx-x
    mapy = mapy-y
    im = cv2.remap(im32, mapx, mapy, interpolation=cv2.INTER_LINEAR).astype(im.dtype)
    center = (np.array(im.shape)/2)[::-1]
    matrix = cv2.getRotationMatrix2D(tuple(center), angle*180/np.pi, 1.0)
    rotatedIm = cv2.warpAffine(im, matrix, im.shape)
    width, height = (int(2*a),int(2* b))
    top, bot = max(0,center[1]-height/2), min(rotatedIm.shape[0],center[1]+height/2)
    left, right = max(0,center[0]-width/2), min(rotatedIm.shape[1],center[0]+width/2)
    res = rotatedIm[top:bot, left:right]
    if flip: res=np.rot90(res,k=2)
    return res

def cropEllipse(im, eParams):
    ''' crops ellipse out of the image, the outside of the ellipse is black.
    NOTE: uses Cassini oval as the ellipse shape.
    INPUT:
    im: image
    eParams: ellipse parameters in form of (a,b), center, angle
    a,b: ellipse size
    center: numpy array with center coordinates in the image
    angle: angular orientation of the ellipse
    
    OUTPUT:
    image of the same size as im with zeros outside of the ellipse
    '''
    imTmp = np.zeros_like(im)
    ellipse = create_cassini_oval(*eParams)
    ellipse = np.array([[[int(point[0]),int(point[1])]] for point in ellipse])
    bbox = np.array(cv2.boundingRect(ellipse))
    ''' fix box boundaries to be within image '''

    imTmp[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = im[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    for i in range(bbox[0],bbox[0]+bbox[2]):
        for j in range(bbox[1],bbox[1]+bbox[3]):
            if cv2.pointPolygonTest(ellipse,(i,j),False)<0: imTmp[j,i]=0
    return imTmp

def create_cassini_oval(r, xc, alpha, n=100, angle_range=(0,2*np.pi)):
    """ Create points on an Cassini oval with uniform angle step
    reference: http://virtualmathmuseum.org/Curves/cassinian_oval/Cassinian_Oval.pdf
    
    Parameters
    ----------
    r: tuple
        (rx, ry): major an minor radii of the ellipse. Radii are supposed to
        be given in descending order. No check will be done.
    xc : tuple
        x and y coordinates of the center of the ellipse
    alpha : float
        angle between the x axis and the major axis of the ellipse
    n : int, optional
        The number of points to create
    angle_range : tuple (a0, a1)
        angles between which points are created.
        
    Returns
    -------
        (n * 2) array of points 
"""
    
    if r[0]<r[1]:
        r=(r[1],r[0])
        alpha-= np.pi/2
    R = np.array([
        [np.cos(alpha), -np.sin(alpha)],
        [np.sin(alpha), np.cos(alpha)]
    ])
    a0,a1 = angle_range
    angles = np.linspace(a0,a1,n)
    a = np.sqrt((r[0]**2-r[1]**2)/2)
    b = np.sqrt((r[0]**2+r[1]**2)/2)
    M = 2*a**2*np.cos(2*angles)+2*np.sqrt((-a**4+b**4)+a**4*np.cos(2*angles)**2)
    X = np.vstack([ np.cos(angles) *np.sqrt(M/2), np.sin(angles) * np.sqrt(M/2)]).T
    return np.dot(X,R.T) + xc

def maxIntensProject(imList):
    ''' Maximum intensity projection.
    INPUT:
    imList: list of images in numpy array form
    OUTPUT:
    single image of the size of input images
    '''
    
    return np.max(imList, axis=0)

def detect_peaks(image):
    """
    Takes an image and detect the peaks using the local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = ndimage.morphology.generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
#     local_max = filters.maximum_filter(image, footprint=neighborhood)==image
    local_max = filters.maximum_filter(image, size = 10)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = ndimage.morphology.binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask
    detected_peaks = local_max - eroded_background
    
    labeled, num_objects = ndimage.label(detected_peaks)
    slices = ndimage.find_objects(labeled)
    maxima = np.zeros(image.shape)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(int(x_center))
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(int(y_center))
        maxima[int(y_center),int(x_center)]=1
    return maxima, np.array(x), np.array(y)
