'''
Created on Jun 10, 2014

@author: renat

Collection of various mathematical functions
'''

import numpy as np
from numpy import pi
from lmfit import Parameters,minimize

def weightedAveStd(values, error):
    """
    Returns the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    weights = np.abs(values)/np.array(error)
    weights = weights/sum(weights)
    average = np.average(values, weights=weights)
    variance = np.dot(weights, (values-average)**2)/weights.sum()  # Fast and numerically precise
    return (average, np.sqrt(variance))

def fitCirc(x,y,xerr = None, rIni = None, aveR=False):
    
    """
    Performs a circle fit to data using least square residuals.

    Parameters
    ----------
    x : An array of length N.
    y : An array of length N.
    xerr : None or an array of length N,
        If provided, it is the standard-deviation of points.
        This vector, if given, will be used as weights in the fit.
    rIni : is a maximum radius of the circle to be fitted.
    aveR : if True, returns the average deviation from the fit.

    Returns
    -------
    xc, yc, R   : center and the radius of the circle.
    errorbars : errorbars on the center x, y and the radius.
    aveResid : (optional) average residual
    """
    
    x=np.array(x)
    y=np.array(y)
    if x.size<2: 
        print('fitCirc: not enough data points to fit circle')
        return
    x_m = np.mean(x)
    y_m = np.mean(y)
    if xerr == None or all(xerr)==0:
        xerr = np.ones(len(x))
    else:  
        xerr=np.array(xerr)
        xerr[np.where(xerr==0)]=100
    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x-xc)**2 + (y-yc)**2)
    
    def resid(pars):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
#         xc,yc, radius = pars
        v = pars.valuesdict()
        xc,yc, radius = v['xc'],v['yc'],v['radius']
        Ri = calc_R(xc,yc)
        if rIni is not None and radius>rIni:
            return 10000000*(Ri - radius) 
        return (Ri - radius)/np.array(xerr)
    
    center_estimate = x_m, y_m
    radius = calc_R(*center_estimate).mean()
    if rIni is not None and radius>rIni: radius = rIni
    
    params = Parameters()
    params.add('xc', x_m)
    params.add('yc', y_m)
    params.add('radius', radius, min=0)
    minzer=minimize(resid,params=params)
    res = minzer.params
    xc, yc, R = res['xc'].value,res['yc'].value,res['radius'].value
    errorbars = [res['xc'].stderr,res['yc'].stderr,res['radius'].stderr]
    aveResid = sum(abs(minzer.residual))/x.size
    if rIni is not None and R>rIni: print('radius greater than initial, resid=',aveResid)
    if aveR: return xc,yc,R,errorbars,aveResid
    else: return xc,yc,R, errorbars
    
def fitLine(xdata, ydata, ysigma=None, xsigma=None):

    """
    Performs a linear fit to data.

    Parameters
    ----------
    xdata : An array of length N.
    ydata : An array of length N.
    sigma : None or an array of length N,
        If provided, it is the standard-deviation of ydata.
        This vector, if given, will be used as weights in the fit.

    Returns
    -------
    a, b   : Optimal parameter of linear fit (y = a*x + b)
    sa, sb : Uncertainties of the parameters
    """
    if xsigma is not None:
        w=1.0/(ysigma**2+xsigma**2)
    elif ysigma is None:
        w = np.ones(len(ydata)) # Each point is equally weighted.
    else:
        w=1.0/(ysigma**2)

    w=w/sum(w)
    sw = sum(w)
    wx = w*xdata # this product gets used to calculate swxy and swx2
    swx = sum(wx)
    swy = sum(w*ydata)
    swxy = sum(wx*ydata)
    swx2 = sum(wx*xdata)
    
    a = (sw*swxy - swx*swy)/(sw*swx2 - swx*swx)
    b = (swy*swx2 - swx*swxy)/(sw*swx2 - swx*swx)
    
    
    if ysigma is None:
        chi2 = sum(((a*xdata + b)-ydata)**2)
    else:
        chi2 = sum((((a*xdata + b)-ydata)/ysigma)**2)
    dof = len(ydata) - 2
    rchi2 = np.sqrt(chi2/dof)
    
    sa = rchi2*np.sqrt(sw/(sw*swx2 - swx*swx))
    sb = rchi2*np.sqrt(swx2/(sw*swx2 - swx*swx))
    return a, b, sa, sb

def rotate(x,y,angle):
    
    '''
    Rotation of data by specified angle.
    
    Parameters
    ----------
    x : An array of length N.
    y : An array of length N.
    angle : angle in radians.
    '''
    
    x = np.array(x)
    y = np.array(y)
    xNew = x*np.cos(angle) - y*np.sin(angle) #rotate data
    yNew = x*np.sin(angle) + y*np.cos(angle)
    return xNew, yNew

def getAngle(x, y, xerr=None, yerr=None, index=None):
    
    '''
    Calculates angle in degrees between the data with errorbars and the positive x-axis using linear fit.
    
    Parameters
    ----------
    x : An array of length N.
    y : An array of length N.
    xerr, yerr : None or an array of length N,
        If provided, it is the standard-deviation of x and y data.
    index : indices of the points to use for alignment

    Returns
    -------
    xAlign, yAlign : Arrays of length N with aligned x and y coordinates.
    errAlign : An array on length N with 1D uncertainty for each point.
    aAngle : angle used for the aligning in radians.
    '''
    
    tol = 0.005
    aligned = False
    if xerr is not None and yerr!=None: xerr = np.sqrt(xerr**2 + yerr**2)
    elif yerr is not None: xerr = yerr
    else: xerr = np.ones_like(x)
    angleFin = 0 #getAnglePC(x,y)
    xNew, yNew = rotate(x, y, angleFin)
    if abs(np.std(xNew))<abs(np.std(yNew)):angleFin+=pi/2
    while not aligned:
        xNew, yNew = rotate(x, y, angleFin)
        if index == None: slope, inter, se, ie = fitLine(xNew, yNew, xerr) #find slope
        elif index.size <= 1: break
        else: slope, inter, se, ie = fitLine(xNew[index], yNew[index], xerr[index])
        if abs(slope)< tol:
            aligned = True
            break
        angle = -np.arctan(slope)
        angleFin += angle #angle to turn the data
       
    if np.sign(xNew.mean())<0: angleFin += pi
    aAngle = -angleFin
    if aAngle>=2*pi: angleFin -= 2*pi
    if aAngle<0: aAngle += 2*pi
    return aAngle #minus comes from conversion from turning angle to angle of data w/ X axis

def getSubIndex(a, left, right):
    if left>right: left, right = right, left
    if a[0]>a[-1]: return a.size-getSubIndex(a[::-1], left, right)-1
    else: return np.where(a[np.where(a <= right)[0]] >= left)[0]

def getSig(x):
    x=abs(x)
    if x!=np.inf and not np.isnan(x): return -int(np.floor(np.log10(x)))
    else: return 0
