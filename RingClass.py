'''
Created on Jun 10, 2014

@author: renat

contractile ring object class
'''

import csv
from myMath import getSubIndex, fitLine,getAngle, rotate
from scipy.optimize import leastsq
from myCKfunc import fitRingSizeHalf
from myFunc import a16a8, pil16pil8
import numpy as np

class RingClass(object):

    def __init__(self, ringFile, timeStep=35.7):
        self.file = ringFile
        self.label = ringFile.split('/')[-1]
        self.timeStep = timeStep # time in seconds between each ring
        self.ringSizeThresh = 0.3 # the threshold for the ring size to consider for ring closure angle
        self.rLinMax = 0.8 # maximum ring size to consider for linear rate fitting
        self.rLinMin = 0.3 # minimum ring size to consider for linear rate fitting
        self.nSlices = None
        self.aveR0 = 1.1
        self.dataList=None
    
    def rescaleTime(self, scaling=0):
        self.timeScaling = scaling
        # 0 - exponential model
        # 1 - exponential with arbitrary defined time origin
        # 2 - linear fit
        # 3 - fit to the average from the flow
        if scaling==0: self.rescaleTimeModel()
        elif scaling==1: self.rescaleTimeModelPolym()
        elif scaling==2: self.rescaleTimeLinear()
        elif scaling==3: self.rescaleTimeAvgFit()
        else: self.rescaleTimeFuncFit(scaling)
    
    def loadRing(self):
        ''' loads information from a Ring.csv file '''
        def getCenterXOldStyle(row):
            ''' returns centerX position from old style stored data '''
            centers = [float(row[2][1:-1])]
            for i in range(3, len(row)):
                centers.append(float(row[i][:-1]))
            return np.mean(centers), np.std(centers)
        def getCenterYOldStyle(rowPrev,row):
            ''' returns centerY position from old style stored data '''
            centersY = []
            for i in range(len(rowPrev)):
                try: centersY.append(float(rowPrev[i]))
                except: pass
            for i in range(len(row)):
                try: centersY.append(float(row[i]))
                except: pass
            return np.array(centersY), np.ones_like(centersY)*0.3
        print('loading ring ', self.label)
        csvFile = csv.reader(open(self.file, 'rb'), delimiter=' ')
        ringIndex, centerX, centerY, radius, centerXerr, centerYerr, radiuserr = [], [], [], [], [], [], []
        dist, disterr, embCenterY, embCenterYerr = [], [], [], []
        flag, flagEmbC, oldEmbCY = False, False, False
        rowPrev=None
        for row in csvFile:
            if row[0]=='ringPosition' or row[0]=='ringCenterY': self.embRingPos = int(float(row[2]))
            if row[0]=='angle': self.embAngle = float(row[2])
            if row[0]=='embryoCenterX':
                try:
                    self.embCenterX, self.embCenterXerr = int(float(row[2])), float(row[3])
                    self.expand=True
                except:
                    self.embCenterX, self.embCenterXerr = getCenterXOldStyle(row)
                    self.expand=False
            if row[0]== 'embryoDiam':
                self.embryoDiam, self.embryoDiamErr = float(row[2]),0.3#float(row[3])
            if row[0]== 'timePoint':
                flag = True
                flagEmbC = False
            if flagEmbC:
                embCenterY.append(float(row[1]))
                embCenterYerr.append(float(row[2]))
            if oldEmbCY:
                embCenterY, embCenterYerr = getCenterYOldStyle(rowPrev, row)
                oldEmbCY = False
            if row[0]=='embryoCenterY':
                oldEmbCY = True
                rowPrev=row
            if row[0]== 'timePint,': flagEmbC = True
            if row[0]== 'nSlides': self.nSlices = int(row[2])
            if flag:
                try:
                    ringIndex.append(int(row[0]))
                    centerX.append(float(row[1]))
                    centerXerr.append(float(row[2]))
                    centerY.append(float(row[3]))
                    centerYerr.append(float(row[4]))
                    radius.append(float(row[5]))
                    radiuserr.append(float(row[6]))
                    dist.append(np.sqrt(centerX[-1]**2+centerY[-1]**2))
                    disterr.append((abs(centerX[-1]*centerXerr[-1])+abs(centerY[-1]*centerYerr[-1]))/dist[-1])
                except:
                    pass
        self.embCenterY = np.array(embCenterY)
        self.embCenterYerr = np.array(embCenterYerr)
        self.ringIndex = np.array(ringIndex) #+1 because flow is imaged before ring and -2 because flow starts at 0s
        self.centerX = np.array(centerX)
        self.centerY = np.array(centerY)
        self.radius = np.array(radius)
        self.centerXerr = np.array(centerXerr)
        self.centerYerr = np.array(centerYerr)
        self.radiuserr = np.array(radiuserr)
        self.dist = np.array(dist)
        self.disterr = np.array(disterr)
        self.getAngle()
    
    def loadRingImages(self):
        import Image
        self.nSlices = 15
        self.zPixelScale = 10
        fileName = self.file[:-3]+'tif'
        im = Image.open(fileName)
        im.seek((self.ringIndex[0])*self.nSlices)
        dataList = []
        imList = []
        try:
            while True:
                if np.max(np.asarray(im))>0:
                    dataList.append(a16a8(np.asarray(im.rotate(self.embAngle, expand=self.expand))))
                    imList.append(pil16pil8(im.rotate(self.embAngle, expand=self.expand)))
                else:
                    dataList.append(dataList[-1])
                    imList.append(imList[-1])
                im.seek(im.tell()+1)
        except EOFError:
            pass # end of sequence
        del im
        if len(imList)%self.nSlices>0: imList = imList[:-(len(imList)%self.nSlices)]
        if len(dataList)%self.nSlices>0: dataList = dataList[:-(len(dataList)%self.nSlices)]
        self.dataList = dataList
    
    def rescaleTimeLinear(self):
        ''' determines rate and time zero from fit to an average ring size curve and rescales time '''
        if self.radius.size<=3: index = np.arange(self.radius.size)
        else:
            index = getSubIndex(self.radius[::-1], self.rLinMin, self.rLinMax)
            index = self.radius.size - index -1
            if index.size<3: index=np.arange(3)
        slope, inter, se, ie = fitLine(self.time[index], self.radius[index])
        self.zeroTime = (1. - inter)/slope
        self.tck = -1./slope
        self.halfTime = (0.5 - inter)/slope
        self.tau = -np.log(2.*self.aveR0)/2./slope
        self.fitR0 = self.aveR0
# 
    def rescaleTimeAvgFit(self):
        def getR(t, err=False):
            ''' the data is from Khaliullin paper
                ave tck=200 +/- 30 s'''
            time=np.arange(-0.2,1.2,0.02)
            Rexp = np.fromfile('/home/renat/Documents/work/python/FlowCombine/ringRadiusExp.dat', dtype=np.float32)[2:]
            RexpErr=Rexp[Rexp.size/2:]
            Rexp=Rexp[:Rexp.size/2]
            if err: return np.interp(t, time, Rexp, 1), np.interp(t, time, RexpErr)
            return np.interp(t, time, Rexp, 1)
        
        def resid(params):
            tck, t0 = params
            t = (self.time - t0)/tck
            r = getR(t)
            return (r - self.radius)#/self.radiuserr
        self.rescaleTimeLinear()
        pars, ier = leastsq(resid, (self.tck, self.zeroTime))
        self.tck, self.zeroTime = pars
        self.halfTime = self.zeroTime+0.5*self.tck
        self.tau = np.log(2.*self.aveR0)/2.*self.tck
        self.Rfit=1
        
    def rescaleTimeFuncFit(self, func):
        def resid(params):
            tck, t0 = params
            t = (self.time - t0)/tck
            r = func(t)
            return (r - self.radius)#/self.radiuserr
        self.rescaleTimeLinear()
        pars, ier = leastsq(resid, (self.tck, self.zeroTime))
        self.tck, self.zeroTime = pars
        self.halfTime = self.zeroTime+0.5*self.tck
        self.tau = np.log(2.*self.aveR0)/2.*self.tck
        self.Rfit=1
    
    def rescaleTimeModel(self):
        self.Rfit = fitRingSizeHalf(self.time, self.radius, None)
        if self.Rfit!=None:
            self.fitR0=self.Rfit.best_values['r0']
            self.tau = self.Rfit.best_values['tau']
            self.halfTime = self.Rfit.best_values['t0']
            self.tck = 2.*self.tau/np.log(2.*self.aveR0)
            self.zeroTime = self.halfTime-0.5*self.tck
        else:
            self.tau, self.halfTime, self.tck, self.zeroTime  = None, None
 
    def getNormT(self, t=None):
        if t==None: t=self.time
        if self.timeScaling==0 or self.timeScaling==1: return (t-self.halfTime)/self.tau
        else: return (t-self.zeroTime)/self.tck
    
    def getRealT(self, t=None):
        if t==None: t=self.time
        return (t-self.zeroTime)
    
    def getNormR(self):
        return self.fitR0*np.power(self.radius/self.fitR0,np.log(2.*self.aveR0)/np.log(2.*self.fitR0))
    
    def getNormT_Ret(self):
        tau, t0 = self.RfitRet.best_values['tau'], self.RfitRet.best_values['t0']
        return (self.time-t0)/tau
    
    def getNormR_Ret(self):
        tau, d, b = self.RfitRet.best_values['tau'], self.RfitRet.best_values['d'], self.RfitRet.best_values['b']
        return ((np.power(self.radius*2.,d)-1)/d/b/tau+1)/2.
    
    def getAngle(self):
        ''' calculates angle (direction) of the ring movement in the division plane '''
        x = np.concatenate(([0], self.centerX))
        y = np.concatenate(([0], self.centerY))
        xe = np.concatenate(([0.01], self.centerXerr)) #the (0,0) center errorbar is calculated from embryo center estimation errorbars/embryo radius
        ye = np.concatenate(([0.01], self.centerYerr))
        index = np.concatenate(([0],np.where(self.radius>=self.ringSizeThresh)[0]+1))
        self.angle = getAngle(x, y, xe, ye, index)
        self.angle+=np.pi/2 #adding pi/2 because angle is calculated between ring direction and x-axis, but imaging plane is 90 degree with x-axis
        if self.angle>=2*np.pi: self.angle-=2*np.pi
        if self.angle<0: self.angle+=2*np.pi
    
    def getRadius(self, time):
        ''' returns interpolated ring radius at normalized time '''
        if np.min(np.abs(time-self.time))<0.01: return self.radius[np.argmin(np.abs(time-self.time))]
        else: return np.interp(time, self.time, self.radius)
    
    def getRadiusErr(self, time):
        ''' returns interpolated error of the ring radius at time '''
        return np.interp(time, self.time, self.radiuserr)

    def getDist(self, radius):
        ''' returns interpolated ring distance from the center of the embryo at time '''
        return np.interp(radius, self.radius[::-1], self.dist[::-1], left=0, right=0)
 
    def getDistErr(self, radius):
        ''' returns interpolated error of the ring distance from the center of the embryo at time '''
        return np.interp(radius, self.radius[::-1], self.disterr[::-1])
    
    def getCenterRot(self, radius, angle):
        cx = self.getCenterX(radius)
        cy = self.getCenterY(radius)
        ce = np.sqrt(self.getCenterXErr(radius)**2+self.getCenterYErr(radius)**2)
        xNew, yNew = rotate(cx, cy, angle)
        return xNew, yNew, ce
    
    def getCenterX(self, radius):
        ''' interpolates position of the ring center for given ring size '''
        return np.interp(radius, self.radius[::-1], self.centerX[::-1])
    
    def getCenterY(self, radius):
        return np.interp(radius, self.radius[::-1], self.centerY[::-1])
    
    def getCenterXErr(self, radius):
        return np.interp(radius, self.radius[::-1], self.centerXerr[::-1])
    
    def getCenterYErr(self, radius):
        return np.interp(radius, self.radius[::-1], self.centerYerr[::-1])

if __name__ == '__main__':
    pass
    