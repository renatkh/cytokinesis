'''
Created on Aug 6, 2015
@author: renat
'''
from RingClass import RingClass
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate as rotateArr
from myFigure import myFigure
from myMath import rotate
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import matplotlib, os
import numpy as np
from myFunc import argclose

class divPlaneClass(object):
    '''
    class to analyze and visualize the division plane
    '''
    
    def __init__(self, fileName, timeStep, nZ, dZ, pixSize, da, drS, drL, loadIms=True, rotate=True, reverse=False, maxProject=False):
        self.nZ = 30#30
        self.dZ = 4.#4 #pixels between z
        self.pixelSize = 0.25 #in microns
        self.da = 20. # averading angle
        self.drS = 0.1 #averaging distance inside the ring
        self.drL = 0.3 #averaging distance outside the ring
        
        self.drStep = 0.03 #step size for division profile
        self.timeStep = timeStep #s between z stacks
        self.reverse = reverse
        self.cutZFlag=False
        self.useSavedInt = True
        self.blur = False
        self.MaxIintProj=maxProject
        self.angles = np.arange(0.,180.+self.da,self.da)
        self.folder = '/'.join(fileName.split('/')[:-1])
        if self.MaxIintProj: projection = '_maxP'
        else: projection = '_sumP'
        self.intsFile = fileName[:-4]+'{0}{1}'.format(projection,['','_blur'][self.blur])+'.pkl'
        self.BG=None
        self.ring = RingClass(fileName, self.timeStep)
        self.ring.loadRing()
        self.embTime=np.arange(self.ring.embCenterY.size)*self.timeStep
        self.ring.time=self.embTime[self.ring.ringIndex]
#         self.ring.rescaleTime(scaling=3)#FIXME: check which scaling to use
        self.ring.rescaleTime(scaling=0)#FIXME: check which scaling to use
        if not rotate: self.ring.angle=0
        self.intNorm=1.
        self.base=0
        self.imFile = fileName[:-3]+'tif'
        self.imLoaded=False
        self.avgInts, self.avgIntsErr, self.pointsN = -np.ones((3,self.embTime.size,self.angles.size))
        if loadIms:
            self.intNorm=1.
            self.generateInts()
        else:
            self.intNorm=1.
    
    def generateInts(self):
        print('generating intensities')
        self.loadIm()
        for slide in self.ring.ringIndex:
            print('slide=',slide)
            self.getRingInt(slide, self.angles)
            
    def correction(self,z):
        if self.reverse: return np.exp(-(self.nZ-z-1.)/15.)
        else: return np.exp(-z/15.)
    
    def loadIm(self):
        self.imLoaded=True
        im = Image.open(self.imFile)
        self.im = []
        try:
            while True:
                imt = np.asarray(im.rotate(self.ring.embAngle, expand=1), dtype=np.float32)
                if self.blur: imt = gaussian_filter(imt, 3)
                self.im.append(imt)
                im.seek(im.tell()+1)
        except EOFError:
            pass # end of sequence
        del im
        self.im = np.array(self.im).reshape(self.embTime.size, self.nZ, self.im[0].shape[0], self.im[0].shape[1])
        if self.cutZFlag: self.cutZ(25, i0=0)
        self.imCenterNew = np.array(self.im[0,0].shape)[::-1]/2
        self.setImCenters()
    
    def cutZ(self, nZ, i0=0):
        if nZ>self.nZ: nZ=self.nZ
        self.im = self.im[:,i0:i0+nZ,:,:]
        self.nZ=nZ
    
    def getNormT(self):
        if self.ring.Rfit!=None: return self.ring.getNormT(self.embTime)
        else: return self.getRealT()
    
    def getRealT(self):
        ''' returns time in seconds from zero time of CK '''
        return self.ring.getRealT(self.embTime)
            
    def setImCenters(self):
        self.imCenter = (self.im[0,0].shape[1]/2,int(self.nZ*self.dZ)/2)
        self.imCenterNew = np.array(self.getDivPlaneRot(0).shape)[::-1]/2
    
    def getEmbCenterYOrg(self, t):
        ''' interpolates position of the embryo center in Y for given time'''
        return np.interp(t, self.embTime, self.ring.embCenterY)
    
    def getEmbCenter(self, t):
        ''' calculates position of the embryo center for given time point after rotational alignment '''
        cx, cz = rotate(self.ring.embCenterX-self.imCenter[0],
                        self.getEmbCenterYOrg(t)-self.imCenter[1], -self.ring.angle)
        return (self.imCenterNew[0]+cx, self.imCenterNew[1]+cz)
    
    def getRingCenter(self, t):
        r=self.ring.getRadius(t)
        if np.min(np.abs(self.ring.radius-r))<0.01:
            slide=np.argmin(np.abs(self.ring.radius-r))
            cx, cz = rotate(self.ring.centerX[slide], self.ring.centerY[slide], -self.ring.angle)
        else: cx, cz = rotate(self.ring.getCenterX(r), self.ring.getCenterY(r), -self.ring.angle)
        ecx, ecz = self.getEmbCenter(t)
        return (ecx+self.ring.embryoDiam/2*cx, ecz+self.ring.embryoDiam/2*cz)
    
    def getBG(self,slide, z):
        ''' use cytoplasmic BG next to the furrow '''
        dy, w, h = 20, 50, 50
        im = self.im[slide,z]
        BG = np.mean(im[self.ring.embRingPos-dy-h:self.ring.embRingPos-dy,self.ring.embCenterX-w/2:self.ring.embCenterX+w/2])/2.
        BG += np.mean(im[self.ring.embRingPos+dy-h:self.ring.embRingPos+dy,self.ring.embCenterX-w/2:self.ring.embCenterX+w/2])/2.
        return BG
    
    def getDivPlane(self, slide):
        zStack = self.im[slide]
        z = np.arange(self.nZ)*self.dZ
        x = np.arange(zStack[0].shape[1])
        intensity = np.array([])
        for i in range(len(zStack)):
            im = zStack[i]
            bg = self.getBG(slide, i)
            imtmp=im-bg
            imtmp[np.where(im<bg)]=0

            dy = int(self.drL*self.ring.embryoDiam) #width of the projection line in each direction (real width is double)
            if self.MaxIintProj: prof = 1.*np.max(imtmp[self.ring.embRingPos-dy:self.ring.embRingPos+dy,:], axis = 0)/self.correction(i*self.dZ/4.)
            else: prof = 1.*np.sum(imtmp[self.ring.embRingPos-dy:self.ring.embRingPos+dy,:], axis = 0)/self.correction(i*self.dZ/4.)
            prof[np.where(prof<0)]=0
            intensity = np.concatenate((intensity, prof))
        intensity = np.transpose(intensity.reshape(z.size, x.size))
        divP = np.zeros((x.size, int(self.nZ*self.dZ)))
        for i in range(divP.shape[0]):
            divP[i] = np.interp(np.arange(int(self.nZ*self.dZ)), z, intensity[i,:])
        return np.transpose(divP)
    
    def getDivPlaneRot(self, slide):
        divP = self.getDivPlane(slide)
        return rotateArr(divP, self.ring.angle*180/np.pi)
    
    def getRingLength(self,slide):
        ''' returns the length of the ring captured in the division plane '''
        t=self.self.embTime[slide]
        r = self.ring.getRadius(t)
        rcz = self.getEmbCenterYOrg(t)+self.ring.getCenterY(r)*self.ring.embryoDiam/2
        rz0, rz1 = 0, self.nZ*self.dZ
        cosa0 = (rz0-rcz)/(r*self.ring.embryoDiam/2)
        cosa1 = (rz1-rcz)/(r*self.ring.embryoDiam/2)
        if -1<cosa0:a0=np.arccos(cosa0)
        else: a0=np.pi
        if cosa1<1:a1=np.arccos(cosa1)
        else: a1=0
        return 2*r*self.ring.embryoDiam/2*(a0-a1)
    
    def getIntNorm(self):
        angles = np.arange(0,180+self.da,self.da)
        ints = np.array([])
        for slide in range(self.ring.ringIndex.size):
            if -1.5<=self.getNormT()[slide]<=-0.7:
                avgInts, avgIntsErr, pointsN = self.getRingInt(slide, angles)
                ints = np.concatenate((ints,avgInts[np.where(avgInts>0)]))
        if ints.size>0: self.intNorm = np.mean(ints)
        else: print('cannot normalize intensity of '+self.ring.label)
    
    def getSubRing(self, slide, ang, da, r0=None, r1=None, side=0):
        ''' returns set of points on the path of the ring. angles in degree from 0 to 180 in ring coordinates (bottom 0 top 180)'''       
        t=self.embTime[slide]
        r = self.ring.getRadius(t)
        if r0==None: r0 = self.ring.embryoDiam/2.*max(0,r-self.drS)
        else: r0*=self.ring.embryoDiam/2
        if r1==None: r1 = self.ring.embryoDiam/2.*(r+self.drL)
        else: r1 *= self.ring.embryoDiam/2.
        if r0<0:r0=0.
        rx, rz = self.getRingCenter(t)
        da*=np.pi/180
        ang=ang*np.pi/180
        def checkAngPos(x,y,a):
            angt = np.arctan2(y,x)
            angt+=np.pi/2
            angt[np.where(angt>np.pi)]-=2*np.pi
            angt[np.where(x**2+y**2<2**2)]=a #small circles have undefined angular position, so include all
            return (a-da/2<angt)&(angt<a+da/2)
        def checkAngNeg(x,y,a):
            angt = np.arctan2(y,x)
            angt+=np.pi/2
            angt[np.where(angt>np.pi)]-=2*np.pi
            angt[np.where(x**2+y**2<2**2)]=a #small circles have undefined angular position, so include all
            return (a-da/2<-angt)&(-angt<a+da/2)
        
        divP = self.getDivPlaneRot(slide)
        divPFlag = np.ones((int(self.nZ*self.dZ),self.im[0,0].shape[1]))
        divPFlag[0,:]=0
        divPFlag[-1,:]=0
        divPFlag = rotateArr(divPFlag, self.ring.angle*180/np.pi)
        subRing = np.zeros_like(divP)
        z,x = np.where(divP>=0.1)
        ecx, ecz = self.getEmbCenter(t)
        xt = x-ecx
        zt = z-ecz
        rEmb=self.ring.embryoDiam/2
        indsEmb=np.where(2.*rEmb**2>xt**2+zt**2)[0]
        xt = x[indsEmb]-rx
        zt = z[indsEmb]-rz
        inds=np.where((r0**2<xt**2+zt**2)&(xt**2+zt**2<r1**2))[0]
        if side>=0:indsAp = np.where(checkAngPos(xt[inds],zt[inds],ang))
        if side<=0:indsAn = np.where(checkAngNeg(xt[inds],zt[inds],ang))
        ns=0
        if inds.size==0: ns=1
        
        if side>=0 and indsAp[0].size>0 and all(divPFlag[z[indsEmb][inds][indsAp],x[indsEmb][inds][indsAp]]>0.1):
            subRing[z[indsEmb][inds][indsAp],x[indsEmb][inds][indsAp]]=divP[z[indsEmb][inds][indsAp],x[indsEmb][inds][indsAp]]
            ns +=1 
        if side<=0 and indsAn[0].size>0 and all(divPFlag[z[indsEmb][inds][indsAn],x[indsEmb][inds][indsAn]]>0.1):
            subRing[z[indsEmb][inds][indsAn],x[indsEmb][inds][indsAn]]=divP[z[indsEmb][inds][indsAn],x[indsEmb][inds][indsAn]]
            ns +=1 
        return subRing, ns
    
    def getInt(self, slide, angles, r0, r1, side):
        angs = np.array(angles, dtype=np.float)
        if angs.size<=1:
            angs=np.array([angs])
            da=self.da
        else: da=angs[1]-angs[0]
        avgInts = np.zeros_like(angs)
        avgIntsErr = np.zeros_like(angs)
        pointsN = np.zeros_like(angs)
        if r0<0: r0=0.
        for i in range(angs.size):
            subRing, pointsN[i]=self.getSubRing(slide, angs[i], da, r0, r1, side)
            if np.sum(subRing>0)>0:
                avgInts[i] = np.sum(subRing)/np.sum(subRing>0)
                avgIntsErr[i] = np.std(subRing[np.where(subRing>0)])/np.sqrt(np.sum(subRing>0))
        if self.intNorm!=None:
            avgInts=(avgInts-self.base)/self.intNorm
            avgIntsErr/=self.intNorm
        return avgInts, avgIntsErr, pointsN
    
    def getSubRingConc(self, slide):
        t=self.embTime[slide]
        r = self.ring.getRadius(t)
        r0 = self.ring.embryoDiam/2*max(0,r-self.drS)
        r1 = self.ring.embryoDiam/2*min(1, r+self.drL)
        rx, rz = self.getRingCenter(t)
        divP = self.getDivPlaneRot(slide)
        subRing = np.zeros_like(divP)
        z,x = np.where(divP>0)
        xt = x-rx
        zt = z-rz
        inds=np.where((r0**2<xt**2+zt**2)&(xt**2+zt**2<r1**2))[0]
        subRing[z[inds],x[inds]]=divP[z[inds],x[inds]]
        conc=np.sum(subRing)/self.getRingLength(slide)
        if self.intNorm!=None:return (conc-self.base)/self.intNorm
        else: return conc
    
    def getRingInt(self, slide, angles):
        if all(angles==self.angles) and np.max(self.avgInts[slide])!=-1:
            return self.avgInts[slide], self.avgIntsErr[slide], self.pointsN[slide]
        if not self.imLoaded: self.loadIm()
        print('generating new data', self.intsFile)
        angs = np.array(angles, dtype=np.float)
        if angs.size<=1:
            angs=np.array([angs])
            da=self.da
        else: da=angs[1]-angs[0]
        avgInts = np.zeros_like(angs)
        avgIntsErr = np.zeros_like(angs)
        pointsN = np.zeros_like(angs)
        rave=self.ring.embryoDiam/2*self.ring.getRadius(self.embTime[slide])
        r=np.arange(-self.drS,self.drL+0.001,self.drStep)+self.ring.getRadius(self.embTime[slide])
        for i in range(angs.size):
            subRing, pointsN[i]=self.getSubRing(slide, angs[i], da, r[0], r[-1]) #Note pointsN is number of sides
            if np.sum(subRing>0)>0:
                avgInts[i] = np.sum(subRing)/rave/(da/180.*np.pi)/pointsN[i]
                avgIntsErr[i] = np.std(subRing[np.where(subRing>0)])/np.sqrt(rave*(da/180.*np.pi)*2.*pointsN[i])
        avgInts[np.where(angs>=180-da/4.)]*=2
        avgIntsErr[np.where(angs>=180-da/4.)]*=np.sqrt(2)
        avgInts[np.where(angs<=da/4.)]*=2
        avgIntsErr[np.where(angs<=da/4.)]*=np.sqrt(2)
        
        if self.intNorm!=None:
            avgInts=(avgInts-self.base)/self.intNorm
            avgIntsErr/=self.intNorm
        
        self.avgInts[slide] = avgInts
        self.avgIntsErr[slide] = avgIntsErr
        self.pointsN[slide] = pointsN
        if self.ring.Rfit!=None: return avgInts, avgIntsErr, pointsN
        else: return avgInts, avgIntsErr, pointsN
    
    def getDivPProf(self, slide, angles,side):
        rr=self.ring.getRadius(self.embTime[slide])
        r=np.arange(-self.drS,self.drL+0.001,self.drStep)+rr
        prof=np.zeros((angles.size,r.size-1))
        profErr=np.zeros((angles.size,r.size-1))
        profN=np.zeros((angles.size,r.size-1))
        for i in range(prof.shape[1]):
            r0,r1=r[i:i+2]
            if r1>=0:
                r0=max(0.,r0)
                prof[:,i], profErr[:,i], profN[:,i] = self.getInt(slide, angles, r0, r1, side)
                if r1>1. and any(profN[:,i])==0:
                    profN[:,i][np.where(profN[:,i]==0)]=1
            else: profN[:,i]=1
        return r[:-1]+0.5*self.drStep-rr, prof, profErr, profN
    
    def getRingConc(self):
        return np.array([self.getSubRingConc(slide) for slide in self.ring.ringIndex])
    
    def showRing(self, slide, fig=None):
        if fig==None:
            fig = myFigure()
            fig.imshow(self.getDivPlaneRot(slide))
        t = self.embTime[slide]
        angles = np.arange(0,np.pi*2+0.1,0.05)
        ecx, ecz = self.getEmbCenter(t)
        embX = ecx+self.ring.embryoDiam/2*np.sin(angles)
        embZ = ecz+self.ring.embryoDiam/2*np.cos(angles)
        fig.plot(embX, embZ, color='k')
        fig.xlim((min(embX)-10,max(embX)+10))
        fig.ylim((min(embZ)-10,max(embZ)+10))
        xticks = ecx+self.ring.embryoDiam/2*np.arange(-1,1.1,0.5)
        yticks = ecz+self.ring.embryoDiam/2*np.arange(-1,1.1,0.5)
        fig.xticks(xticks, ['{0}'.format(i) for i in np.arange(-1,1.1,0.5)])
        fig.yticks(yticks, ['{0}'.format(i) for i in np.arange(-1,1.1,0.5)])
        if slide in self.ring.ringIndex:
            rcx, rcy = self.getRingCenter(t)
            ringX = rcx+self.ring.embryoDiam/2*self.ring.getRadius(t)*np.sin(angles)
            ringZ = rcy+self.ring.embryoDiam/2*self.ring.getRadius(t)*np.cos(angles)
            r0 = max(0,self.ring.getRadius(t)-self.drS)
            r1 = min(2,self.ring.getRadius(t)+self.drL)
            ring0X = rcx+self.ring.embryoDiam/2*r0*np.sin(angles)
            ring0Z = rcy+self.ring.embryoDiam/2*r0*np.cos(angles)
            ring1X = rcx+self.ring.embryoDiam/2*r1*np.sin(angles)
            ring1Z = rcy+self.ring.embryoDiam/2*r1*np.cos(angles)
            fig.plot(ringX, ringZ, color='r')
            fig.plot(ring0X, ring0Z, color='r', ls='--')
            fig.plot(ring1X, ring1Z, color='r', ls='--')
        return fig
    
    def showSubRing(self, slide, angle=None, da=None, r0=None, r1=None):
        if angle!=None: 
            if da==None: da=self.da
        else:
            da=180
            angle=0.
        subRing =self.getSubRing(slide, angle, da, r0, r1)[0]
        fig=myFigure()
        fig.imshow(subRing)
        fig = self.showRing(slide, fig)
        return fig
    
    def showRingIntProfs(self, slide=None):
        da=self.da
        angles = np.arange(0,180+da,da)
        fig = myFigure()
        fig2=myFigure()
        for slide in self.ring.ringIndex:
            r = self.ring.getRadius(self.embTime[slide])
            if r>0.2:
                avgInts,avgIntsErr, pointsN = self.getRingInt(slide, angles)
                inds = np.where(pointsN>0)
                fig.errorbar(angles[inds], avgInts[inds],avgIntsErr[inds], label='R={0:.2f}'.format(r))
                fig2.plot(angles[inds], avgInts[inds]*r, label='R={0:.2f}'.format(r))
            else: break
        fig.legend(2)
        fig.title(self.ring.label)
        fig2.legend(2)
        fig2.title('total '+self.ring.label)
        return fig, fig2

    def showDivPProf(self, slide, angle, fig=None):
        if fig==None: fig = myFigure()
        x, prof, profErr, profN = self.getDivPProf(slide, angle)
        fig.errorbar(x,prof[0],profErr[0])
        fig.legend(2)
        fig.title(self.ring.label)
        return fig
    
    def showRingConc(self, fig=None):
        ints=self.getRingConc()
        if fig==None:
            fig=myFigure()
            fig.title(self.ring.label)
        else: fig.legend(2)
        fig.plot(self.embTime, ints, label=self.ring.label)
        return fig
    
    def showDivPlane(self, slide, fig=None):
        if not self.imLoaded: self.loadIm()
        if fig==None:
            fig=myFigure()
        else: fig.legend(2)
        fig.imshow(self.getDivPlaneRot(slide), bw=False, colorbar=False)
        delta = 10
        ecx, ecz = self.getEmbCenter(self.embTime[slide])
        x0 = ecx-self.ring.embryoDiam/2+delta
        z0 = ecz+self.ring.embryoDiam/2-delta
        fig.plot((x0,x0+int(10./self.pixelSize)),(z0,z0), color='k')
        fig=self.showRing(slide, fig)
        return fig
    
    def getEmbCrop(self, slide, z, dx,dy):
        x = self.ring.embRingPos
        y = self.ring.embCenterX
        return np.transpose(self.im[slide,z,x-dx/2:x+dx/2,y-dy/2:y+dy/2])
    
    def getKymo(self):
        if not self.imLoaded: self.loadIm()
        dx=60
        z=self.nZ/2
        dy=170
        i=0
        timeLabel = []
        kymo = np.zeros((dy,dx*(self.ring.ringIndex.size)))
        for slide in self.ring.ringIndex:
            kymo[:,i*dx:(i+1)*dx]=self.getEmbCrop(slide, z, dx,dy)
            timeLabel.append([(i+0.1)*dx,-10, '{0}'.format(int(self.ring.getRealT()[i]))])
            i+=1
        return kymo, timeLabel
    
def report(emb):
    da=20
    angles = np.arange(0,180+da,da)
    print('embryo: tau={0}s, angle(degree)={1}'.format(emb.ring.tau, emb.ring.angle*180/np.pi))
    print('angles (degree): {}'.format(angles))
    for ind in emb.ring.ringIndex:
        ints = emb.getRingInt(ind, angles)[0]
        print('Intensity Values for Ring #{0}'.format(ind))
        print('{}\n'.format(ints))
        
def checkRingFits(emb):
    ''' displays ring size change over time and the division planes to check ring fits'''
    fig = myFigure()
    fig.plot(emb.ring.getNormT(), emb.ring.getNormR())
    for slide in emb.ring.ringIndex[:]:
        emb.showDivPlane(slide)
#     plt.show()

def showKymo(emb):
    ''' displays kymograph for a given embryo '''
    kymo, timeLabel = emb.getKymo()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.axis('off')
    im = ax.imshow(kymo,cmap='Greys_r')
    for j in range(len(timeLabel)):
        x,y,string = timeLabel[j]
        ax.text(x,y, string, fontdict={'color':'k','size':12})
    fig.colorbar(im)

def getEmbCortFluor(emb):
    z=15
    slide = emb.ring.ringIndex[argclose(0.8,emb.ring.radius)]
    emb.loadIm()
    im = emb.im[slide, z]
    bg = emb.getBG(slide, z)
    im = im.astype(np.float)-bg
    im *= (im>0).astype(np.float)
    print '{1} embryo cortical fluorescence = {0}'.format(np.sum(im), emb.ring.label)
    return np.sum(im)

if __name__ == '__main__':
    fileName = '01Ring.csv' #full path to the input file
    dts = 30 #time between time points
    nZ = 30 #number of Z planes
    dZ = 4. #pixels between z
    pixelSize = 0.25 #pixel size in microns
    da = 20. #averaging angle in degrees (angle step size)
    drS = 0.1 #averaging distance inside the ring
    drL = 0.3 #averaging distance outside the ring
    emb = divPlaneClass(fileName, dts, nZ, dZ, pixelSize, da, drS, drL, loadIms=True, rotate=True, maxProject = True)
    checkRingFits(emb)
    report(emb)
    showKymo(emb)
    getEmbCortFluor(emb)
    plt.show()
