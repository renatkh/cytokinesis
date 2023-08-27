import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from ImageWindowUI import *
from PIL import Image
from myFunc import PIL2QPixmap

class ImageWindow(QtWidgets.QDialog):
    def __init__(self,fileName,parent=None,imSeq=None, arraySeq=None, surfaceSeq=None):
        # Define all elements necessary for the image manipulations
        self.fileName = fileName
        self.imSeq = []
        self.arraySeq = []
        self.surfaceSeq = []
        self.timer = QtCore.QTimer()
        QtCore.QObject.connect(self.timer, QtCore.SIGNAL("timeout()"), self.playFrameUpdate)
        self.frameRate = 25
        self.playFlag = False 
        self.zoomScale = 1.0
        # Load sequence of images, create arrays and
        # surfaces (difference from arrays in coordinate system [x,y]) of intensities
        if imSeq == None and arraySeq == None and surfaceSeq == None:
            self.imSeq = []
            self.arraySeq = []
            self.surfaceSeq = []
            imTmp = Image.open(self.fileName)
            try:
                while 1:
                    self.imSeq.append(imTmp.point(lambda i: i))
                    imTmp.seek(imTmp.tell()+1)
            except EOFError:
                pass # end of sequence
            for imTmp in self.imSeq:
                self.arraySeq.append(np.asarray(imTmp))
                self.surfaceSeq.append(np.transpose(self.arraySeq[-1][::-1,:]))
        elif imSeq != None:
            self.imSeq = imSeq
            for imTmp in self.imSeq:
                self.arraySeq.append(np.asarray(imTmp))
                self.surfaceSeq.append(np.transpose(self.arraySeq[-1][::-1,:]))
        elif arraySeq != None:
            self.arraySeq = arraySeq
            for arrayTmp in self.arraySeq:
                self.imSeq.append(Image.fromarray(arrayTmp))
                self.surfaceSeq.append(np.transpose(arrayTmp[::-1,:]))
        elif surfaceSeq != None:
            self.surfaceSeq = surfaceSeq
            for surfTmp in self.surfaceSeq:
                self.arraySeq.append(np.transpose(surfTmp[:,::-1]))
                self.imSeq.append(Image.fromarray(self.arraySeq[-1]))
        
        #default values for contrast and brightness
        self.maxPixelValue = 255
        self.thresholdMaximum = self.maxPixelValue
        self.thresholdMinimum = 0
        self.brightness = 50
        self.contrast = 50
        
        self.threshhold = None
        self.playSpeed = 25 #frames/second that the sequence is played
        self.zoom = 1 #scale to the original image size
                
        #create UI with its buttons, fields and connections
        QtGui.QDialog.__init__(self,parent)
        self.ui = Ui_ImageWindow()
        self.ui.setupUi(self) #put all elements on the window
        #set window size same as image
        self.resize(max(self.imSeq[0].size[0]+50,250), self.imSeq[0].size[1]+70)
        #set window title to be fileName
        self.setWindowTitle(QtGui.QApplication.translate("ImageWindow", fileName, None, QtGui.QApplication.UnicodeUTF8))
        #set maximum values of spinBox and slider as a number of images
        self.ui.spinBox.setMaximum(len(self.imSeq)-1)
        self.ui.imageScrollBar.setMaximum(len(self.imSeq)-1)
        #update image when slider moved
        QtCore.QObject.connect(self.ui.imageScrollBar, QtCore.SIGNAL("valueChanged(int)"), self.updateImage)
        #display image
        self.scene = QtGui.QGraphicsScene()
        self.scene.setObjectName('scene')
        self.scene.addPixmap(PIL2QPixmap(self.imSeq[0].resize(np.asarray(np.asarray(imTmp.size)*self.zoomScale, dtype = np.int))))
        self.ui.imageSpace.setScene(self.scene)
        #Play button
        QtCore.QObject.connect(self.ui.playPause, QtCore.SIGNAL("clicked()"), self.playPause)
    # define various functions and operations the image
    def updateImage(self,imNum=None):
        imNum = self.ui.spinBox.value()
        def adjastBrCont(i):
            if i >= self.thresholdMaximum:
                return self.maxPixelValue
            elif i <= self.thresholdMinimum:
                return 0
            else:
                return self.maxPixelValue*(i-self.thresholdMinimum)/(self.thresholdMaximum-self.thresholdMinimum)
        imTmp = self.imSeq[imNum].point(adjastBrCont)
        self.scene = QtGui.QGraphicsScene()
        self.scene.addPixmap(PIL2QPixmap(imTmp.resize(np.asarray(np.asarray(imTmp.size)*self.zoomScale, dtype = np.int))))
        self.ui.imageSpace.setScene(self.scene)
    def playFrameUpdate(self):
            if self.ui.imageScrollBar.value() < len(self.imSeq)-1:
                imNum = self.ui.imageScrollBar.value() + 1
            else:
                imNum = 0
            self.ui.imageScrollBar.setValue(imNum)
            self.timer.start(1000/self.frameRate)
    def mousePressEvent(self, event):
        button = event.button()
        item = self.childAt(event.x(), event.y())
        if self.parent().mouseState == 'zoom' and item.parent() == self.ui.imageSpace:
            if button == 1:
                self.zoomScale = self.zoomScale*1.5
            else:
                self.zoomScale = self.zoomScale/1.5
            self.resize(max(int(self.imSeq[0].size[0]*self.zoomScale)+50,250),
                             int(self.imSeq[0].size[1]*self.zoomScale)+70)
            self.updateImage()
                
        if button == 2 and item == self.ui.playPause:
            class FrameRateDialog(QtGui.QDialog):
                def __init__(self,parent=None):
                    QtGui.QDialog.__init__(self,parent)
                    self.gridLayout = QtGui.QGridLayout(self)
                    self.label = QtGui.QLabel(self)
                    self.label.setText('Frame Rate')
                    self.spinBox = QtGui.QSpinBox(self)
                    self.spinBox.setValue(self.parent().frameRate)
                    QtCore.QObject.connect(self.spinBox, QtCore.SIGNAL("valueChanged(int)"), self.setFrameRate)
                    self.gridLayout.addWidget(self.label)
                    self.gridLayout.addWidget(self.spinBox)
                def setFrameRate(self,rate):
                    self.parent().frameRate = rate
            self.userDialog = FrameRateDialog(self)
            self.userDialog.show()
    def playPause(self):
        if self.playFlag:
            self.timer.stop()
            self.playFlag = False
        else:
            self.timer.start(1000/self.frameRate)
            self.playFlag = True
    def focusOutEvent(self, event):
        pass
    def makeSubArray (self,area):
        pass
#        return ImageObj(array = self.array[max(area.box_ind[0],0):min(area.box_ind[0]+area.size,self.array.shape[0]),\
#                                            max(area.box_ind[1],0):min(area.box_ind[1]+area.size,self.array.shape[1])])
    def makeSubSurf (self,area):
        pass
#        return ImageObj(surf = self.surf[max(area.box_coor[0],0):min(area.box_coor[0]+area.size,self.surf.shape[0]),\
#                                            max(area.box_coor[1],0):min(area.box_coor[1]+area.size,self.surf.shape[1])])
    def plotSurf3D (self):
        pass
#        x, y = np.meshgrid(np.arange(self.surf.shape[0]),np.arange(self.surf.shape[1]))
#        return mlab.mesh(x,y,self.surf)

class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=10, height=10, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class Histogram(MplCanvas):
    """A canvas that updates itself every second with a new plot."""
    def __init__(self, *args, **kwargs):
        MplCanvas.__init__(self, *args, **kwargs)
        self.updateHisto()

    def updateHisto(self):
        maxHistValue = 255
        average = 100 - self.parent().parent().ui.sliderBrightness.value()
        contrast = self.parent().parent().ui.sliderContrast.value()
        slope = -1
        offset = 100
        hDiff = int(slope*contrast + offset)
        maximum = (average + hDiff)*maxHistValue/100
        minimum = (average - hDiff)*maxHistValue/100
        self.axes.hold(False)
        if self.parent().parent().parent().lastImFocus != None:
            imNum = self.parent().parent().parent().lastImFocus.ui.spinBox.value()
            data = self.parent().parent().parent().lastImFocus.imSeq[imNum].array.ravel()
            histList, bins = np.histogram(data, 100)
            self.axes.hist(data, 100, normed=False, facecolor='gray')
            self.axes.hold(True) 
            maxBin = np.max(histList)
        else:
            maxBin = 1.0
        x = np.arange(maxHistValue)
        brConLine = maxBin/(maximum-minimum)*(x - minimum)
        self.axes.plot(x, brConLine, 'k-')
        # update the view limits
        self.axes.set_xlim(0, maxHistValue)
        self.axes.set_ylim(0, maxBin)
        self.axes.set_xticks([0,maxHistValue])
        self.axes.set_yticks([])
        self.draw()