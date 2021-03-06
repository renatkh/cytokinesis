# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ImageWindowUI.ui'
#
# Created: Mon Jan  2 18:31:10 2012
#      by: PyQt4 UI code generator 4.8.5
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_ImageWindow(object):
    def setupUi(self, ImageWindow):
        ImageWindow.setObjectName(_fromUtf8("ImageWindow"))
        ImageWindow.resize(585, 387)
        ImageWindow.setFocusPolicy(QtCore.Qt.StrongFocus)
        ImageWindow.setWindowTitle(QtGui.QApplication.translate("ImageWindow", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.gridLayout = QtGui.QGridLayout(ImageWindow)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.spinBox = QtGui.QSpinBox(ImageWindow)
        self.spinBox.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.spinBox.setMaximum(15)
        self.spinBox.setObjectName(_fromUtf8("spinBox"))
        self.gridLayout.addWidget(self.spinBox, 2, 0, 1, 1)
        self.playPause = QtGui.QToolButton(ImageWindow)
        self.playPause.setFocusPolicy(QtCore.Qt.NoFocus)
        self.playPause.setText(QtGui.QApplication.translate("ImageWindow", ">", None, QtGui.QApplication.UnicodeUTF8))
        self.playPause.setObjectName(_fromUtf8("playPause"))
        self.gridLayout.addWidget(self.playPause, 2, 2, 1, 1)
        self.imageSpace = QtGui.QGraphicsView(ImageWindow)
        self.imageSpace.setFocusPolicy(QtCore.Qt.NoFocus)
        self.imageSpace.setObjectName(_fromUtf8("imageSpace"))
        self.gridLayout.addWidget(self.imageSpace, 1, 0, 1, 3)
        self.imageScrollBar = QtGui.QScrollBar(ImageWindow)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.imageScrollBar.sizePolicy().hasHeightForWidth())
        self.imageScrollBar.setSizePolicy(sizePolicy)
        self.imageScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.imageScrollBar.setObjectName(_fromUtf8("imageScrollBar"))
        self.gridLayout.addWidget(self.imageScrollBar, 2, 1, 1, 1)

        self.retranslateUi(ImageWindow)
        QtCore.QObject.connect(self.playPause, QtCore.SIGNAL(_fromUtf8("clicked()")), self.playPause.toggle)
        QtCore.QObject.connect(self.spinBox, QtCore.SIGNAL(_fromUtf8("valueChanged(int)")), self.imageScrollBar.setValue)
        QtCore.QObject.connect(self.imageScrollBar, QtCore.SIGNAL(_fromUtf8("valueChanged(int)")), self.spinBox.setValue)
        QtCore.QMetaObject.connectSlotsByName(ImageWindow)

    def retranslateUi(self, ImageWindow):
        pass

