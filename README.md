# README #
This application consists of two programs.

1) FindRing.py:
* finds a contractile ring in a z-stack time series. This generates a csv file containing positions and the size of the ring at each time point.

2) divPlaneClass.py (requires the csv file generated by FindRing.py): 
* to validate ring position and size this program generates reconstructed division planes and draws contractile rings detected by FindRing;
* quantifies the fluorescence intensities along the contractile ring perimeter within arcs of a specified angular range;
* generates a kymograph of the division plane from the central z plane;
* calculates cortical fluorescence from the central plane;

* Version 1.0

SETUP:
* Install python with all dependencies: opencv, scipy, numpy, pillow, matplotlib, pyqt4, lmfit
* FindRing.py parameters (change at the bottom of the file):
   -series = '01' #prefix in the file name, i.e. *series*Ring.tif (input images)
   -timePointStart = 1  #slide number to start ring detection. Make sure that intensity in the equatorial band is higher than at the surrounding cortex
   -flip = False #Check that anterior side is at the top, otherwise change to True
   -folder = '' #folder name with the images
   -filename = folder+series+'Ring.tif' #file name of the images with embryo z stacks
   -nSlices = 30 #number of z slices in a stack
   -zPixelScale = 4. # number of pixels in the camera per 1um (ie 1um = 250nm x 4 pixels)
   -tol = 8 # number of pixels allowed for ring position misalignment
   -chromosomeMarker = False #if chromosome marker is present, the algorithm will try to detect and remove them before processing
   -lateStage=False #if the ring is half way closed or more at the first slide, change to True
   -kernelSize = 5 # the size of the gaussian filter before processing (has to be odd)
   -embryoCenterDrift = 'linear'  #defines how the center of the embryo in z is calculated; 'independent': calculate center for each timepoint independently (use when the depth is manually adjusted through out imaging); 'linear': the center depth is calculated from a linear fit to detected individual centers (use when a steady drift is observed, like imaging on agarose pads); 'median': calculates median position for the center (use only when there is no drift of the embryo center)
     
* divPlaneClass parameters (change at the bottom of the file):
   -fileName = '01Ring.csv' #full path to the input file
   -dts = 30 #time between time points
   -nZ = 30 #number of Z planes
   -dZ = 4. #pixels between z
   -pixelSize = 0.25 #pixel size in microns
   -da = 20. #averaging angle in degrees (angle step size)
   -drS = 0.1 #averaging distance inside the ring
   -drL = 0.3 #averaging distance outside the ring

CONTACT:
* Renat Khaliullin
renatkh at gmail.com