'''
Created on Jul 7, 2014

Library that is responsible for plotting in the same style.
It accepts latex style labels confined in $ sign. 

@author: renat
'''

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
from _codecs_cn import getcodec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import gaussian_kde

class myFigure(object):
    
    def __init__(self, black=False):
        self.labelSize = 24
        self.tickLabelSize = 22
        self.legentFontSize = 8
        self.lineWidth = 2
        self.markerSize = 5
        self.legendFrame = False
        self.legendLine = 2
        self.tickLength = 5
        self.tickWidth = self.lineWidth
        self.tickPad = 10
        self.flag3D = False
        self.marker='o'
        
        rc('axes', linewidth=self.lineWidth)
        rc('font',**{'family':'sans-serif','sans-serif':['Arial'],'size':self.labelSize})
        paramstring=r'\usepackage{bm}'
        matplotlib.rcParams['text.latex.preamble'] = paramstring
        matplotlib.rcParams['svg.fonttype'] = 'none'
        rc('text', usetex=False)
        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.tick_params(axis='both',          # changes apply to the x-axis and y-axis
                    which='both',      # both major and minor ticks are affected
                    direction='out',
                    left='on',      # ticks along the bottom edge are off
                    right='off',         # ticks along the top edge are off
                    bottom='on',
                    top='off',
                    labelleft='on',
                    labelright='off',
                    labeltop='off',
                    labelbottom='on',
                    length = self.tickLength,
                    width = self.tickWidth,
                    pad = self.tickPad,
                    labelsize = self.tickLabelSize)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.colors = ['sienna', 'firebrick', 'orangered', 'gold', 'y', 'green', 'turquoise', 'c', 'dodgerblue', 'b', 'm', 'purple', 'grey']
        self.nPlots=0
    
    def getColor(self):
        color = self.colors[self.nPlots]
        self.nPlots+=1
        if self.nPlots==len(self.colors): self.nPlots=0
        return color
    
    def plot(self, x, y,*args, **kwargs):
        '''
        linestyle '', ' ', 'None', '--', '-.', '-', ':'
        
        marker    description
        .    point
        ,    pixel
        o    circle
        v    triangle_down
        ^    triangle_up
        <    triangle_left
        >    triangle_right
        1    tri_down
        2    tri_up
        3    tri_left
        4    tri_right
        8    octagon
        s    square
        p    pentagon
        *    star
        h    hexagon1
        H    hexagon2
        +    plus
        x    x
        D    diamond
        d    thin_diamond
        |    vline
        _    hline
        TICKLEFT    tickleft
        TICKRIGHT    tickright
        TICKUP    tickup
        TICKDOWN    tickdown
        CARETLEFT    caretleft
        CARETRIGHT    caretright
        CARETUP    caretup
        CARETDOWN    caretdown
        None    nothing
        None    nothing
             nothing
            nothing
        '$...$'    render the string using mathtext.'''
        
        if len(args)==0:
            if 'color' in kwargs:
                color=kwargs['color']
                kwargs.pop('color')
                if color is None: color=self.getColor()
            else:
                color=self.getColor()
            if 'linewidth' in kwargs:
                linewidth = kwargs['linewidth']
                kwargs.pop('linewidth')
            else: linewidth = self.lineWidth
            lines = self.ax.plot(x, y, linewidth=linewidth, markersize=self.markerSize, color=color, *args, **kwargs)
        else: lines = self.ax.plot(x, y, linewidth=self.lineWidth, markersize=self.markerSize, *args, **kwargs)
        return lines[0].get_color()
    
    def scatter(self, x, y, *args, **kwargs):
        dens=False
        if len(args)==0:
            if 'color' in kwargs:
                color=kwargs['color']
                kwargs.pop('color')
            else:
                color=self.getColor()
            if 'density' in kwargs:
                dens = True
                kwargs.pop('density')
            if dens: # Calculate the point density
                xy = np.vstack([x,y])
                z = gaussian_kde(xy)(xy)
#                 scatter = self.ax.scatter(x, y, c=z, s=100, edgecolor='')
                scatter = self.ax.scatter(x, y, s=self.markerSize*5, c=z, edgecolor='', *args, **kwargs)
            else: scatter = self.ax.scatter(x, y, s=self.markerSize*5, color=color, *args, **kwargs)
        else: 
            scatter = self.ax.scatter(x, y, s=self.markerSize*5, *args, **kwargs)
        return None#scatter._facecolors_original
    
    def errorbar(self, x, y, yerr, join=True, *args,**kwargs):
        if len(args)==0:
            if 'color' in kwargs:
                color=kwargs['color']
                kwargs.pop('color')
                if color is None: color=self.getColor()
            else:
                color=self.getColor()
            if 'linewidth' in kwargs:
                linewidth = kwargs['linewidth']
                kwargs.pop('linewidth')
            else: linewidth = self.lineWidth
            if 'fmt' in kwargs:
                fmt=kwargs['fmt']
                kwargs.pop('fmt')
            else: fmt='o'
#         if 'color' in kwargs:
#             color=kwargs['color']
#             kwargs.pop('color')
#         else: color=self.getColor()
        lines = self.ax.errorbar(x, y, yerr=yerr, linewidth=linewidth, markersize=self.markerSize,markeredgecolor = 'none',ecolor=color, color=color, fmt=fmt, *args, **kwargs)
        if join: lines = self.plot(x, y, color=color)
        return lines
    
    def hist(self, x, *args, **kwargs):
        if 'color' in kwargs:
            color=kwargs['color']
            kwargs.pop('color')
        else: color=self.getColor()
        return self.ax.hist(x, *args, linewidth=self.lineWidth, color='k', facecolor=color, **kwargs)
    
    def xlabel(self, label):
        self.ax.set_xlabel(label)
        
    def ylabel(self, label):
        self.ax.set_ylabel(label)
    
    def show00(self):
        self.ax.axhline(linewidth=self.lineWidth, color ='#000000', ls='--')#0,0 axis lines
        self.ax.axvline(linewidth=self.lineWidth, color='#ED1C24', ls='--')
    
    def boxLayout(self):
        self.ax.tick_params(axis='both',          # changes apply to the x-axis and y-axis
                    which='both',      # both major and minor ticks are affected
                    direction='in',
                    left='on',      # ticks along the bottom edge are off
                    right='on',         # ticks along the top edge are off
                    bottom='on',
                    top='on',
                    labelleft='on',
                    labelright='off',
                    labeltop='off',
                    labelbottom='on',
                    length = self.tickLength,
                    width = self.tickWidth,
                    pad = self.tickPad,
                    labelsize = self.tickLabelSize)
        self.ax.spines['right'].set_visible(True)
        self.ax.spines['top'].set_visible(True)
    
    def noAxis(self):
        self.ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        self.ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        self.ax.axis('off')
    
    def legend(self, loc=None):
        ''' set legend locations:
        upper right    1
        upper left    2
        lower left    3
        lower right    4
        right    5
        center left    6
        center right    7
        lower center    8
        upper center    9
        center    10 '''
        if loc==False: self.ax.legend().set_visible(False)
        else: self.ax.legend(fontsize=self.legentFontSize, loc=loc, frameon=self.legendFrame, numpoints=self.legendLine)
        
    def xlim(self, *args, **kwargs):
        self.ax.set_xlim(*args, **kwargs)
    
    def ylim(self, *args, **kwargs):
        self.ax.set_ylim(*args, **kwargs)
    
    def xticks(self, ticks, strings=None):
        if not strings: self.ax.set_xticks(ticks)
        else:
            self.ax.set_xticks(ticks)
            self.ax.set_xticklabels(strings)
        
    def yticks(self, ticks, strings=None):
        if not strings: self.ax.set_yticks(ticks)
        else:
            self.ax.set_yticks(ticks)
            self.ax.set_yticklabels(strings)
    
    def setVerticalLayout(self):
        for tick in self.ax.xaxis.get_major_ticks()+self.ax.yaxis.get_major_ticks():
            tick.label.set_rotation('vertical')
    
    def title(self, title):
        self.fig.suptitle(title)
        
    def text(self, x, y, string, color='black', size=None):
        if size is None: size = self.labelSize
        self.ax.text(x,y, string, fontdict={'color':color,'size':size})
    
    def show(self):
        self.fig.tight_layout()
        plt.show()
        
    def save(self, name='figure.svg'):
        self.fig.tight_layout()
#         self.noClip()
        self.fig.savefig(name, transparant = False, bbox_inches='tight',pad_inches = 0)
    
    def noClip(self):
        for o in self.fig.findobj():
            o.set_clip_on(False)
    
    def close(self):
        plt.close(self.fig)
        
    def loglog(self):
        self.ax.loglog()
    
    def loglin(self):
        self.ax.semilogy()
    
    def imshow(self,im, vmin=None, vmax=None, bw=True, colorbar=True, *args, **kwargs):
        if bw: im = self.ax.imshow(im, vmin=vmin, vmax=vmax, cmap='Greys_r', *args, **kwargs)
        else: im = self.ax.imshow(im, vmin=vmin, vmax=vmax, *args, **kwargs)
        if colorbar: self.fig.colorbar(im)
        
    def rectangle(self,diagPoints, color=None):
        p0, p1 = diagPoints
        w = p1[0]-p0[0]
        h = p1[1]-p0[1]
        if color is None: p = matplotlib.patches.Rectangle((p0[0],p0[1]), w, h, fill=False)
        else: p = matplotlib.patches.Rectangle((p0[0],p0[1]), w, h, facecolor=color)
        self.ax.add_patch(p)

    def scatter3D(self, x, y, z, color=None):
        if color is None: color=self.getColor()
        if not self.flag3D:
            self.noAxis()
            self.ax = self.fig.add_subplot(111, projection='3d', aspect='equal')
            self.flag3D=True
        self.ax.scatter(x,y,z, c=color, marker=self.marker, lw = 0, s=self.markerSize*5)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x.max()+x.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y.max()+y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(z.max()+z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            self.ax.plot([xb], [yb], [zb], 'w')
#         self.set_axes_equal()


    def plot3D(self, x, y, z, color=None):
        if color is None: color=self.getColor()
        if not self.flag3D:
            self.ax = self.fig.add_subplot(111, projection='3d', aspect='equal')
            self.flag3D=True
        self.ax.plot_surface(x,y,z, color=color)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
    
    def set_axes_equal(self):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    
        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        '''
        if self.flag3D:
            x_limits = self.ax.get_xlim3d()
            y_limits = self.ax.get_ylim3d()
            z_limits = self.ax.get_zlim3d()
        
            x_range = abs(x_limits[1] - x_limits[0])
            x_mean = np.mean(x_limits)
            y_range = abs(y_limits[1] - y_limits[0])
            y_mean = np.mean(y_limits)
            z_range = abs(z_limits[1] - z_limits[0])
            z_mean = np.mean(z_limits)
        
            # The plot bounding box is a sphere in the sense of the infinity
            # norm, hence I call half the max range the plot radius.
            plot_radius = 0.5*max([x_range, y_range, z_range])
            
            #to make same scaling and position fix this values
            x_mean, y_mean, z_mean, plot_radius = 40.0, 55.0, 30.0, 45
        
            self.ax.set_xlim3d([x_mean - plot_radius, x_mean + plot_radius])
            self.ax.set_ylim3d([y_mean - plot_radius, y_mean + plot_radius])
            self.ax.set_zlim3d([z_mean - plot_radius, z_mean + plot_radius])
        else: self.ax.set_aspect('equal')
            
    def plotStats(self, sets, keys, zeroY=True, stdFlag=True):
        # plots sets of values sparsing them on x axis
        nSets = len(sets)
        maxV = np.max(sets)
        if zeroY: minV = 0
        for i in range(nSets):
            myset = np.sort(sets[i])
            std = np.std(myset)
            size = np.sqrt(np.array(myset).size)
#             size = np.sqrt(myMath.getSubIndex(myset, mean-std, mean+std).size)
            dx = 0.75/size
            dy = 4.*std/size
            x = []
            j0 = 0
            k=0
            for y in myset:
                j = int((y-myset[0])/dy)
                if j==j0: k+=1
                else:
                    k=1
                    j0=j
                if k%2==0: x.append(dx*k/2)
                else: x.append(-dx*k/2)
            x = np.array(x)
            self.scatter(x+2*i+1, myset)
            if stdFlag: self.errorbar([2*i+1], np.mean(myset), np.std(myset), color='k')
            else: self.errorbar([2*i+1], np.mean(myset), np.std(myset)/np.sqrt(myset.size), color='k')
        self.xticks(np.arange(nSets)*2+1, keys)
        if zeroY: self.ylim((0,None))
        
    
    def setBGColor(self, color):
        self.ax.set_axis_bgcolor(color)
    
    def makeBlack(self):
        self.setBGColor('k')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.yaxis.label.set_color('white')
        self.ax.xaxis.label.set_color('white')

if __name__ == '__main__':
    import numpy
    
    myFigure1 = myFigure()
    x = np.random.normal(size=1000)
    y = x * 3 + np.random.normal(size=1000)
    myFigure1.scatter(x, y, density=True)
    myFigure1.show()
#     myFigure1.save('test.svg')
#     x = numpy.arange(100)
#     y = numpy.sin(numpy.pi*x**2/50.) 
#     myFigure1.plot(x,y, color='r', ls = '--', label = r'$\sin\left(\frac{\pi x^2}{50}\right)$')
#     myFigure1.plot(x,numpy.sqrt(numpy.abs(y)), color='b', ls = '-', marker='o', label = r'$\sqrt{\left|\sin\left(\frac{\pi x^2}{50}\right)\right|}$')
#     myFigure1.plot(x,numpy.sin(numpy.pi*x/50.),'--g')
#     myFigure1.legend(3)
#     myFigure1.xlabel('x')
#     myFigure1.ylabel('y')
#     myFigure1.save()
#     myFigure1.show()
