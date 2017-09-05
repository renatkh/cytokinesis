'''
Created on Jan 26, 2016

@author: renat
'''
import numpy as np
from lmfit import Model, Parameters
from myFigure import myFigure
from lmfit.printfuncs import fit_report

def ringSizeFuncHalf(x, r0, tau, t0):
    '''
    Function defining the equation for ring size change using equation:
    R(t) = r0*exp(g*(exp(t/tau)-1)),
    tau:=1/alpha*C_0, where C_0 is initial concentration of active myosin and alpha defines compression velocity as v(t)=alpha*C(t)
    g:=beta*w/alpha, where w is the active zone width and beta defines the ring shrinkage rate as 1/R*dR/dt=beta*C(t)
    '''
    return r0*(2.*r0)**(-np.exp((x-t0)/tau))

def fitRingSizeHalf(t, r, rErr=None):
    '''
    Fits ring size data to the ringSizeFunc (above)
    Returns: minimizer object defined by lmfit class. To access the parameters use res.best_values['r0'].
    '''
    
    if rErr==None: rErr = np.ones_like(r)
    if r[0]<0.5 or r.size<=3: return None
    params = Parameters()
    params.add('tau', (t[-1]-t[0])/3., vary=True, min = 0, max=max(t))
    params.add('t0', t[np.argmin(np.abs(r-0.5))], vary=True)
    params.add('r0', 1.1, vary=False, min=0.)
    mod = Model(ringSizeFuncHalf)
    ind=np.where(r>0.2)[0]
    if ind.size<3: ind = np.where(r>0.)
    w = 1./rErr**2
    res=mod.fit(r[ind],x=t[ind],weights = w[ind], params=params)
    if False:
        fig = myFigure()
        print(fit_report(res))
        fig.plot(t, r, 'r')
        delta = (max(t)-min(t))
        x=np.arange(min(t)-0.2*delta,max(t)+0.2*delta,delta/100.)
        fig.plot(x, ringSizeFuncHalf(x,res.best_values['r0'], res.best_values['tau'],res.best_values['t0']),'k')
        fig.show()
    return res

if __name__ == '__main__':
    pass