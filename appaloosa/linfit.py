import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin

def logL(X,*args):
    '''
    Calculate the term that has to be minimized 
    to find the best-fit line parameters in X 
    
    Parameters:
    -----------
    X - (b, theta=arctan(m)) intercept and slope of line fit
    *args - (x, y, sigy, sigx, sigxy) 
    
    
    Return:
    --------
    -ln(likelihood)
    
    '''
    
    b, theta = X
    L = []
    Lx, Ly, Lsigy, Lsigx, Lsigxy = args
    
    for (x, y, sigy, sigx, sigxy) in list(zip(Lx, Ly, Lsigy, Lsigx, Lsigxy)):

        c = np.cos(theta)
        s = np.sin(theta)
        B = ((-1.) * (s * x)) + (c * y) - (b * c)
        A = (sigx * (s**2)) - (2 * sigxy * s * c) + (sigy * (c**2))
        L.append(0.5 * (B**2) / A)

    return sum(L)

def linfit(data):
    
    '''
    Fits a linear function to a data set with errors in x and y,
    calculates errors on best fit parameters using jackknife algorithm
    
    
    Parameters:
    ------------
    data - DataFrame with x, y, sigx, sigy, rho (correlation factor)
    
    Return:
    (m_mean, msig) - slope with uncertainty
    (b_mean, bsig) - intercept with uncertainty
    '''
    
    b_, theta_ = [], []
    data = data.applymap(float)
    data['sigxy']=data.sigy*data.sigx*data.rho
    mi, ma = data.x.min(),data.x.max()
    t = np.arange(mi,ma,(ma-mi)/200.)
    
    #fig = plt.figure(figsize=(8,6))
        
    for id_ in data.index.values:
        d = data.drop(id_)
        x, y, sigy, sigx, sigxy = d.x, d.y, d.sigy, d.sigx, d.sigxy
        b, theta = fmin(logL, [60,1.5],
                        args=(d.x, d.y, d.sigy, d.sigx, d.sigxy),
                        disp=0)
        lin = b + np.tan(theta) * t
        plt.plot(t, lin, alpha=0.2)
        b_.append(b)
        theta_.append(theta)
    
    #plt.scatter(x=data.x, y=data.y)
    N = data.shape[0]
    b_, theta_ = np.asarray(b_), np.asarray(theta_)
    m_ = np.tan(theta_)
    m_mean = m_.mean()
    b_mean = b_.mean()
    sigm = np.sqrt( (N-1) / N * ( (m_ - m_mean)**2 ).sum() )
    sigb = np.sqrt( (N-1) / N * ( (b_ - b_mean)**2 ).sum() )
    print('m = {} +/- {},\nb = {} +/- {} '.format(m_mean, sigm, b_mean, sigb))
    return (m_mean, sigm), (b_mean, sigb)
 