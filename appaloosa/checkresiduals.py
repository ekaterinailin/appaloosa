import numpy as np
import matplotlib.pyplot as plt
from lightkurve import KeplerTargetPixelFile
import logging
logging.basicConfig(filename='logfile.txt',level=logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger('simplistic logging')
logger.setLevel(logging.INFO)

def pixelplot(tpf, cadenceno=0):
    '''
    Plots the TPF model, TPF itself and the residual from the fit.
    
    Input:
    ---------------------
    tpf - KeplerTargetPixelFile
    parameters - TPFModel fit parameters
        
    Return:
    --------------------
    Panel of above plots as .png file.
    '''
    model = tpf.get_model()
    parameters = model.fit(tpf.flux[cadenceno] + tpf.flux_bkg[cadenceno])
    _, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,4))
    model.plot(parameters, ax = ax[0])
    tpf.plot(ax=ax[1])
    ax[2].imshow(np.flipud(parameters.residual_image))
    ax[2].set_title(cadenceno)
    plt.savefig('debugplot.png')
    return


def check_residual(tpf,model,cadenceno=0):
    '''
    Checks residuals of a cadence in the TPF for 3-sigma outliers.
    
    Input:
    -----------------------------
    tpf - KeplerTargetPixelFile
    model - TPFmodel
    cadenceno - cadence number in tpf default:0, 
    
    
    Return:
    -----------------------------
    bool - True iff there is an outlier in the residual pixel file.
    '''
    parameters = model.fit(tpf.flux[cadenceno] + tpf.flux_bkg[cadenceno])
    resid = np.flipud(parameters.residual_image)
    median = np.full_like(resid,np.nanmedian(resid))
    std = np.nanstd(resid)
    dev = np.abs(median-resid)
    if median[np.where(dev > 3*std)].shape[0] != 0:
        logger.debug('Cadence {} has an outlier in the residuals'.format(cadenceno))
        return True
    else: 
        return False

def check_residuals(tpf,cadences):
    
    '''
    Wrapper to check for residual outliers in a TPF for multiple cadences
    
    Input:
    --------------------
    tpf - KeplerTargetPixelFile
    cadences - array-like, contains cadence numbers to inspect in tpf
    
    Return:
    ----------------
    array-like - slice of input cadence that contain outliers in the residuals
    
    '''
    model = tpf.get_model()
    logging.debug(model)
    outlier = [check_residual(tpf,model,i) for i in cadences]
    return cadences[outlier]

