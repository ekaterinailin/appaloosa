import pandas as pd
import numpy as np
import math

def write_flares(forf,cluster, test, EPIC, typ='flares'):
    
    '''
    Writes a data frame with flares or flux 
    for a certain light curve with EPIC ID from a cluster
    analysed during a certain test.
    
    Parameter:
    ------------
    
    Returns:
    ------------
    '''
    
    loc = 'stars_shortlist/{}/results/{}'.format(cluster,test)
    if typ == 'flares':
        forf.to_csv('{}/{}_flares.csv'.format(loc,EPIC))
    elif typ == 'flux':
        forf.to_csv('{}/{}_flux.csv'.format(loc,EPIC))
    return 

def find_nearest(array,value):
    '''
    
    Helper function that performs a binary search for the closest value in an array:
    
    Input:
    
    array - sorted array
    value - same type as elements in array
    
    Return:
    
    closest element in array
    index of the closest element in array
    
    '''

    idx = np.searchsorted(array, value, side="right")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1], idx-1
    else:
        return array[idx], idx

def fakedf(EPIC,C,cluster,run,LCtype,mode='numcorr'):
    loc = '/home/eilin/research/k2_cluster_flares/aprun/k2sc_05'
    if LCtype=='everest':
        fakes = pd.read_csv('{}/{}-c{}_kepler_v2.0_lc.fits_all_fakes.csv'.format(loc, LCtype, EPIC, C),
               usecols=['ed_fake', 'rec_fake', 'ed_rec', 'ed_rec_err', 'istart_rec', 'istop_rec'])
    elif LCtype == 'k2sc':
        fakes = pd.read_csv('{}/{}-c{}_kepler_v2_lc.fits_all_fakes.csv'.format(loc, EPIC, C),
               usecols=['ed_fake', 'rec_fake', 'ed_rec', 'ed_rec_err', 'istart_rec', 'istop_rec'])
    bins = np.power(10,np.arange(-2,6,0.015))
    fakes['ed_ratio']=fakes.ed_rec/fakes.ed_fake
    m = pd.DataFrame()
    if mode=='numcorr':

        fakes = fakes.sort_values(by='ed_fake')[(fakes.ed_fake < 1e6) & (fakes.ed_ratio < 1.2)]
        fakes['range1'], bins = pd.cut(fakes.ed_fake, bins, retbins=True,include_lowest=True)
        m['mean_ed_fake'] = fakes.groupby('range1').ed_fake.mean()
        m['mean_rec_fake'] = fakes.groupby('range1').rec_fake.mean()
        m['std_rec_fake'] = fakes.groupby('range1').rec_fake.std()
        #m['mean_ed_fake'] = pd.rolling_mean(m.mean_ed_fake, 3)
        
    elif mode=='EDcorr':
        fakes = fakes.sort_values(by='ed_fake')[fakes.ed_fake < 1e6]
        fakes['range1'], bins = pd.cut(fakes.ed_rec, bins, retbins=True,include_lowest=True)
        m['mean_ed_rec'] = fakes.groupby('range1').ed_rec.mean()
        m['mean_ED_corr'] = fakes.groupby('range1').ed_ratio.mean()
        m['std_ED_corr'] = fakes.groupby('range1').ed_ratio.std()
        #m['mean_ed_rec'] = pd.rolling_mean(m.mean_ed_rec, 3)
        
    elif mode=='falsepos':
        fakes = fakes.sort_values(by='ed_fake')[fakes.ed_fake < 1e6]
        fakes['range1'], bins = pd.cut(fakes.ed_rec, bins, retbins=True,include_lowest=True)
        group = fakes.groupby('range1').ed_ratio #identify false positives...
        m['mean_ed_rec'] = fakes.groupby('range1').ed_rec.mean()
        m['mean_fp_rate'] = group.apply(lambda x: x[x > 1.].shape[0]) / group.size() #...here.
        #m['summe'] = false.groupby('range1').ed_ratio.size()
        
    m = m.dropna(how='any')
    return m


def fakescorr(flares, EPIC, C, cluster, test, LCtype):
    
    m = fakedf(EPIC,C,cluster,test,LCtype,mode='numcorr')
    n = fakedf(EPIC,C,cluster,test,LCtype,mode='EDcorr')
    p = fakedf(EPIC,C,cluster,test,LCtype,mode='falsepos')
    #m.plot('mean_ed_fake','mean_rec_fake',yerr='std_rec_fake',
#            xlim=(0,10000),ylim=(0,1.1),figsize=(8,4),label='recovery rate',loglog=True)

    #flares = pd.read_csv('{}{}/results/{}/{}_flares.csv'.format(loc,cluster,test,EPIC))

    numcorr, EDcorr, falsepos, counttrue = [], [], []
    for ED_rec in flares.myed:
        if ED_rec == np.nan:
            numcorr.append(0)
        else:
            ED_recbinned, id_ = find_nearest(np.asarray(n.mean_ed_rec),ED_rec)
            ED_true = ED_recbinned / n.mean_ED_corr.iloc[id_]
            EDcorr.append(ED_true)
            
            _, id_ = find_nearest(np.asarray(p.mean_ed_rec),ED_rec)
            falsefrac = p.mean_fp_rate.iloc[id_]
            falsepos.append(falsefrac)
            
            _, id_ = find_nearest(np.asarray(m.mean_ed_fake),ED_true)
            numcorrfactor = 1./m.mean_rec_fake.iloc[id_]
            numcorr.append(numcorrfactor)
            
            #counttrue.append( (1.-falsefrac) * numcorrfactor )
            counttrue.append( numcorrfactor )
    flares['corrected'] = numcorr
    flares['falsepos_corrected'] = falsepos
    flares['ED_true'] = EDcorr
    flares['count_true'] = counttrue
    flares = flares.replace([np.inf, -np.inf], np.nan)
    write_flares(flares,cluster,test,EPIC)
    return flares
