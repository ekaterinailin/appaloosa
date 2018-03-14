'''
Script that removes flare candidates detected by Appaloosa that coincide with thruster firing events and that are classified as systematics due to simultaneous occurrence in several LCs in a single campaign. 
'''

import warnings
warnings.filterwarnings("ignore")

import time
import os
import math
import numpy as np
#import pyke as pk
import pandas as pd
from k2flix import TargetPixelFile
import matplotlib.pyplot as plt


from matplotlib import rcParams
rcParams['font.family'] = 'monospace'


def read_cluster(oid_list,test):
    
    '''
    Reads in TargetPixelFiles, flux and flare candidates for a list of Appaloosa processed LCs.
    
    Input:
    
    oid_list - list of EPIC IDs
    test - suffix for path
    
    Return:
    
    tpfs - dict of TargetPixelFiles including all quality flags
    flux - dict of DataFrames with LC results
    flares - dict of DataFrames with flare edges
    '''
    
    tpfs, flux, flares = dict(), dict(), dict()
    
    for oid in oid_list:
        tpfs[oid] = TargetPixelFile('ktwo{}-c05_lpd-targ.fits.gz'.format(oid))
        
        flares[oid] = pd.read_csv('results/{}/{}_flares.csv'.format(test,oid),
                                 usecols=['istart','istop'],
                                 dtype=int)
        flux[oid] = pd.read_csv('results/{}/{}_flux.csv'.format(test,oid),usecols=['flux_gap','time','flux_model'])
        flux[oid] = tpf_time_match(flux[oid],tpfs[oid])
        flux[oid].to_csv('results/{}/{}_tpf_times_FLUX.csv'.format(test,oid))
        #flux[oid] = pd.read_csv('results/{}/{}_tpf_times_FLUX.csv'.format(test,oid))
        #flux[oid] = pd.read_csv('results/{}/{}_tpf_times_FLUX.csv'.format(test,oid),
        #                        usecols=['flux_gap','time','flux_model','tpf_time','tpf_flags'])
    return flux, flares, tpfs
                                          

def tpf_time_match(flux, tpf):

    '''
    
    Matches the times given in flux.time to the times in the target pixel file tpf
    and adds the flags that are stored in tpf
    
    Input:
    
    flux - DataFrame for LC analysis 
    tpf - TargetPixelFile
    
    Return:
    
    flux - now with extra column for matched times from tpf and quality flags
    '''
    
    flags, timed = [], []
    quality = tpf.hdulist[1].data['QUALITY']
    for _, row in flux.iterrows():

        t, idx_ = find_nearest(tpf.bkjd(), row.time)
        timed.append(t)
        flags.append(quality[idx_])

    flux['tpf_flags'], flux['tpf_time'] = flags, timed
    return flux


def remove_thruster_firings(tpfs, flux, flares, oid):
    remove = []
#    for i in range(len(tpfs[oid].hdulist[1].data['FLUX'])): 
#        if tpfs[oid].hdulist[1].data['QUALITY'][i] > 524288:#flag values are summed up!
#            remove.append(tpfs[oid].bkjd(i))
#            remove.append(tpfs[oid].bkjd(i-1))
#            remove.append(tpfs[oid].bkjd(i+1))
    t = tpfs[oid].bkjd()
    fluxlen = len(tpfs[oid].hdulist[1].data['FLUX'])
    quality = tpfs[oid].hdulist[1].data['QUALITY']
    remove = [k for item in [t[i-1:i+2] for i in range(fluxlen) if quality[i] > 524288] for k in item]
    remove_id = [flux[oid].index.values[np.round(flux[oid].time,6) == np.round(remove_time,6)][0] for remove_time in sorted(list(set(remove)))]
    isflare = edges_to_bool(flares[oid],flux[oid])
    new_isflare = np.array(isflare)
    new_flags = np.array(flux[oid].tpf_flags)
    for id_ in remove_id:
        if isflare[id_] == 1.:
            for j, row in flares[oid].iterrows():
                if (row.istart <= id_) & (row.istop+1 >=id_):
                    new_isflare[row.istart:row.istop+1] = 0
                    new_flags[row.istart:row.istop+1] = 0

    flux[oid]['isflare'] = isflare
    flux[oid]['new_isflare'] = new_isflare
    flux[oid]['new_tpf_flags'] = new_flags
    return


def import_remove_systematics(s, flux,test):
    
    '''
    
    Add boolean array of flare candidates and leftover flags from systematics file that contains
    all indices into flux where systematic errors have been found while comparing several LCs in a batch.
    
    Input:
    
    flux - Main DataFrame that contains all info about the processed LCs
    test - suffix, points to folder in which the systematics file is stored
    
    '''
    #s = pd.read_csv('systematics.csv'.format(test),names=['systematics']).systematics.tolist()
    flux['tpf_flags_wo_systematics'] = np.array(flux.tpf_flags)
    flux['isflare_wo_systematics'] = np.array(flux['isflare'])
    flux['tpf_flags_wo_systematics'].iloc[s]= 0
    flux['isflare_wo_systematics'].iloc[s]= 0
    return

def display_overlay(oid_list,location, s, t, display=False,save=False):

    os.chdir(location)
    if display == True:
        all_flux={}
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for idx, oid in enumerate(oid_list):
            flux = pd.read_csv(str(oid)+'_flux.csv',
                              usecols = ['flux_gap','time'])
            mean = flux.flux_gap.median()
            plt.plot(list(flux.time)[s:t],list(flux.flux_gap/mean)[s:t],label='EPIC {}'.format(oid))
        plt.show()
        input('Next!')
        if save == True:
            plt.savefig('overlay_{}_{}'.format(s,t))
        plt.close()
    return
    
    
def edges_to_bool(start_stop,flux):
    
    '''
    
    Converts a dataframe with given edges for flares in istart/istop pairs to a single boolean array.
    
    Input:
    start_stop - pd.DataFrame with istart/istop columns
    flux - pd.DataFrame where isflare array shall be added to
    
    
    Return:
    isflare - boolean array with 1 = flare candidate detected in that data point
    
    '''
    
    rangeflare = [list(range(row.istart,row.istop+1)) for _, row in start_stop.iterrows()]

    #flatten the nested list:
    rangeflare = [item for sublist in rangeflare for item in sublist]
    
    #generate array 
    isflare = np.zeros_like(flux.flux_gap.tolist())
    np.put(isflare,rangeflare,np.ones_like(rangeflare))

    return isflare  

def bool_to_edges(isflare):

    isflare = list(isflare)
    start = [id_+1 for id_, bool_ in enumerate(isflare[1:]) if (isflare[id_] == False) & (bool_ == True)]
    stop = [id_ for id_, bool_ in enumerate(isflare[1:]) if (isflare[id_] == True) & (bool_ == False)]
    print(start[:5],stop[:5])
    #If LC starts or ends with a  flare candidate
    if isflare[0] == 1: 
        start = [0] + start
    if isflare[-1] == 1:
        stop = stop +[len(isflare)-1]

    start_stop = pd.DataFrame({'istart':start},dtype=int)
    start_stop = start_stop.join(pd.Series(stop,name='istop',dtype=int))
    
    return start_stop   

def create_inflated_binoms(oid_list,p,maxlen,display=False):

    inflated_binoms={}
    for i in [1,2,3,4,5]:
        binom = 0
        for oid in oid_list:
            not_inflated = np.random.binomial(1,p[oid]/i,maxlen//i)
            inflated = [item for sublist in [[j]*i for j in not_inflated] for item in sublist]
            binom = np.add(inflated,binom)
        inflated_binoms[i]=binom
    if display == True:
        fig, ax = plt.subplots()
        ax.plot(inflated_binoms[3])
        plt.show()
        input('Next!')
        plt.close()
    return inflated_binoms

def remove_systematics(oid_list,seq,maxlen,maxpoints=5):
    
    #create cleaned data frame:
    systematics = list(seq[maxpoints<seq.sum(axis=1)].index)
    remove = pd.Series(sorted(list(set([item for sublist in [[i-1,i,i+1] for i in systematics] for item in sublist]))))
    #sys_set = systematic_set(remove)
  
    #overlap dirty, overlap clean:
    seq_drop = seq.drop(labels=remove)
    od = [seq[seq.sum(axis=1)>=i].shape[0] for i in range(len(oid_list))]
    oc = [seq_drop[seq_drop.sum(axis=1)>=i].shape[0] for i in range(len(oid_list))]

    #Calculate the binomial probability for cleaned version
    p = seq.sum()/maxlen
    return od, oc, p, remove

def display_comparison(inflated_binoms, overlap_dirty,overlap_clean,display=False,save=False):
    
    if display == True:
        fig = plt.figure(figsize=(10,2))
        ax = fig.add_subplot(111)
        for i,binom in inflated_binoms.items():
            overlap_binom, edges = np.histogram(binom, bins=21, range=(0,21))
            #print(overlap_binom)
            overlap_binom = np.cumsum(overlap_binom[::-1])[::-1]
            #print(overlap_binom,'\n',overlap)
            ax.plot(overlap_binom,label='{}-times inflated binomial distribution'.format(i))
        ax.plot(overlap_dirty,label='Real distribution with systematic errors')
        ax.plot(overlap_clean,label='Real distribution - systematics removed')
        #ax.plot(np.array(overlap)-overlap_binom,label='real-random')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.set_yscale('log')
        fig.show()
        input('Next!')
        if save == True:
            plt.savefig('systematics.png')
        plt.close()
    return

def generate_sequence(flares,flux,oid_list):

    all_isflare = {}
    for oid in oid_list:
        all_isflare[oid] = edges_to_bool(flares[oid],flux[oid])
    maxlen = max([len(val) for key,val in all_isflare.items()])

    for key, val in all_isflare.items():
        all_isflare[key] = np.concatenate((val,np.zeros(maxlen-len(val))))

    seq = pd.DataFrame(all_isflare)
    return maxlen, seq

def systematic_set(remove):
    
    '''
    Takes a boolean array of indices to remove and converts them to a list of edges to continuous sets of indices.
    
    Input:
    
    remove - boolean array of indices into flux
    
    Return:
    
    s - Series of 
    
    '''
    s, j = [], 0
    print(remove[:20])
    for i, item in enumerate(remove[:-1]):
            if item+1 != remove[i+1]:
                s.append(remove[j:i+1])
                j = i+1
    print(s[:20])
    return pd.Series([i for item in s for i in item])

def systematics_wrap(flares, flux, oid_list, display=False, save=False):
    
    '''
    
    Find systematic errors that can be identified from correlations between LCs in a single campaign 
    and write the indices into flux that need to be removed according to it in a file.
    
    Input:
    
    location - path to data
    oid_list - list of EPIC IDs that have LCs and TargetPixelFiles in the same campaign
    display, save - flags for figure outputs
    
    '''
    
    maxlen, seq = generate_sequence(flares, flux, oid_list)
    overlap_dirty, overlap_clean, p, remove = remove_systematics(oid_list, seq, maxlen, maxpoints=0.12*len(oid_list))
    
    #Write results
    #remove.to_csv('systematics.csv')
    
    #Show results:
    inflated_binoms = create_inflated_binoms(oid_list, p, maxlen,display=display)
    display_comparison(inflated_binoms, overlap_dirty, overlap_clean,display=display,save=save)
    display_overlay(oid_list,location, 1330, 1360,display=display,save=save)
    display_overlay(oid_list, location, 1310, 1330,display=display,save=save)
    
    return remove


#------------------------------------------------------------------------------
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

    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1], idx-1
    else:
        return array[idx], idx






#--------------------------------------------------------------------------------------------------
#Parameters for script go here:

cluster = 'M44'
os.chdir('/home/ekaterina/Documents/appaloosa/stars_shortlist/{}'.format(cluster))
oid_list = pd.read_csv('M44_full.txt',names=['EPIC'])
oid_list = oid_list.EPIC.tolist()
test = 'run_01'
save = True
display = True

#---------------------------------------------------------------------------------------------------
# Main:

flux, flares, tpfs = read_cluster(oid_list, test)

location = '/home/ekaterina/Documents/appaloosa/stars_shortlist/{}/results/{}'.format(cluster,test)

remove = systematics_wrap(flares, flux, oid_list, display=display,save=save)

for oid in oid_list:
    remove_thruster_firings(tpfs, flux, flares, oid)
    import_remove_systematics(remove, flux[oid], test)
    flux[oid].to_csv('{}_tpf_times_FLUX.csv'.format(oid))
    
for oid in oid_list:
    flux[oid]['isflare_no_sys_no_thruster'] = (flux[oid].new_isflare == 1.) & (flux[oid].isflare_wo_systematics == 1.)
    flux[oid].to_csv('{}_flux.csv'.format(oid))
    start_stop = bool_to_edges(flux[oid].isflare_no_sys_no_thruster)
    flares[oid] = flares[oid].join(start_stop,rsuffix='_no_sys_no_thruster')
    flares[oid].to_csv('{}_flares.csv'.format(oid))