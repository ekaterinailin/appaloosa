import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import glob
import os
from appaloosa import RunLC
import os
import time

from joblib import Parallel, delayed
import multiprocessing
    

def fetch(ID,C,dbmode):
    if dbmode=='k2sc':
        myfile = 'hlsp_k2sc_k2_llc_{}-c{}_kepler_v2_lc.fits'.format(ID,C)
        if glob.glob(myfile)==[]:
            os.system('wget -q http://archive.stsci.edu/hlsps/k2sc/v2/c{2}/{0}00000/hlsp_k2sc_k2_llc_{1}-c{2}_kepler_v2_lc.fits'.format(str(ID)[:4],ID,C))
    elif dbmode=='everest':
        myfile = 'hlsp_everest_k2_llc_{}-c{}_kepler_v2.0_lc.fits'.format(ID,C)
    return myfile

#    for i,ID in enumerate(list(df.EPIC)):
def processInput(i,ID,C):
    print('This is {}. Injections started! {}th light curve.'.format(ID,i))
    
    dbmode = 'k2sc'
    myfile = fetch(ID, C, dbmode) 

    respath = os.getcwd()
    print(respath)
    
    if dbmode=='k2sc': mode=1
    elif dbmode=='everest': mode=3

    if glob.glob('/home/eilin/research/k2_cluster_flares/aprun/k2sc/{}-c{}_kepler_v2_lc.fits_all_fakes.csv'.format(ID,C))==[]:
        try:
            RunLC(myfile, dbmode=dbmode, mode=mode, display=False, debug=False, dofake=True, nfake=20, iterations=300, respath=respath) 
        except:
            fails.append(ID)
            print('LC {} failed. ID saved.'.format(ID))
    print(time.clock()) 
    return






os.chdir('/work1/eilin/appaloosa/appaloosa')
clusters =[["Pleiades","04",[]],]




for cluster,C,fails in clusters:
    df = pd.read_csv('{}_parameter_extra.csv'.format(cluster))
    os.chdir('/work1/eilin/data/CLUSTERS_01/CLUSTERS_01_{}/LC/LLC'.format(cluster))
    
    num_cores = multiprocessing.cpu_count()
    print('Number of cores: {}'.format(num_cores))
    Parallel(n_jobs=num_cores-1)(delayed(processInput)(i,ID,C) for (i, ID) in enumerate( list( df.EPIC ) ) )
        
    with open('{}_fails.txt'.format(cluster), 'a') as f:
        f.write(str(fails))


#('M67','05'),check,check
#('M44','05'),check,check
#('Pleiades','04'),check,check
#('NGC_1647','13'),nocheck,check


    
