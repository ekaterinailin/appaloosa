import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import glob
import os
from appaloosa import RunLC
import os
import time
os.chdir('/work1/eilin/appaloosa/appaloosa')
clusters =[("M67","05")]


def fetch(ID,C,dbmode):
    if dbmode=='k2sc':
        myfile = 'hlsp_k2sc_k2_llc_{}-c{}_kepler_v2_lc.fits'.format(ID,C)
        if glob.glob(myfile)==[]:
            os.system('wget -q http://archive.stsci.edu/hlsps/k2sc/v2/c{2}/{0}00000/hlsp_k2sc_k2_llc_{1}-c{2}_kepler_v2_lc.fits'.format(str(ID)[:4],ID,C))
    elif dbmode=='everest':
        myfile = 'hlsp_everest_k2_llc_{}-c{}_kepler_v2.0_lc.fits'.format(ID,C)
    return myfile

for cluster,C in clusters:
    df = pd.read_csv('{}_parameter.csv'.format(cluster))
    os.chdir('/work1/eilin/data/CLUSTERS_01/CLUSTERS_01_{}/LC/LLC'.format(cluster))
    for i,ID in enumerate(list(df.EPIC)):
        print('This is {}. Injections started! {}th light curve.'.format(ID,i))
        
        dbmode = 'everest'
        myfile = fetch(ID, C, dbmode) 

        respath = os.getcwd()
        print(respath)
        
        if dbmode=='k2sc': mode=5
        elif dbmode=='everest': mode=3
        
        RunLC(myfile, dbmode=dbmode, mode=mode, display=False, debug=False, dofake=True, nfake=20, iterations=300,respath=respath) 
        print(time.clock())

#('M67','05'),check,check
#('M44','05'),check,check
#('Pleiades','04'),check,check
#('NGC_1647','13'),nocheck,check
