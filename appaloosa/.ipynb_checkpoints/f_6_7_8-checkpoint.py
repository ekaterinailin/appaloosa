import numpy as np
import random
import pandas as pd
import glob
import matplotlib.pyplot as plt
import specmatchemp.library
import specmatchemp.plots as smplot
from scipy.integrate import trapz
from scipy.constants import pi, h, c, k

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import CustomJS, Button
from bokeh.layouts import row, column

h*=1e7 # Planck constant in erg*s
c*=1e2 # light speed in cm/s
k*=1e7 # Boltzmann constant in erg/K

def dprint(str,debug):
    
    '''
    Prints out all help text snippets where the debug flag is passed.
    '''
    
    if debug==True:
        print(str)
    return

def display(opt,d,debug):
    
    '''
    Displays additional information about a resulting 
    combo _opt_ of spectral classes given some colors 
    and approximates their likelihood by the mutual distance
    between the rows _d_. Conditional on the _debug_ flag being passed.
    '''
    
    dprint('New combination: {}'.format(opt),debug)
    dprint('This combination has distance {}'.format(d),debug)
    
    return

def spec_class_hist(specs,cluster,sort):

    counts = specs.spec_class.value_counts(sort=False)
    y = counts.sort_index()
    plot = y.plot(kind = 'bar',color='blue',ylim=(0,210))#,xlim=('F4','M5.5'))
    fig = plot.get_figure()
    #fig.savefig('/home/ekaterina/Documents/appaloosa/stars_shortlist/share/clean_CMD_{}.jpg'.format(cluster),dpi=300)
    fig.savefig('stars_shortlist/share/clean_CMD_{}.jpg'.format(cluster),dpi=300)
    return y

def CMD(specs,cluster,cid1='SDSS_g',cid2='SDSS_i',colour='g_i',ylim=(19,5),outliers=pd.Series()):
    '''
    Plots and saves CMDs for specified bands, 
    marks outliers if any are passed.
    '''
    specs[colour]=specs[cid1]-specs[cid2]

    plot = specs.plot(x=colour,y=cid1,ylim=ylim,kind = 'scatter', 
                     # colormap=color_outlier_red(specs.index.values,outliers),
                      figsize=(5,4),title=readable(cluster))
    plot.set_ylabel(cid1[0])
    plot.set_xlabel('{}-{}'.format(cid1[0],cid2[0]))
    fig = plot.get_figure()
    #fig.savefig('/home/ekaterina/Documents/appaloosa/stars_shortlist/share/CMD_{}_{}.jpg'.format(cluster,color),dpi=300)
    fig.savefig('stars_shortlist/share/CMD_{}_{}.png'.format(cluster,colour),dpi=300)

    return

def readable(string):
    return ''.join([' ' if i=='_' else i for i in string])

def color_outlier_red(val,outliers):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    if outliers.empty: outliers=[]
    color=[]
    for id_ in val:
        if id_ in list(outliers):
            color.append('red')
        else:
            color.append('k')
    #color = ['red' if val in outliers else 'black'
    return color


def interactive_CMD(specs,cid1='SDSS_g',cid2='SDSS_i'):
    '''
    Simplistic tool to create an interactive 
    bokeh plot where outliers can be marked and saved in
    '/home/ekaterina/Documents/appaloosa/stars_shortlist/share/temp'
    '''
    # Create some random data and put it into a ColumnDataSource
    s = specs[[cid1, cid2]].dropna(how='any')
    x = list(s[cid1]-s[cid2])
    y = list(s[cid2])
    z = list(s.index.values)
    source_data = ColumnDataSource(data=dict(x=x, y=y,desc=z))
    
    # Create a button that saves the coordinates of selected data points to a file
    savebutton = Button(label="Save", button_type="success")
    savebutton.callback = CustomJS(args=dict(source_data=source_data), code="""
            var inds = source_data.selected['1d'].indices;
            var data = source_data.data;
            var out = "";
            for (i = 0; i < inds.length; i++) {
                out += data['desc'][inds[i]] + " ";
            }
            var file = new Blob([out], {type: 'text/plain'});
            var elem = window.document.createElement('a');
            elem.href = window.URL.createObjectURL(file);
            elem.download = 'selected-data.txt';
            document.body.appendChild(elem);
            elem.click();
            document.body.removeChild(elem);
            """)

    # Plot the data and save the html file
    p = figure(plot_width=800, plot_height=400, 
               #y_range=(20,7),
               tools="lasso_select, reset",)
    p.circle(x='x', y='y', source=source_data)
    p.xaxis.axis_label = '{}-{}'.format(cid1[0],cid2[0])
    p.yaxis.axis_label = cid1[0]
    plot = column(p, savebutton)
    output_file("test.html")
    show(plot)
    return

#----------------------------------------
#ENERGY CALC
#---------------------------------------

#Solution to nan-bug:
#https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolate_nan(y):
    nans, x= nan_helper(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    return

#End: Solution to nan_bug

def spectrum(T, R, lib, wavmin=3480., wavmax=9700.):
    
    '''
    
    Returns the spectrum of a star at effective temperature T in wavelength range [wavmin,wavmax]
    Default range corresponds to Kepler response
    
    '''
    #loads library of spectra within wavelength range
    
    if T > 7000:
        print('{}K is hotter than 7000K.'.format(T))
        return lib.wav, []
    elif T < 3000:
        print('{}K is cooler than 3000K.'.format(T))
        return lib.wav, []
    else: 

        #find the spectrum that fits T best
        print('T= ',T)
        Tmin = str(T-100.)
        Tmax = str(T+100.)
        Rmin = str(R-0.2)
        Rmax = str(R+0.2)
        cut = lib.library_params.query(Rmin + '< radius < ' + Rmax + ' and' + Tmin + ' < Teff < ' + Tmax)
        #print('cut\n',cut.head())
        T_offer = zip(list(cut['Teff']), list(cut['lib_index']))
        T_minindex = min(T_offer, key = lambda t: abs(t[0]-T))[1]
        cut = cut.loc[T_minindex]
        #return the spectrum
        spec = lib.library_spectra[cut.lib_index,0,:]
        return lib.wav, spec.T

def kepler_spectrum(T, R, lib,deriv=False):
    '''
    
    Convolves a blackbody of effective temperature T, 
    the spectrum of a dwarf star with corresponding spectral class,
    and the Kepler response function 
    to return the wavelength dependent flux of that star per area
    
    Parameters:
    -----------
    T - effective temperatur in K
    
    Returns:
    --------
    Kp_flux in erg/cm*(cm**2)
    Kp_midwav in angström
    
    
    '''
 #   Kp = pd.read_csv('/home/ekaterina/Documents/appaloosa/stars_shortlist/static/Kepler_response.txt',
    Kp = pd.read_csv('stars_shortlist/static/Kepler_response.txt',
                                  skiprows=9,
                                  header=None,
                                  delimiter='\t',
                                  names=['wav','resp'])
    Kp.wav = Kp.wav.astype(np.float)*10. #convert to angström for spectrum function
    #load the spectrum within a wavelength range that fits given T_eff best
    Spec_wav, Spec_flux = spectrum(T, R, lib,Kp.wav.min(),Kp.wav.max())
    #map Kepler response linearly into wavelengths given with the spectrum
    if Spec_flux == []:
        print('{}K is too hot or too cool for specmatch-emp.'.format(T))
        return [],[],[]
    else:
        Spec_flux = np.interp(Kp.wav,Spec_wav,Spec_flux)
        Kp_flux = np.empty(Kp.wav.shape[0]-1)
        Kp_midwav  = np.empty(Kp.wav.shape[0]-1)
        planck  = np.empty(Kp.wav.shape[0]-1)
        #calculate the flux of a star with given T_eff 
        #accounting for Kepler filter 
        #and the corresp. spectrum of that stellar type 
        try:
            for i, response in Kp.resp[:-1].iteritems():
                dlambda = (Kp.wav[i+1]-Kp.wav[i])*1e-8 #infin. element of wavelength in cm
                lambda_ = Kp.wav[i:i+2].mean()*1e-8 #wavelength in cm
                Kp_midwav[i] = lambda_
                if deriv == False:
                    planck[i] = 2. * h * c**2 / lambda_**5 / (np.exp( h * c / ( lambda_ * k * T ) ) - 1. )     
                elif deriv == True:
                    e_ = np.exp( h * c / ( lambda_ * k * T ) )
                    planck[i] = 2. * h**2 * c**3 / lambda_**6 / T**2 / k / (e_ - 1. )**2 * e_     
                Kp_flux[i] = Spec_flux[i] * response * planck[i]
        except IndexError:
            pass
        
        return Kp_midwav, Kp_flux, planck

def plot_kepler_spectrum(T,R):

    wav, flux, planck = kepler_spectrum(T,R,lib)
    
    #Kp = pd.read_csv('/home/ekaterina/Documents/appaloosa/stars_shortlist/static/Kepler_response.txt',
    Kp = pd.read_csv('stars_shortlist/static/Kepler_response.txt',
                                  skiprows=9,
                                  header=None,
                                  delimiter='\t',
                                  names=['wav','resp'])

    if flux == []:
        return print('No data for T = {}K'.format(T))
    else:
        plt.figure()
        plt.plot(wav*1e8, flux,color='green')
        plt.plot(wav*1e8, Kp.resp[:-1]*planck.max(),color='red')
        plt.plot(wav*1e8, planck,color='black')
        plt.show()  
    return

def kepler_luminosity(T,R,lib, error=False):
    
    '''
    Integrates the Kepler flux,
    multiplies by the area A=pi*(R**2),
    to obtain observed quiescent flux for an object
    
    Return:
    
    total Kepler luminosity in erg/s of a dwarf star with effective temperature T
    
    '''
    params=pd.read_csv('stars_shortlist/static/merged_specs.csv')

  #  params=pd.read_csv('/home/ekaterina/Documents/appaloosa/stars_shortlist/static/merged_specs.csv')
    
    #calculate Kepler spectrum of a dwarf star with temperature T
    if error==False:
        wav, flux, _ = kepler_spectrum(T,R,lib)
    elif error==True:
        wav, flux, _ = kepler_spectrum(T,R,lib,deriv=True)
        
    if flux == []:
        return print('{}K is too hot or too cold'.format(T))
    else:
        #interpolate where nans occur
        interpolate_nan(wav)
        interpolate_nan(flux)

        #select the relevant columns from params
        radii_teff = params[['T','R_Rsun']]
        radii_teff.set_index('T',inplace=True)
        radius_cm = radii_teff.R_Rsun[T]*6.96342e10 #stellar radius in cm
        if error == False:
            return np.trapz(flux, wav) * pi * (radius_cm**2)
        elif error == True:
            return np.trapz(flux, wav) * pi * (radius_cm**2)
    
def Kp_to_Lum(df, dm, Kp='Kp'):
    df['Kp_abs'] = df[Kp]+dm
    #print(df.head())
    return

def Mbol_to_Lum(Mbol, errMbol=pd.Series(), err=False):
    
    '''
    Returns:
    --------
    Bolometric luminosity of a star with Mbol in erg/s.
    '''
    
    Lum_Sun = 3.84e33 #erg/s
    Mbol_Sun = 4.74 #mag
    if err==False:
        return Lum_Sun * 10**( Mbol_Sun - Mbol )
    elif err==True:
        return Lum_Sun * 10**( Mbol_Sun - Mbol ) * np.log(10) * errMbol

def merged_spec_class(params):
    #p = pd.read_csv('/home/ekaterina/Documents/appaloosa/stars_shortlist/static/merged_specs.csv')
    p = pd.read_csv('stars_shortlist/static/merged_specs.csv')
    colors = {'g_r':('SDSS_g','SDSS_r'),
              'r_i':('SDSS_r','SDSS_i'),
              'i_z':('SDSS_i','SDSS_z'),
              'z_J':('SDSS_z','J'),
              'J_H':('J','H'),
              'H_K':('H','K'),
              'i-z':('SDSS_i','SDSS_z'),
              'z-Y':('SDSS_z','SDSS_y'),
              'J-H':('J','H'),
              'H-K':('H','K'),}
    p['T_err'] = Terr(p['T'])
    p['R_Rsun_err'] = 0.2*p.R_Rsun
    dif = p.Mbol.diff().fillna(method='bfill').fillna(method='ffill')
    dif = pd.rolling_max(dif,2)/2.
    p['Mbol_err'] = dif.fillna(method='bfill')
    for col in p.columns.values:
        params[col]=np.nan
    for i, s in params.iterrows():
        _ = np.array([((s[item[0]]-s[item[1]])-p[key]).abs().argmin() for key, item in colors.items()])
        idx = np.int(np.median(_[~np.isnan(_)]))    
        for col in p.columns.values:
            params[col][i]=p[col].iloc[idx]

    params = params.join(p.iloc[idx])    
    return params


def L_quieterr(L, R, Rerr, T, Terr,lib):
    R*=6.96342e10 #stellar radius in cm
    Rerr*=6.96342e10 #stellar radius error in cm
    t1 = np.abs(L/R)
    deriv_L = kepler_luminosity(T, R, lib, error=True)
    t2 = np.abs(deriv_L)
    #print(t1,t2)
    return (t1 * Rerr)+ (t2 * Terr)

def Terr(Tseq):
    #python version problem
    Terr = pd.rolling_std(Tseq,window=3,center=True)#.std()
    Terr = Terr.fillna(100)
    #print(Terr)
    return Terr
