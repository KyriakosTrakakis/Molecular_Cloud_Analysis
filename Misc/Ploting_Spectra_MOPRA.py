import matplotlib.pyplot as plt
from astropy.io import fits
import os
import matplotlib as mpl
import pandas as pd
import math as math
from astropy import units as u
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from pyspeckit.spectrum.units import SpectroscopicAxis
import pybaselines
import aplpy
import itertools
from specutils.manipulation import FluxConservingResampler, LinearInterpolatedResampler, SplineInterpolatedResampler
from specutils import Spectrum1D
from pybaselines.polynomial import modpoly
#from pybaselines import Baseline, utils
import mysql.connector


mydb = mysql.connector.connect(
  host="127.0.0.1",
  user="root",
 # passwd=" ",      #<----
  database = "HiGAL_250_280"   #<----
)

### Define multi-dimensional Gaussians ###

def gaus(x,a,mu,sigma):
	return a*np.exp(-(x-mu)**2/(2*sigma**2))

def gaus3(x,a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3):
	return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + a2*np.exp(-(x-mu2)**2/(2*sigma2**2)) +a3*np.exp(-(x-mu3)**2/(2*sigma3**2))

def gaus2(x,a1,mu1,sigma1,a2,mu2,sigma2):
	return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + a2*np.exp(-(x-mu2)**2/(2*sigma2**2))

def gaus4(x,a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3, a4,mu4,sigma4):
	return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + a2*np.exp(-(x-mu2)**2/(2*sigma2**2)) +a3*np.exp(-(x-mu3)**2/(2*sigma3**2)) +a4*np.exp(-(x-mu4)**2/(2*sigma4**2))

def gaus5(x,a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3, a4,mu4,sigma4, a5,mu5,sigma5):
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + a2*np.exp(-(x-mu2)**2/(2*sigma2**2)) +a3*np.exp(-(x-mu3)**2/(2*sigma3**2)) +a4*np.exp(-(x-mu4)**2/(2*sigma4**2)) +a5*np.exp(-(x-mu5)**2/(2*sigma5**2))

def gaus6(x,a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3, a4,mu4,sigma4, a5,mu5,sigma5, a6,mu6,sigma6):
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + a2*np.exp(-(x-mu2)**2/(2*sigma2**2)) +a3*np.exp(-(x-mu3)**2/(2*sigma3**2)) +a4*np.exp(-(x-mu4)**2/(2*sigma4**2)) +a5*np.exp(-(x-mu5)**2/(2*sigma5**2)) +a6*np.exp(-(x-mu6)**2/(2*sigma6**2))

def gaus7(x,a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3, a4,mu4,sigma4, a5,mu5,sigma5, a6,mu6,sigma6, a7,mu7,sigma7):
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + a2*np.exp(-(x-mu2)**2/(2*sigma2**2)) +a3*np.exp(-(x-mu3)**2/(2*sigma3**2)) +a4*np.exp(-(x-mu4)**2/(2*sigma4**2)) +a5*np.exp(-(x-mu5)**2/(2*sigma5**2)) +a6*np.exp(-(x-mu6)**2/(2*sigma6**2)) +a7*np.exp(-(x-mu7)**2/(2*sigma7**2))

def create_db_connection():
    # Use SQLAlchemy for better connection management
    db_url = "mysql+pymysql://root@127.0.0.1:3306/HiGAL_250_280"
    engine = create_engine(db_url)
    return engine

def execute_query(engine, query):
    try:
        with engine.connect() as connection:
            connection.execute(query)
            connection.commit()
    except Exception as e:
        print(f"An error occurred: {e}")

def noise_calc(y,file):

    fig, ax = plt.subplots()
    y=np.array(y)
    #y = y[~np.isnan(y)]
    print(np.shape(y), y)
    y_negative = y[np.where(y <  0)]
    y_new = np.concatenate((y_negative,-1*y_negative),axis=None)
    y_mean = np.nanmean(y_new)
    y_stdev = np.nanstd(y_new)

    y_noise_array =  y[np.where(y <  abs(y_stdev*5))]

    n, bin_edges, p = plt.hist(y_noise_array,bins = 20, alpha=0.5,color='rosybrown',align='mid' )

    gauss_bins = bin_edges[:-1]+((bin_edges[1]-bin_edges[0])/2.)

    #print(np.mean(y_noise_array), np.max(y_noise_array))
    popt, pcov = curve_fit(gaus,gauss_bins, n, p0=[np.nanmax(n),np.nanmean(y_noise_array),np.nanstd(y_noise_array)])
    #print(popt)

    guass_fit_bins = (np.array(range(-90,100,1))-5)/200.
    ax.plot(guass_fit_bins,gaus(guass_fit_bins,*popt),'r')
    plt.axvline(x=3*abs(popt[2]), color='grey', linestyle=':')
    ax.set_xlabel('T$_{mb}$ [K]')
    ax.set_ylabel('Number of Channels')
    ax.set_title(f"{file}")
    noise_string = r'RMS Noise ($\sigma$)= '+str(round(abs(popt[2]),2)) + ' K'
    mean_string = r'Mean Offset= '+str(round(abs(popt[1]),2)) + ' K'
    plt.text(0.05, 0.9, noise_string, transform=ax.transAxes)
    plt.text(0.05, 0.8, mean_string, transform=ax.transAxes)

    y = [n-popt[1] for n in y]
    y_noise_array = [n-popt[1] for n in y_noise_array]
    n, bins, p = plt.hist(y_noise_array,bins=bin_edges, alpha=0.25,color='green',align='mid' )
    ###plt.savefig(output_dir+f'{file}_noise.png')
    #plt.show()    #<--------------------- make shure this is commented when running big loop
    plt.close()

    print('')
    print('-----Noise parameter is calculated -----')
    print('')

    return(1*abs(popt[2]), y)

def baseline_correction(flux, x):

    #fig, ax = plt.subplots()
    #ax.step(x, flux, color = '#539ecd')

    smooth_y = uniform_filter1d(flux, 9)
    smooth_x=x

    #baseline_fitter = Baseline(x_data=smooth_x)
    #bkg_1 = baseline_fitter.imodpoly(smooth_y, poly_order=5)[0]

    bkg_1, f = modpoly(smooth_y, poly_order=5)

    flux_blsub = [flux[num]-bkg_1[num] for num in range(len(flux))]#corecting the flux value by subtracting baseline

    fig, ax = plt.subplots()
    ax.step(x,flux, color = '#539ecd')
    ax.step(x,flux_blsub, color = 'red')
    ax.step(x, np.array(bkg_1), color = 'lime')

    #plt.title(each)
    plt.xlim(-50,150)
    plt.ylabel(r'T$_a^*$ (K)')
    plt.xlabel(r'Velocity (km s$^{-1}$)')
    #print(path_to_output+each+'_data_raw.png')
    #plt.savefig(path_to_output+each+'_data_raw.png')
    #plt.show()    #<--------------------- make shure this is commented when running big loop
    #plt.close()

    #ax.step(smooth_x, smooth_y, color = 'orange')
    #ax.step(smooth_x, bkg_1, color = 'blue')

    #plt.show()    #<--------------------- make shure this is commented when running big loop
    print('')
    print('-----Baseline correction has been executed-----')
    print('')

    return(flux_blsub,x,bkg_1)

def change_v_res(Vlsr,Tmb,sm_idx):
    print(len(Vlsr), len(Tmb))
    print(Vlsr[:10])
    new_res = (Vlsr[1]-Vlsr[0])*sm_idx
    new_Vlsr = np.arange(min(Vlsr),max(Vlsr),new_res)
    print(len(new_Vlsr), new_Vlsr[:10])

    print(Tmb[:10])
    new_Tmb = list(itertools.chain.from_iterable([i]*sm_idx for i in [sum(Tmb[i:i+sm_idx])/sm_idx for i in range(0,len(Tmb),sm_idx)]))
    new_Tmb2 = new_Tmb[::sm_idx]
    print(len(new_Tmb2), new_Tmb[:10])
    return (new_Vlsr, new_Tmb2)

def smooth_1d_spectrum(Vlsr,Tmb,sm_idx):
    '''
    smooth the 1d spectra to sm_idx*resolution of 1d spectra
    for example, if want to reduce the velocity resolution from 0.25km/s to 0.5 km/s, set sm_idx=2
    '''

    Vlsr = np.array(Vlsr)*u.km/u.s
    Tmb  = np.array(Tmb)*u.K
    input_spec = Spectrum1D(spectral_axis=Vlsr, flux=Tmb)
    deltaVlsr =Vlsr[1]-Vlsr[0]
    print('the orignal spectral resolution',deltaVlsr)
    new_resolution = sm_idx*deltaVlsr.value
    print('the new spectral resolution',new_resolution)
    new_disp_grid = np.arange(min(Vlsr).value,max(Vlsr).value,new_resolution)*u.km/u.s
    print('channels of original spectral and new spectra',len(Vlsr),len(new_disp_grid))
    fluxcon = FluxConservingResampler(extrapolation_treatment='zero_fill')
    new_spec_fluxcon = fluxcon(input_spec, new_disp_grid)
    #outspec = Table([new_spec_fluxcon.spectral_axis,new_spec_fluxcon.flux],names=['Vlsr','Tmb'])
    ##you can save the results
    #ascii.write(outspec,infile[0:-4]+'_smooth_'+str(sm_idx)+infile[-4:],format='commented_header',overwrite=True)

    print('')
    print('-----Velocity resolution is reduced by a factor of 2 -----')
    print('')

    return new_spec_fluxcon.spectral_axis, np.nan_to_num(new_spec_fluxcon.flux,copy=True, nan=0.0)

def smooth(Tmb,window_len,window):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    #print(x.ndim)
    #print('length x',len(Tmb))
    #x=np.array(Tmb)
    #print('Tmb', Tmb[0:5])



    s=np.r_[Tmb[window_len-1:0:-1],Tmb,Tmb[-2:-window_len-1:-1]]

    #print('first part ', x[window_len-1:0:-1])
    #print('end part ',x[-2:-window_len-1:-1])

    #print('length first part',len( x[window_len-1:0:-1]))
   # print('length last part',len(x[-2:-window_len-1:-1]))
   # print('s ',s[0:8])
    #print(len(x[0:5]))

    #print('length s',len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')[window_len-3:-window_len+3]
    #x= vlsr[window_len-2:-window_len+2]
    #print('length y',len(y))
    #print(y)
    return  y

def fit_parameters_to_table(popt, pcov):
    #print(pcov)
    #print(popt)
    perr = np.sqrt(np.diag(pcov))

    #print('peak =',popt[0])
    #print('vlsr =',popt[1])
    #print ('width = ', popt[2])

    ppp=0
    lll = 0


    while ppp <= len(popt)-1:
        lll=lll+1

        print('integer ', ppp)
        print('num elements', len(popt))
        print('peak =',popt[ppp])
        print('vlsr =',popt[ppp+1])
        print ('width = ', popt[ppp+2])
        print('integrated flux ',  popt[ppp]*popt[ppp+2]*2.355, noise[iii]*9)

        if (popt[ppp] > noise[iii]*3. or popt[ppp]*popt[ppp+2]*2.355 > noise[iii]*9) and popt[ppp+2] > 0.15  :

            #print('perr[0]',str(round(chi2/float(NDF),4)))
            integrated_intensity_reference = popt[ppp+0]*popt[ppp+2]

            if math.isinf(perr[ppp]):
                perr[ppp] = -999

            if math.isinf(perr[ppp+1]):
                perr[ppp+1] = -999

            if math.isinf(perr[ppp+2]):
                perr[ppp+2] = -999


            ###query = "INSERT INTO table_name (higal_name, component_index,  Temp, Temp_error, vlsr, vlsr_error, dvlsr, dvlsr_error,integrated_intensity,intensity_ratio, input_time) VALUES ('"+each+"','"+str(lll)+ "','"+str(round((popt[ppp]),4)) +"', '" + str(round(abs(perr[ppp]),5)) +"', '"+str(round((popt[ppp+1]),4)) +"','" + str(round(abs(perr[ppp+1]),5)) + "','" +str(round((popt[ppp+2]),4)) + "','" + str(round(abs(perr[ppp+2]),5)) +"', '"+str(round(popt[ppp+0]*popt[ppp+2],4))+"','"+str(round(popt[ppp+0]*popt[ppp+2]/integrated_intensity_reference,4))+"', now()) "
            ###print(query)
            ###execute_query(mydb, query)
            #print('gauss fit ',np.array(xaxis))

            ppp= ppp+3
            
def plot_spectra(spectra_dir):
    # List all FITS files in the spectra directory. You might want to change or remove the 12CO/13CO -v specification depending on needs
    spectra_files = [f for f in os.listdir(spectra_dir) if f.endswith('spec.fits') and f.startswith('12CO')]
    df = pd.read_sql_query(f'SELECT gal_long, gal_lat, higal_name, VLSR FROM higal_mopra_co', con=mydb)

	# Loop through the list of files :
	
    for spectra_file in spectra_files[:]:
        with fits.open(os.path.join(spectra_dir, spectra_file)) as hdul:
            Tmb = hdul[0].data
            hd = hdul[0].header
        refc = hd['CRPIX3']
        velo = hd['CRVAL3']
        dv = hd['CDELT3']
        #print(hd)
        #print('length', np.shape(Tmb))
        #print(Tmb)


        Vlsr = np.divide(np.add(np.multiply(np.subtract(np.arange(1,hd['NAXIS1']+1), refc), dv), velo), 1000)
        xaxis = np.array(SpectroscopicAxis(Vlsr, refX=hd['RESTFREQ']*u.Hz, velocity_convention='radio'))
        xaxis=xaxis[674:-1658]
        Tmb = Tmb[674:-1658]

    # Reduce the velocity resoluton by a factor of 4:

        #xaxis, Tmb =smooth_1d_spectrum(Vlsr, Tmb,1)
        xaxis, Tmb =change_v_res(xaxis, Tmb,4)
        print(len(xaxis), len(Tmb))
        
    # Hanning smooth the data to improve the signal to noise:

        Tmb =smooth(Tmb, 5,'hanning')

        #Tmb, xaxis, bkg_1 = baseline_correction(Tmb, xaxis)
        noise, y = noise_calc(Tmb, spectra_file)
        
    # Figures out how many Gaussians to fit:
    
        peak_channel = 100+np.argmax([y[100:-100]])
        peak_vlsr = xaxis[peak_channel]
        print(peak_vlsr)
        #print(np.where(y[200:-400] >  noise*3.5))
        channels_above_noise = 100+np.where(y[100:-100] >  noise*3.5)[0]
        gaussian_flag = channels_above_noise.size
        print('Gauss flag',gaussian_flag)

        peaks, prop= find_peaks(y, height = noise*3.5, width = 3)
        component_index = len(peaks)
        
        if (component_index > 0 and component_index < 8):
            width_guess = .5
            if component_index == 1:
                guesses = [Tmb[peaks[0]], xaxis[peaks[0]], width_guess]
                gauss_fn =   gaus
            if component_index == 2:
                guesses = [Tmb[peaks[0]], xaxis[peaks[0]], width_guess, Tmb[peaks[1]], xaxis[peaks[1]], width_guess]
                gauss_fn =   gaus2
            if component_index == 3:
                guesses = [Tmb[peaks[0]], xaxis[peaks[0]], width_guess, Tmb[peaks[1]], xaxis[peaks[1]], width_guess,Tmb[peaks[2]], xaxis[peaks[2]], width_guess]
                gauss_fn =   gaus3
            if component_index == 4:
                guesses = [Tmb[peaks[0]], xaxis[peaks[0]], width_guess, Tmb[peaks[1]], xaxis[peaks[1]], width_guess,Tmb[peaks[2]], xaxis[peaks[2]], width_guess,Tmb[peaks[3]], xaxis[peaks[3]], width_guess]
                gauss_fn =   gaus4
            if component_index == 5:
                guesses = [Tmb[peaks[0]], xaxis[peaks[0]], width_guess, Tmb[peaks[1]], xaxis[peaks[1]], width_guess,Tmb[peaks[2]], xaxis[peaks[2]], width_guess,Tmb[peaks[3]], xaxis[peaks[3]], width_guess,Tmb[peaks[4]], xaxis[peaks[4]], width_guess]
                gauss_fn =   gaus5
            if component_index == 6:
                guesses = [Tmb[peaks[0]], xaxis[peaks[0]], width_guess, Tmb[peaks[1]], xaxis[peaks[1]], width_guess,Tmb[peaks[2]], xaxis[peaks[2]], width_guess,Tmb[peaks[3]], xaxis[peaks[3]], width_guess,Tmb[peaks[4]], xaxis[peaks[4]], width_guess,Tmb[peaks[5]], xaxis[peaks[5]], width_guess]
                gauss_fn =   gaus6
            if component_index == 7:
                guesses = [Tmb[peaks[0]], xaxis[peaks[0]], width_guess, Tmb[peaks[1]], xaxis[peaks[1]], width_guess,Tmb[peaks[2]], xaxis[peaks[2]], width_guess,Tmb[peaks[3]], xaxis[peaks[3]], width_guess,Tmb[peaks[4]], xaxis[peaks[4]], width_guess,Tmb[peaks[5]], xaxis[peaks[5]], width_guess,Tmb[peaks[6]], xaxis[peaks[6]], width_guess]
                gauss_fn =   gaus7

        y = np.nan_to_num(y)
        weight=0.0*y + noise
        
        ### STARTING FIGURE ###
        
        fig, ax = plt.subplots()
        plt.step(xaxis,y, color = '#539ecd')
        if (component_index > 0 and component_index < 8):
            popt, pcov = curve_fit(gauss_fn,xaxis, y,guesses, sigma=weight, absolute_sigma=True)
            plt.plot(xaxis,gauss_fn(xaxis,*popt),'r')
        #fit_parameters_to_table(popt, pcov)    #to output parameters to SQL table
        higal_v = df[df['higal_name'] == spectra_file[5:-10]]['VLSR'].iloc[0]
        noise_string = r'RMS Noise ($\sigma$)= '+str(round(abs(noise),2)) + ' K' ###noise[iii]
        plt.text(0.05, 0.85, noise_string, transform=ax.transAxes)
        plt.text(0.05, 0.8, f'HiGAL Vlsr={higal_v} km/s', transform=ax.transAxes)
        ax.axvline(higal_v, color='pink', alpha = .8, zorder=0)
        ax.set_xlim([-25, 150])
        plt.ylabel(r'T$_{\rm mb}$ (K)')
        plt.xlabel(r'Velocity (km s$^{-1}$)')
        plt.title(f'{spectra_file[:-10]}')
        plt.savefig(f'{output_dir}/{spectra_file[:-10]}.png')
        #plt.show()    #<--------------------- make shure this is commented when running big loop


spectra_dir = '/Users/kyriakos.trakakis/Downloads/MOPRA_HiGAL'		# directory holding the spectra
output_dir = '/Users/kyriakos.trakakis/Downloads/MOPRA_HiGAL/spectra_png'		# output directory
plot_spectra(spectra_dir)

