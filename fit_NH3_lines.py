import pyspeckit
import astropy.io.fits as fits
import numpy as np

from spectral_cube import SpectralCube
from pyspeckit.spectrum.units import SpectroscopicAxis
import signal_id
from radio_beam import Beam
import astropy.units as u
from skimage.morphology import remove_small_objects,closing,disk,opening


freq11= 23.6944955*u.GHz
freq22= 23.7226336*u.GHz

def fix_IC348():
    OneOneFile = 'fits_files/HH211_Per1/IC348.NH3.11.fits'
    TwoTwoFile = 'fits_files/HH211_Per1/IC348.NH3.22.fits'
    data, hd= fits.getdata( OneOneFile, header=True)
    hd['CTYPE3'] = 'VOPT'
    hd['VELREF'] = 257
    fits.writeto(OneOneFile, data, hd, clobber=True)

    data, hd= fits.getdata( TwoTwoFile, header=True)
    hd['CTYPE3'] = 'VOPT'
    hd['VELREF'] = 257
    fits.writeto(TwoTwoFile, data, hd, clobber=True)

def fix_IRAS03282():
    OneOneFile = 'fits_files/IRAS03282_Per5/IRAS03282.NH3.11.fits'
    TwoTwoFile = 'fits_files/IRAS03282_Per5/IRAS03282.NH3.22.fits'
    data, hd= fits.getdata( OneOneFile, header=True)
    hd['CUNIT3'] = 'm/s'
    fits.writeto(OneOneFile, data, hd, clobber=True)

    data, hd= fits.getdata( TwoTwoFile, header=True)
    hd['CUNIT3'] = 'm/s'
    fits.writeto(TwoTwoFile, data, hd, clobber=True)

def cubefit(region='IC348', do_plot=True, 
            snr_min=5.0, multicore=1):
    """
    Fit NH3(1,1) and (2,2) cubes for the requested region. 
    It fits all pixels with SNR larger than requested. 
    Initial guess is based on moment maps and neighboring pixels. 
    The fitting can be done in parallel mode using several cores, 
    however, this is dangerous for large regions, where using a 
    good initial guess is important. 
    It stores the result in a FITS cube. 

    TODO:
    -Store results in hdu list
    -Improve initial guess
    
    Parameters
    ----------
    region : str
        Name of region to reduce
    blorder : int
        order of baseline removed
    do_plot : bool
        If True, then a map of the region to map is shown.
    snr_min : numpy.float
        Minimum signal to noise ratio of the spectrum to be fitted.
    multicore : int
        Numbers of cores to use for parallel processing. 
    """
    if region == 'IC348':
        OneOneFile = 'fits_files/HH211_Per1/IC348_NH3_11.fits'
        TwoTwoFile = 'fits_files/HH211_Per1/IC348_NH3_22.fits'
        vmin=-1.0
        vmax=1.0
        rms=3e-3
    elif region == 'IRAS03282':
        OneOneFile = 'fits_files/IRAS03282_Per5/IRAS03282_NH3_11.fits'
        TwoTwoFile = 'fits_files/IRAS03282_Per5/IRAS03282_NH3_22.fits'
        vmin=6.0
        vmax=9.0
        rms=3e-1
    else:
        message('Nothing defined here yet... check the regions')
    
    beam11 = Beam.from_fits_header(fits.getheader(OneOneFile))
    beam22 = Beam.from_fits_header(fits.getheader(TwoTwoFile))
    cube11sc = SpectralCube.read(OneOneFile)
    cube22sc = SpectralCube.read(TwoTwoFile)
    xarr11 = SpectroscopicAxis( cube11sc.spectral_axis,
                             refX=freq11,
                             velocity_convention='radio')
    xarr22 = SpectroscopicAxis( cube22sc.spectral_axis,
                             refX=freq22,
                             velocity_convention='radio')

    #errmap11 = fits.getdata(RMSFile)
    vcube11sc = cube11sc.with_spectral_unit(u.km/u.s, rest_value=freq11, velocity_convention='radio')
    vcube22sc = cube22sc.with_spectral_unit(u.km/u.s, rest_value=freq22, velocity_convention='radio')   

    snr = vcube11sc.filled_data[:].value/rms
    peaksnr = np.max(snr,axis=0)
    #rms = np.nanmedian(errmap11)
    errmap11 = np.full( snr.shape[1:], rms)


    planemask = (peaksnr>snr_min) # *(errmap11 < 0.15)
    planemask = remove_small_objects(planemask,min_size=40)
    planemask = opening(planemask,disk(1))
    #planemask = (peaksnr>20) * (errmap11 < 0.2)

    mask = (snr>3)*planemask
    maskcube = vcube11sc.with_mask(mask.astype(bool))
    slab = maskcube.spectral_slab( vmax*u.km/u.s, vmin*u.km/u.s)
    w11=slab.moment( order=0, axis=0).value
    peakloc = np.nanargmax(w11)
    ymax,xmax = np.unravel_index(peakloc,w11.shape)
    moment1 = slab.moment( order=1, axis=0).value
    moment2 = (slab.moment( order=2, axis=0).value)**0.5
    moment2[np.isnan(moment2)]=0.2
    moment2[moment2<0.2]=0.2
    maskmap = w11>0.5
    # PySpecKit cube for NH3(1,1)
    cube11 = pyspeckit.Cube(cube=vcube11sc,maskmap=planemask, xarr=xarr11)
    #cube11 = pyspeckit.Cube(file=OneOneFile,maskmap=planemask)#, xarr=xarr11)
    cube11.cube *= beam11.jtok( freq11)
    cube11.unit="K"
    # PySpecKit cube for NH3(2,2)
    cube22 = pyspeckit.Cube(cube=vcube22sc,maskmap=planemask, xarr=xarr22)
    #cube22 = pyspeckit.Cube(file=TwoTwoFile,maskmap=planemask)#, xarr=xarr22)
    cube22.cube *= beam22.jtok( freq22)
    cube22.unit="K"
#    cubes = pyspeckit.CubeStack([cube11,cube22],maskmap=planemask)
#    cubes.unit="K"
    guesses = np.zeros((6,)+cube11.cube.shape[1:])
    moment1[moment1<vmin] = vmin+0.2
    moment1[moment1>vmax] = vmax-0.2
    guesses[0,:,:] = 12                    # Kinetic temperature 
    guesses[1,:,:] = 8                     # Excitation  Temp
    guesses[2,:,:] = 14.5                  # log(column)
    guesses[3,:,:] = moment2  # Line width / 5 (the NH3 moment overestimates linewidth)               
    guesses[4,:,:] = moment1  # Line centroid              
    guesses[5,:,:] = 0.5                   # F(ortho) - ortho NH3 fraction (fixed)
    if do_plot:
        import matplotlib.pyplot as plt
        plt.imshow( w11, origin='lower')
        plt.show()
    F=False
    T=True
    print('start fit')
    cube11.fiteach(fittype='ammonia',  guesses=guesses,
                  integral=False, verbose_level=3, 
                  fixed=[T,F,F,F,F,T], signal_cut=2,
                  limitedmax=[F,F,F,F,T,T],
                  maxpars=[0,0,0,0,vmax,1],
                  limitedmin=[T,T,T,T,T,T],
                  minpars=[5,2.8,12.0,0,vmin,0],
                  start_from_point=(xmax,ymax),
                  use_neighbor_as_guess=True, 
                  position_order = 1/peaksnr,
                  errmap=errmap11, multicore=multicore)

    #fitcubefile = fits.PrimaryHDU(data=np.concatenate([cubes.parcube,cubes.errcube]), header=cubes.header)
    fitcubefile = fits.PrimaryHDU(data=np.concatenate([cube11.parcube,cube11.errcube]), header=cube11.header)
    fitcubefile.header.update('PLANE1','TKIN')
    fitcubefile.header.update('PLANE2','TEX')
    fitcubefile.header.update('PLANE3','COLUMN')
    fitcubefile.header.update('PLANE4','SIGMA')
    fitcubefile.header.update('PLANE5','VELOCITY')
    fitcubefile.header.update('PLANE6','FORTHO')
    fitcubefile.header.update('PLANE7','eTKIN')
    fitcubefile.header.update('PLANE8','eTEX')
    fitcubefile.header.update('PLANE9','eCOLUMN')
    fitcubefile.header.update('PLANE10','eSIGMA')
    fitcubefile.header.update('PLANE11','eVELOCITY')
    fitcubefile.header.update('PLANE12','eFORTHO')
    fitcubefile.header.update('CDELT3',1)
    fitcubefile.header.update('CTYPE3','FITPAR')
    fitcubefile.header.update('CRVAL3',0)
    fitcubefile.header.update('CRPIX3',1)
    fitcubefile.writeto("{0}_parameter_maps.fits".format(region),clobber=True)

def test_cube():
    import numpy as np
    from spectral_cube import SpectralCube
    import pyspeckit
    import astropy.units as u

    freq11= 23.6944955*u.GHz
    OneOneFile = 'fits_files/IRAS03282_Per5/IRAS03282_NH3_11.fits'
    cube11sc = SpectralCube.read(OneOneFile)
    vcube11sc = cube11sc.with_spectral_unit(u.km/u.s, rest_value=freq11, velocity_convention='radio')
    print cube11sc
    print vcube11sc

    snr = vcube11sc.filled_data[:].value/rms
    peaksnr = np.max(snr,axis=0)
    #rms = np.nanmedian(errmap11)
    errmap11 = np.full( snr.shape[1:], rms)


    planemask = (peaksnr>snr_min) # *(errmap11 < 0.15)
    planemask = remove_small_objects(planemask,min_size=40)
    planemask = opening(planemask,disk(1))

    from pyspeckit.spectrum.units import SpectroscopicAxis
    xarr = SpectroscopicAxis( cube11sc.spectral_axis,
                             refX=freq11,
                             velocity_convention='radio')

    cube11  = pyspeckit.Cube(cube=cube11sc)
    vcube11 = pyspeckit.Cube(cube=vcube11sc, xarr=xarr)
    cube11.unit="K"
    vcube11.unit="K"

    guesses = np.zeros((6,)+cube11.cube.shape[1:])
    guesses[0,:,:] = 12                    # Kinetic temperature 
    guesses[1,:,:] = 8                     # Excitation  Temp
    guesses[2,:,:] = 14.5                  # log(column)
    guesses[3,:,:] = 0.4  # Line width / 5 (the NH3 moment overestimates linewidth)               
    guesses[4,:,:] = 8.0  # Line centroid              
    guesses[5,:,:] = 0.5                   # F(ortho) - ortho NH3 fraction (fixed)
    F=False
    T=True
    print('start fit')
    cube11.fiteach(fittype='ammonia',  guesses=guesses,
                  integral=False, verbose_level=3, 
                  fixed=[T,F,F,F,F,T], signal_cut=2,
                  limitedmax=[F,F,F,F,T,T],
                  maxpars=[0,0,0,0,9.0,1],
                  limitedmin=[T,T,T,T,T,T],
                  minpars=[5,2.8,12.0,0,6.0,0],
                  start_from_point=(243,243),
                  use_neighbor_as_guess=True)

    print('start fit')
    vcube11.fiteach(fittype='ammonia',  guesses=guesses,
                  integral=False, verbose_level=3, 
                  fixed=[T,F,F,F,F,T], signal_cut=2,
                  limitedmax=[F,F,F,F,T,T],
                  maxpars=[0,0,0,0,9.0,1],
                  limitedmin=[T,T,T,T,T,T],
                  minpars=[5,2.8,12.0,0,6.0,0],
                  start_from_point=(243,243),
                  use_neighbor_as_guess=True)
