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
        OneOneFile = 'IC348mm/IC348mm-11_cvel_clean_rob05.fits'
        TwoTwoFile = 'IC348mm/IC348mm-11_cvel_clean_rob05.fits'
        vmin=7.4
        vmax=10.0
        rms=3e-3
    elif region == 'IRAS03282':
        OneOneFile = 'IRAS03282/IRAS03282-11_cvel_clean_rob05.fits'
        TwoTwoFile = 'IRAS03282/IRAS03282-11_cvel_clean_rob05.fits'
        vmin=6.0
        vmax=8.5
        rms=2.5e-3
    elif region == 'L1451mm':
        OneOneFile = 'L1451mm/L1451MM-11_cvel_clean_rob05.fits'
        TwoTwoFile = 'L1451mm/L1451MM-11_cvel_clean_rob05.fits'
        vmin=3.2
        vmax=4.9
        rms=1.6e-3
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
                  limitedmax=[F,F,T,F,T,T],
                  maxpars=[0,0,17.0,0,vmax,1],
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
    fitcubefile.writeto("{0}_parameter_maps_snr{1}.fits".format(region,snr_min),clobber=True)
    convert_param_file(region=region, snr_min=snr_min)

def convert_param_file(region='L1451mm', snr_min=5.0):
    """
    Updates the fit parameters storage format from cube (v0, one channel per 
    parameter) into a set of files (v1, one FITS per parameter). 
    """

    if region != 'L1451mm' and region != 'IC348' and region != 'IRAS03282':
        return 'error'

    hdu=fits.open("{0}_parameter_maps_snr{1}.fits".format(region,snr_min))
    hd=hdu[0].header
    cube=hdu[0].data
    hdu.close()
    # blank bad pixels
    cube[cube==0]=np.nan
    # Clean header keyword
    rm_key=['NAXIS3','CRPIX3','CDELT3', 'CUNIT3', 'CTYPE3', 'CRVAL3']
    for key_i in rm_key:
        hd.remove(key_i)
    hd['NAXIS']= 2
    hd['WCSAXES']= 2
    # Tkin
    hd['BUNIT']='K'
    param=cube[0,:,:]
    file_out="{0}_Tkin_snr{1}_v1.fits".format(region,snr_min)
    fits.writeto(file_out, param, hd, clobber=True)
    #Tex
    hd['BUNIT']='K'
    param=cube[1,:,:]
    file_out="{0}_Tex_snr{1}_v1.fits".format(region,snr_min)
    fits.writeto(file_out, param, hd, clobber=True)
    # N_NH3
    hd['BUNIT']='cm-2'
    param=cube[2,:,:]
    file_out="{0}_N_NH3_snr{1}_v1.fits".format(region,snr_min)
    fits.writeto(file_out, param, hd, clobber=True)
    # sigma
    hd['BUNIT']='km/s'
    param=cube[3,:,:]
    file_out="{0}_Sigma_snr{1}_v1.fits".format(region,snr_min)
    fits.writeto(file_out, param, hd, clobber=True)
    # Vlsr
    hd['BUNIT']='km/s'
    param=cube[4,:,:]
    file_out="{0}_Vlsr_snr{1}_v1.fits".format(region,snr_min)
    fits.writeto(file_out, param, hd, clobber=True)
    # Fortho
    hd['BUNIT']=''
    param=cube[5,:,:]
    file_out="{0}_Fortho_snr{1}_v1.fits".format(region,snr_min)
    fits.writeto(file_out, param, hd, clobber=True)
    # eTkin
    hd['BUNIT']='K'
    param=cube[6,:,:]
    file_out="{0}_eTkin_snr{1}_v1.fits".format(region,snr_min)
    fits.writeto(file_out, param, hd, clobber=True)
    # eTex
    hd['BUNIT']='K'
    param=cube[7,:,:]
    file_out="{0}_eTex_snr{1}_v1.fits".format(region,snr_min)
    fits.writeto(file_out, param, hd, clobber=True)
    # eN_NH3
    hd['BUNIT']='cm-2'
    param=cube[8,:,:]
    file_out="{0}_eN_NH3_snr{1}_v1.fits".format(region,snr_min)
    fits.writeto(file_out, param, hd, clobber=True)
    # eSigma
    hd['BUNIT']='km/s'
    param=cube[9,:,:]
    file_out="{0}_eSigma_snr{1}_v1.fits".format(region,snr_min)
    fits.writeto(file_out, param, hd, clobber=True)
    # eVlsr
    hd['BUNIT']='km/s'
    param=cube[10,:,:]
    file_out="{0}_eVlsr_snr{1}_v1.fits".format(region,snr_min)
    fits.writeto(file_out, param, hd, clobber=True)
    # eFortho
    hd['BUNIT']=''
    param=cube[11,:,:]
    file_out="{0}_eFortho_snr{1}_v1.fits".format(region,snr_min)
    fits.writeto(file_out, param, hd, clobber=True)
    
