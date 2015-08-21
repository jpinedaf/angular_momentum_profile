from spectral_cube import SpectralCube

import astropy.units as u
freq11= 23.6944955*u.GHz

def cube_w11(region='IC348'):
    if region == 'IC348':
        OneOneFile = 'IC348mm/IC348mm-11_cvel_clean_rob05.fits'
        TwoTwoFile = 'IC348mm/IC348mm-11_cvel_clean_rob05.fits'
        vmin=7.4
        vmax=10.0
    elif region == 'IRAS03282':
        OneOneFile = 'IRAS03282/IRAS03282-11_cvel_clean_rob05.fits'
        TwoTwoFile = 'IRAS03282/IRAS03282-11_cvel_clean_rob05.fits'
        vmin=6.0
        vmax=8.5
    elif region == 'L1451mm':
        OneOneFile = 'L1451mm/L1451MM-11_cvel_clean_rob05.fits'
        TwoTwoFile = 'L1451mm/L1451MM-11_cvel_clean_rob05.fits'
        vmin=3.2
        vmax=4.9
    cube = SpectralCube.read(OneOneFile)
    vcube = cube.with_spectral_unit(u.km/u.s, rest_value=freq11, velocity_convention='radio')
    slab = vcube.spectral_slab( vmax*u.km/u.s, vmin*u.km/u.s)
    w11=slab.moment( order=0, axis=0)
    w11.write(OneOneFile.replace('.fits','_w11.fits') )