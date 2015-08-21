import numpy as np
from astropy import units as u
from astropy.io import fits
import astropy.wcs as wcs

#import aplpy
import matplotlib.pyplot as plt

def mean_ang_mom( x, y, dx):
    """ Define averaging function
    """
    xmin=np.min(x)
    n_bin=int(np.ceil((np.max(x)-xmin)/dx))
    xbin=np.zeros(n_bin)
    dxbin=np.zeros(n_bin)
    ybin=np.zeros(n_bin)
    dybin=np.zeros(n_bin)
    for i in range(n_bin):
        idx=np.where( (x>xmin+dx*i) & (x<xmin+dx*(i+1)))
        xbin[i] =np.mean(x[idx])
        ybin[i] =np.mean(y[idx])
        dxbin[i]=np.std(x[idx])
        dybin[i]=np.std(y[idx])
    return xbin, ybin, dxbin, dybin

def calculate_j( file_v, distance=250., sep_max=150., do_plot=True, ra0=0.0, dec0=0.0, angle=0.0):
    """ 
    ra0 and dec0 are the center for the calculation of the distance
    """
    # Loads FITS file with centroid velocity
    hdu =fits.open(file_v)
    hd =hdu[0].header
    w = wcs.WCS(hd, hdu)
    v  =np.squeeze(hdu[0].data)
    size=v.shape
    hdu.close()
    if ra0 == dec0 == 0.0:
        xc = size[1]*0.5
        yc = size[0]*0.5
    else:
        if hd['NAXIS'] == 2:
            xc, yc = w.all_world2pix(ra0, dec0, 1)
        else:
            temp = w.all_world2pix([[ra0, dec0, 0]], 1)
            xc, yc=temp[0][0:2]
    # replaces bad data (-999) with NaN
    v[v==-999] = np.nan
    # Uses distance to determine pixel size in au, and au -> pc conversion
    d_pix=abs(hd['CDELT1']*(u.deg.to(u.arcsec))) * distance
    au_pc=(u.au).to(u.pc)
    # Determine beam size in pixel units
    beam=np.abs( hd['BMAJ']/hd['CDELT1'] )
    # Calculate relative distance and velocity
    xv, yv = np.meshgrid(np.arange(0,size[1]), np.arange(0,size[0]), sparse=False, indexing='xy')
    r_dist = np.sqrt( (xv - xc)**2 + (yv - yc)**2 )
    theta_dist = np.arctan( (yv - yc)/(xv - xc))
    dist= r_dist * np.abs( np.sin( angle-theta_dist))
    # dist= r_dist * cos(np.sqrt( ( (xv - xc)*np.cos(angle) )**2 + ( (yv - yc)*np.sin(angle) )**2)
    v_rel=v - v[ int(yc), int(xc)]
    j = np.abs(v_rel*dist*d_pix*au_pc)
    # 
    gd= (np.isfinite(v)) & (dist<sep_max)  & (dist>0.5*beam)
    #
    dist_gd=dist[gd]*d_pix
    j_gd=j[gd]
    if do_plot:
        # Add a quick plot
        xmin=int(np.max([np.floor(xc-sep_max),0]))
        xmax=int(np.min([np.ceil( xc+sep_max),(size[1]-1)]))
        ymin=int(np.max([np.floor(yc-sep_max),0]))
        ymax=int(np.min([np.ceil( yc+sep_max),(size[0]-1)]))
        extent = [xmin-xc, xmax-xc, ymin-yc, ymax-yc]
        print xc, yc, sep_max
        print xmin, xmax, ymin, ymax
        print j[ymin:ymax,xmin:xmax].shape
        print dist[int(xc),int(yc)]
        print extent
        f, axarr = plt.subplots(1,3)
        img0=axarr[0].imshow( v_rel[ymin:ymax,xmin:xmax], origin='lower', extent=extent)
        axarr[0].scatter(0,0, c='r', alpha=0.5)
        axarr[0].arrow(0,0, 0.5*sep_max*np.cos(angle), 0.5*sep_max*np.sin(angle))
        img1=axarr[1].imshow( j[ymin:ymax,xmin:xmax], origin='lower', extent=extent)
        img2=axarr[2].imshow( dist[ymin:ymax,xmin:xmax], origin='lower', extent=extent)
        axarr[1].scatter(0,0, c='r', alpha=0.5)
        axarr[2].scatter(0,0, c='r', alpha=0.5)
        plt.colorbar(img0, orientation='horizontal')
        plt.colorbar(img1, orientation='horizontal')
        plt.colorbar(dist, orientation='horizontal')
    return mean_ang_mom( dist_gd, j_gd, beam*d_pix)
