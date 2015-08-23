import numpy as np
from astropy import units as u
from astropy.io import fits
import astropy.wcs as wcs

#import aplpy
import matplotlib.pyplot as plt
import aplpy

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

def calculate_j( file_v, distance=250.*u.pc, do_plot=True, ra0=0.0, dec0=0.0, angle=0.0*u.deg):
    """ 
    Function that calculates the specific angular momentum. Result is distance (in pc) and 

    Parameters
    -----------
    ra0 and dec0: These are the center for the calculation of the distance, in degrees. 
    The default is to use the image's center as the calculation center.

    distance: Distance to the source, in pc.

    angle: Position angle in which to project the rotation. Usually this is determined from the 
    outflow orientation. This is measured from North due East (counter-clockwise from North). 

    do_plot: Boolean. If True it will display the relative velocity, specific angular momentum and 
    projected separation maps. Great for debuging/understanding your data.
    """
    # Loads FITS file with centroid velocity
    hdu =fits.open(file_v)
    hd =hdu[0].header
    # This is a hack to get the units from the header
    # What is the proper way?
    if hd['BUNIT'] == 'km/s':
        v_unit=u.km/u.s
    if hd['BUNIT'] == 'm/s':
        v_unit=u.m/u.s
    #
    w = wcs.WCS(hd, hdu)
    v  =np.squeeze(hdu[0].data)*v_unit
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
    # Uses distance to determine pixel size in au, and au -> pc conversion
    #d_pix=abs(hd['CDELT1']*(u.deg.to(u.arcsec))) * distance
    #
    # Pixel size, convert pixel size from degrees to radians and then 
    # multiply by the distance (with units)
    d_pix=np.abs(hd['CDELT1'])*u.deg.to(u.rad) * distance
    au_pc=(u.au).to(u.pc)
    # # Determine beam size in pixel units
    # beam=np.abs( hd['BMAJ']/hd['CDELT1'] )
    # Calculate relative distance and velocity
    xv, yv = np.meshgrid(np.arange(0,size[1]), np.arange(0,size[0]), sparse=False, indexing='xy')
    # for each position we calculate the distance and the angle with respect to the center
    r_dist = np.sqrt( (xv - xc)**2 + (yv - yc)**2 ) * d_pix.to(u.au)
    theta_dist = np.arctan( (yv - yc)/(xv - xc))*u.rad
    # The projected distance is the distance multiplied by the sin of the 
    # angle between outflow PA and position angle
    dist= r_dist * np.abs( np.sin(90*u.deg+angle-theta_dist))
    # dist= r_dist * cos(np.sqrt( ( (xv - xc)*np.cos(angle) )**2 + ( (yv - yc)*np.sin(angle) )**2)
    # Calculate the relative velocity with respect to the pixel the closest to the center
    v_rel=v - v[ int(yc), int(xc)]
    # j = np.abs(v_rel*dist*d_pix*au_pc)
    j = np.abs(v_rel*dist)
    # 
    # gd= (np.isfinite(v)) & (dist<sep_max)  & (dist>0.5*beam)
    #
    # dist_gd=dist[gd]*d_pix
    # j_gd=j[gd]
    if do_plot:
        # Add a quick plot
        fig = plt.figure(figsize=(12, 5))
        dx_fig=3/12.; dy_fig=4/5.
        x0_pos=0.1
        y0_pos=0.1
        subplot0=[x0_pos         ,y0_pos,dx_fig,dy_fig]
        subplot1=[x0_pos+  dx_fig,y0_pos,dx_fig,dy_fig]
        subplot2=[x0_pos+2*dx_fig,y0_pos,dx_fig,dy_fig]
        # 
        fig0= aplpy.FITSFigure( fits.PrimaryHDU(v_rel.to(u.km/u.s).value, hd), hdu=0, figure=fig, subplot=subplot0)
        fig1= aplpy.FITSFigure( fits.PrimaryHDU(j.to(u.pc*u.km/u.s).value, hd), hdu=0, figure=fig, subplot=subplot1)
        fig2= aplpy.FITSFigure( fits.PrimaryHDU(dist.to(u.au).value, hd), hdu=0, figure=fig, subplot=subplot2)
        # show figure colorscale
        fig0.recenter(ra0, dec0, radius=30.*u.arcsec.to(u.deg))
        fig1.recenter(ra0, dec0, radius=30.*u.arcsec.to(u.deg))
        fig2.recenter(ra0, dec0, radius=30.*u.arcsec.to(u.deg))
        fig0.add_beam()
        fig1.add_beam()
        fig2.add_beam()
        fig0.show_colorscale()
        fig1.show_colorscale()
        fig2.show_colorscale(vmin=0, vmax=(30.*distance.to(u.pc)).value )
        # YSO marker
        fig0.show_markers(ra0,  dec0,  marker='*', alpha=0.7, layer='lay_yso',  c='orange', s=40)
        fig1.show_markers(ra0,  dec0,  marker='*', alpha=0.7, layer='lay_yso',  c='orange', s=40)
        fig2.show_markers(ra0,  dec0,  marker='*', alpha=0.7, layer='lay_yso',  c='orange', s=40)
        # Hide labels
        fig1.tick_labels.hide()
        fig2.tick_labels.hide()
        fig1.axis_labels.hide()
        fig2.axis_labels.hide()
        fig0.add_colorbar()
        fig0.colorbar.set_location('top')
        fig1.add_colorbar()
        fig1.colorbar.set_location('top')
        fig2.add_colorbar()
        fig2.colorbar.set_location('top')
        # Plot Outflow axis
        dr_arrow=15*u.arcsec.to(u.deg)
        fig0.show_arrows(ra0, dec0, np.sin( angle)*dr_arrow, np.cos( angle)*dr_arrow, color='blue')
        fig0.show_arrows(ra0, dec0, np.sin(180*u.deg+angle)*dr_arrow, np.cos(180*u.deg+angle)*dr_arrow, color='red')
        fig1.show_arrows(ra0, dec0, np.sin( angle)*dr_arrow, np.cos( angle)*dr_arrow, color='blue')
        fig1.show_arrows(ra0, dec0, np.sin(180*u.deg+angle)*dr_arrow, np.cos(180*u.deg+angle)*dr_arrow, color='red')

        ang_sep = 1e4/(distance/u.pc) / 3600.
        fig1.add_scalebar(ang_sep.value)
        fig1.scalebar.set(color='black')
        fig1.scalebar.set_label('10,000 au')
        
    return dist, j
    #return mean_ang_mom( dist_gd, j_gd, beam*d_pix)
