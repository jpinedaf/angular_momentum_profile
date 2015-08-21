from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import aplpy
import numpy as np

file_in=['IRAS03282/IRAS03282','IC348mm/IC348mm','L1451mm/L1451mm']
Vlsr_ext='_Vlsr_snr5.0_v1.fits'
W11_ext='-11_cvel_clean_rob05_w11.fits'
label=['IRAS03282','HH211','L1451-mm']
ra=np.array( [03+(31+20.94/60.)/60., 03+(43+56.52/60.)/60., 03+(25+10.21/60.)/60.]) * 15.
dec=np.array([30+(45+30.30/60.)/60., 32+(00+52.80/60.)/60., 30+(23+55.30/60.)/60.])
w_rms=[ 1.6, 1.9, 9e-1]
out_pa=[ 122., 116.6, 10.]

rad=45./3600.


cmap_w11='hot'
cmap_w11='Blues_r'
cmap_Vlsr='RdYlBu_r'
label_color='white'
color_sp='#377eb8'
xpos_label=0.05; ypos_label=0.9

xsize=8; ysize=5
dx_pos=3./xsize; dy_pos=3./ysize; x0_pos=0.175; y0_pos=0.1


subplot=[x0_pos, y0_pos, dx_pos, dy_pos]
subplot2=[x0_pos+dx_pos, y0_pos, dx_pos, dy_pos]
subplot_bar =[x0_pos+0.1*dx_pos, y0_pos+0.95*dy_pos, 0.8*dx_pos, 0.05*dy_pos]
subplot2_bar=[x0_pos+1.1*dx_pos, y0_pos+0.95*dy_pos, 0.8*dx_pos, 0.05*dy_pos]

w_min=[-1,-1,-1]
w_max=[20,18,14]

v_min=[6.8,8.8,3.9]
v_max=[7.5,9.5,4.4]

w_ticks=[ [0,5,10,15,20], [0,4.5,9,13.5,18], [0,3.5,7,10.5,14]]
v_ticks=[ [6.8, 7.15, 7.5], [8.8, 9.15, 9.5], [3.9, 4.15, 4.4]]

c_lev=np.arange(5,20,2)

for i in range(len(file_in)):
    # file_i=file_in[i]
    file_v=file_in[i]+Vlsr_ext
    file_w_raw=file_in[i]+W11_ext
    file_w='test.fits'

    w11, hd_w11=fits.getdata( file_w_raw, header=True)
    fits.writeto(file_w, w11*1e3, hd_w11, clobber=True)

    figure = plt.figure(figsize=(xsize, ysize))

    fig0=aplpy.FITSFigure( file_w, subplot=subplot, figure=figure)
    fig1=aplpy.FITSFigure( file_v, subplot=subplot2, figure=figure)
    fig0.show_colorscale( cmap=cmap_w11, vmin=w_min[i], vmax=w_max[i])
    fig1.show_colorscale( cmap=cmap_Vlsr,vmin=v_min[i], vmax=v_max[i])

    fig0.show_contour( file_w, levels=w_rms[i]*c_lev, colors='black')
    fig1.show_contour( file_w, levels=w_rms[i]*c_lev, colors='gray')

    fig0.recenter(ra[i], dec[i], radius=rad)
    fig1.recenter(ra[i], dec[i], radius=rad)
    fig0.show_markers(ra[i], dec[i],  marker='*', alpha=0.9, layer='lay_yso',  c='red', zorder=50, s=80)
    fig1.show_markers(ra[i], dec[i],  marker='*', alpha=0.9, layer='lay_yso',  c='black', zorder=50, s=80)
    fig0.add_label( 0.05, 0.9, label[i], relative=True, horizontalalignment='left', color='red')
    fig1.add_label( 0.05, 0.9, label[i], relative=True, horizontalalignment='left')
    fig0.add_beam(color='red')
    fig1.add_beam()
    fig0.ticks.set_minor_frequency(4)
    fig1.ticks.set_minor_frequency(4)
    fig0.tick_labels.set_xformat('hh:mm:ss')
    fig0.tick_labels.set_yformat('dd:mm:ss')
    fig1.tick_labels.set_xformat('hh:mm:ss')
    fig1.tick_labels.set_yformat('dd:mm:ss')
    # Scalebar
    distance=250.
    ang_sep = 5e3/distance / 3600. # from arcsec -> deg
    fig0.add_scalebar(ang_sep)
    fig0.scalebar.set(color='red')
    fig0.scalebar.set_label('5,000 au')
    fig1.add_scalebar(ang_sep)
    fig1.scalebar.set(color='black')
    fig1.scalebar.set_label('5,000 au')
    #fig.set_nan_color('gray')
    fig1.set_nan_color('#D3D3D3')
    fig0.ticks.set_color('black')
    fig1.ticks.set_color('black')

    fig0.ticks.set_minor_frequency(4)
    fig0.tick_labels.set_xformat('hh:mm:ss')
    fig0.tick_labels.set_yformat('dd:mm:ss')

    #Colorbars 
    # Panel a)
    fig0.add_colorbar()
    fig0.colorbar.set_location('top')
    fig0.colorbar.set_box( subplot_bar, box_orientation='horizontal')
    fig0.colorbar.set_axis_label_text('Flux (mJy beam$^{-1}$ km s$^{-1}$)')
    fig0.colorbar.set_ticks( w_ticks[i])
    # Panel b)
    fig1.add_colorbar()
    fig1.colorbar.set_location('top')
    fig1.colorbar.set_box( subplot2_bar, box_orientation='horizontal')
    fig1.colorbar.set_axis_label_text('V$_{LSR}$ (km s$^{-1}$)')
    fig1.colorbar.set_ticks( v_ticks[i])




    fig1.hide_axis_labels()
    fig1.hide_tick_labels()
    # Add outflow arrows
    dr_arrow=20*(u.arcsec).to(u.deg)
    fig1.show_arrows( ra[i], dec[i], dr_arrow*np.sin(np.deg2rad(out_pa[i])), dr_arrow*np.cos(np.deg2rad(out_pa[i])), color='blue', zorder=51)
    fig1.show_arrows( ra[i], dec[i], dr_arrow*np.sin(np.deg2rad(180+out_pa[i])), dr_arrow*np.cos(np.deg2rad(180+out_pa[i])), color='red', zorder=52)


    figure.savefig( 'figures/'+label[i]+'_w11_Vlsr.pdf', tight_layout=True)
    plt.close(figure)
