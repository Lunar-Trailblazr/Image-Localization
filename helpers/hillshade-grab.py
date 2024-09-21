# Kevin Gauld 2024
#
# Provides infrastructure for testing hillshade generation.

import os, shutil
from osgeo import gdal, gdalconst
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob

def get_hillshade(minlat, minlon, maxlat, maxlon, configs, outdir=None):
    clon = (minlon+maxlon)/2
    clat = (minlat+maxlat)/2

    #Input topography file to be used to generate hillshade
    inlola = "Topography/LunarTopography_60mpx.tif"

    #Create a directory to run all of the processes/make files in.
    workdir=f'hshwork'
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    os.mkdir(workdir)

    if outdir is None:
        outdir=f'hshout_{minlat:.2f}_{minlon:.2f}_{maxlat:.2f}_{maxlon:.2f}'
        if os.path.isdir(outdir):
            shutil.rmtree(outdir)
        os.mkdir(outdir)


    #Clip the global topography to the bounding box. The LOLA DEM is 60 m/px and does not need to be resampled for this procedure. Only clip.
    gdal.Warp(f'{workdir}/topo_clip.tif', inlola,
            options = gdal.WarpOptions(
                format='GTiff',
                outputBounds=(minlon,
                              minlat,
                              maxlon,
                              maxlat),
                outputBoundsSRS=f'+proj=longlat +a=1737400 +b=1737400 +no_defs'
            ))
    
    # Reproject the topography to sinusoidal
    gdal.Warp(f'{workdir}/topo_sinu.tif',f'{workdir}/topo_clip.tif',
            options=gdal.WarpOptions(
                format='GTiff',
                dstSRS=f'+proj=sinu +lon_0={clon} +x_0=0 +y_0=0 +a=1737400 +b=1737400 +no_defs'
            ))
    
    sinu_img = cv.imread(f'{workdir}/topo_sinu.tif', cv.IMREAD_ANYDEPTH)
    plt.figure(figsize=(10,10))
    plt.hist(sinu_img.flatten(), 255)
    plt.savefig(f'{workdir}/sinu_hist.png')
    print(np.max(sinu_img))
    sinu_mask = cv.inRange(sinu_img, 32767, 32767)
    sinu_mask = cv.dilate(sinu_mask, np.ones((5,5)), iterations=1)
    

    def makeHSH(alt, az, clon, clat, zfactor=100000):
        hshfn = f'hillshade_{clon:03.0f}_{clat:03.0f}_{alt:03.0f}_{az:03.0f}_{zfactor}'
        reffile = f"{workdir}/topo_sinu.tif" #f"{workdir}/topo_ortho.tif" if clat < 45 else f"{workdir}/topo_stere.tif"
        gdal.DEMProcessing(
            f"{outdir}/{hshfn}.tif", 
            reffile,#f"{workdir}/topo_sinu.tif", 
            "hillshade",
            options=gdal.DEMProcessingOptions(
                format='GTiff',
                alg='ZevenbergenThorne',
                azimuth=az,
                altitude=90-alt,
                #scale=30323,
                zFactor=zfactor
            ))
        return f"{outdir}/{hshfn}.tif"
    
    ofns = []
    for zf in np.arange(1, 15, 1):
        for (alt,az) in configs:
            ofns.append(makeHSH(alt, az, clon, clat, zfactor=zf))
    
    return ofns, sinu_mask
    # make_backplane(minlat, minlon, maxlat, maxlon, cv.imread(f'{workdir}/topo_sinu.tif', cv.IMREAD_ANYDEPTH))
    # make_backplane(minlat, minlon, maxlat, maxlon, 'hshtestdir/hillshade_-11_-73_020_315.tif')
    # shutil.rmtree(workdir)
    

if __name__ == "__main__":
    # configs = [(x,315) for x in range(0,182,2)]
    # get_hillshade(6.07941, 81.60264,
    #               11.72785, 87.45002, configs)
    
    configs = [(10,315)]
    latbnds = [(-5,5),(15,25),(35,60), (50,80), (80,89.9)]
    out_fn = 'hshtestdir'
    # for k in latbnds:
    #     get_hillshade(k[0], 80,
    #                   k[1], 90, configs, outdir=out_fn)
    ofns1, mask10 = get_hillshade(-79.93014228527048, 344.57656629590826-360,
                  -65.92539924277551, 352.6450695389247-360, [(10,315)], outdir=out_fn)
    ofns2, mask45 = get_hillshade(-79.93014228527048, 344.57656629590826-360,
                  -65.92539924277551, 352.6450695389247-360, [(45,315)], outdir=out_fn)
    ofns3, mask80 = get_hillshade(-79.93014228527048, 344.57656629590826-360,
                  -65.92539924277551, 352.6450695389247-360, [(80,315)], outdir=out_fn)
    plt.figure(figsize=(10,10))
    imgs10 = []
    for k in ofns1:
        im = cv.imread(k, cv.IMREAD_ANYDEPTH)
        imgs10.append(im)
        plt.hist(im[np.where(im>1)], 255, label=k.split('_')[-1].split('.')[0], alpha=0.3)
    plt.legend()
    plt.savefig(f'{out_fn}/output_hist_10.png')

    plt.figure(figsize=(10,10))
    imgs45 = []
    for k in ofns2:
        im = cv.imread(k, cv.IMREAD_ANYDEPTH)
        imgs45.append(im)
        plt.hist(im[np.where(im>1)], 255, label=k.split('_')[-1].split('.')[0], alpha=0.3)
    plt.legend()
    plt.savefig(f'{out_fn}/output_hist_45.png')

    plt.figure(figsize=(10,10))
    imgs80 = []
    for k in ofns3:
        im = cv.imread(k, cv.IMREAD_ANYDEPTH)
        imgs80.append(im)
        plt.hist(im[np.where(im>1)], 255, label=k.split('_')[-1].split('.')[0], alpha=0.3)
    plt.legend()
    plt.savefig(f'{out_fn}/output_hist_80.png')

    print('outputhists done')

    im_mean_10 = np.mean(imgs10, axis=0)
    im_mean_45 = np.mean(imgs45, axis=0)
    im_mean_80 = np.mean(imgs80, axis=0)

    im_med_10 = np.median(imgs10, axis=0)
    im_med_45 = np.median(imgs45, axis=0)
    im_med_80 = np.median(imgs80, axis=0)
    
    plt.figure(figsize=(10,10))
    plt.hist(im_mean_10[np.where(im_mean_10>1)],255,alpha=0.3,label='10deg')
    plt.hist(im_mean_45[np.where(im_mean_45>1)],255,alpha=0.3,label='45deg')
    plt.hist(im_mean_80[np.where(im_mean_80>1)],255,alpha=0.3,label='80deg')
    plt.legend()
    plt.savefig(f'{out_fn}/hist_means.png')

    plt.figure(figsize=(10,10))
    plt.hist(im_med_10[np.where(im_mean_10>1)],255,alpha=0.3,label='10deg')
    plt.hist(im_med_45[np.where(im_mean_45>1)],255,alpha=0.3,label='45deg')
    plt.hist(im_med_80[np.where(im_mean_80>1)],255,alpha=0.3,label='80deg')
    plt.legend()
    plt.savefig(f'{out_fn}/hist_meds.png')

    cv.imwrite(f'{out_fn}/mean_img_10.tif', im_mean_10.astype(np.uint8))
    cv.imwrite(f'{out_fn}/mean_img_45.tif', im_mean_45.astype(np.uint8))
    cv.imwrite(f'{out_fn}/mean_img_80.tif', im_mean_80.astype(np.uint8))

    cv.imwrite(f'{out_fn}/med_img_10.tif', im_med_10.astype(np.uint8))
    cv.imwrite(f'{out_fn}/med_img_45.tif', im_med_45.astype(np.uint8))
    cv.imwrite(f'{out_fn}/med_img_80.tif', im_med_80.astype(np.uint8))

    # 348.6108179174165,344.57656629590826,352.6450695389247,
    # -72.927770764023,-79.93014228527048,-65.92539924277551,

    # get_hillshade(-81.7889287904793,-13.756994328720385,
    #               -64.06661273756666,-9.021369836446231, [(44,315)], outdir=out_fn)
    
    # -11.389182082583307,-13.756994328720385,-9.021369836446231,
    # -72.92777076402298,-81.7889287904793,-64.06661273756666,
    

