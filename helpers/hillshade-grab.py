# Kevin Gauld 2024
#
# Provides infrastructure for testing hillshade generation.

import os, shutil
from osgeo import gdal
import cv2 as cv

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
                outputBoundsSRS='+proj=longlat +a=1737400 +b=1737400 +no_defs'
            ))

	#Reproject topography to sinusoidal to increase chances of a match. Can't get the $clon variable in the proj4 syntax, so sending it to temp file, then executing that.
    gdal.Warp(f'{workdir}/topo_sinu.tif',f'{workdir}/topo_clip.tif',
            options=gdal.WarpOptions(
                format='GTiff',
                dstSRS=f'+proj=sinu +lon_0={clon} +x_0=0 +y_0=0 +a=1737400 +b=1737400 +no_defs'
            ))

    def makeHSH(alt, az, clon, clat):
        hshfn = f'hillshade_{clon:03.0f}_{clat:03.0f}_{alt:03.0f}_{az:03.0f}'
        gdal.DEMProcessing(
            f"{outdir}/{hshfn}.tif", 
            f"{workdir}/topo_sinu.tif", 
            "hillshade",
            options=gdal.DEMProcessingOptions(
                format='GTiff',
                alg='ZevenbergenThorne',
                azimuth=az,
                altitude=alt
            ))
        # cv.imwrite(f"{outdir}/{hshfn}.tif", 
        #            cv.imread(f"{workdir}/{hshfn}.tif", cv.IMREAD_ANYDEPTH))
    
    for (alt,az) in configs:
        makeHSH(alt, az, clon, clat)
    
    shutil.rmtree(workdir)
    

if __name__ == "__main__":
    # configs = [(x,315) for x in range(0,182,2)]
    # get_hillshade(6.07941, 81.60264,
    #               11.72785, 87.45002, configs)
    
    configs = [(45, 315)]
    latbnds = [(-5,5),(15,25),(35,60), (50,80), (80,89.9)]
    out_fn = 'hshtestdir'
    for k in latbnds:
        get_hillshade(k[0], 80,
                      k[1], 90, configs, outdir=out_fn)
    get_hillshade(-79.93014228527048, 344.57656629590826-360,
                  -65.92539924277551, 352.6450695389247-360, configs, outdir=out_fn)
    # 348.6108179174165,344.57656629590826,352.6450695389247,
    # -72.927770764023,-79.93014228527048,-65.92539924277551,

    get_hillshade(-81.7889287904793,-13.756994328720385,
                  -64.06661273756666,-9.021369836446231, [(44,315)], outdir=out_fn)
    
    # -11.389182082583307,-13.756994328720385,-9.021369836446231,
    # -72.92777076402298,-81.7889287904793,-64.06661273756666,
    

