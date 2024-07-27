##This code takes M3 data and uses feature matching to register it to LOLA/Kaguya topography within 60 degrees of the equator, and LOLA data poleward of 60 degrees.
##The first half of the code prepares the Radiance and Topography for matching
##The second half of the code matches an averaged Radiance file with a hillshade map generated with similar illumination conditions.
##If the first match does not work, the code iteratively creates hillshades with different lighting until a match is successful.

##The code uses M3 data that have been clipped to 2677 rows to be more similar to HVM3 data. The data are all in Data_M3/. The middle 2677 rows were extracted for each M3 product.

#To execute the code, run the script with the M3 image ID as the one argument, i.e. "./LTB_FeatureMatching_M3.sh M3G20090207T090807" (no quotes)

##Jay Dickson (jdickson@umn.edu)
##August, 2023

#The code expects one argument: the M3 image ID.


import os, shutil
from osgeo import gdal
import cv2 as cv

def get_hillshade(minlat, minlon, maxlat, maxlon, configs):
    clon = (minlon+maxlon)/2
    clat = (minlat+maxlat)/2

    #Input topography file to be used to generate hillshade
    inlola = "Topography/LunarTopography_60mpx.tif"

    #Create a directory to run all of the processes/make files in.
    workdir=f'hshwork'
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    os.mkdir(workdir)

    outdir=f'hshout_{minlat:.2f}_{minlon:.2f}_{maxlat:.2f}_{maxlon:.2f}'
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.mkdir(outdir)


    #Clip the global topography to the bounding box. The LOLA DEM is 60 m/px and does not need to be resampled for this procedure. Only clip.
    gdal.Warp(f'{workdir}/topo_clip.tif', inlola,
            options = gdal.WarpOptions(
                format='GTiff',
                outputBounds=(minlon,minlat,maxlon,maxlat),
                outputBoundsSRS='+proj=longlat +a=1737400 +b=1737400 +no_defs'
            ))

	#Reproject topography to sinusoidal to increase chances of a match. Can't get the $clon variable in the proj4 syntax, so sending it to temp file, then executing that.
    gdal.Warp(f'{workdir}/topo_sinu.tif',f'{workdir}/topo_clip.tif',
            options=gdal.WarpOptions(
                format='GTiff',
                dstSRS=f'+proj=sinu +lon_0={clon} +x_0=0 +y_0=0 +a=1737400 +b=1737400 +no_defs'
            ))

    def makeHSH(alt, az):
        gdal.DEMProcessing(f"{workdir}/hillshade_{alt:03.0f}_{az:03.0f}.tif", f"{workdir}/topo_clip.tif", "hillshade",
                    options=gdal.DEMProcessingOptions(
                        format='GTiff',
                        alg='ZevenbergenThorne',
                        azimuth=az,
                        altitude=alt
                    ))
        cv.imwrite(f"{outdir}/hillshade_{alt:03.0f}_{az:03.0f}.tif", 
                   cv.imread(f"{workdir}/hillshade_{alt:03.0f}_{az:03.0f}.tif", cv.IMREAD_ANYDEPTH))
    
    for (alt,az) in configs:
        makeHSH(alt, az)
    
    shutil.rmtree(workdir)
    

if __name__ == "__main__":
    configs = [(x,315) for x in range(0,182,2)]
    get_hillshade(6.07941, 81.60264,
                  11.72785, 87.45002, configs)

