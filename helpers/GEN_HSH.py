import os, sys, shutil
from PIL import Image
from osgeo import gdal, osr, gdalconst
from osgeo_utils import gdal_calc
import numpy as np
import pandas as pd
from FM_toolkit import *

def run_match(m3id):
    '''
        Runs feature matching for a given m3id

        Returns 3 booleans:
            1) First run -- Is this the first time the m3id was matched, or has it been matched and re-found? True if never run before
            2) File found -- Was the m3id file successfully found? True if file was found
            3) Match succeeded -- Did the match succeed? True if match succeeded
    '''

    # #Check to see if this image has already been matched or attempted to match.
    # work = any(x.startswith(m3id) for x in os.listdir('Results/Worked'))
    # fail = any(x.startswith(m3id) for x in os.listdir('Results/Failed'))
    # if work:
    #     print(f"{m3id} has been run and successfully matched, with output data in Results/Worked/.")
    #     return False, True, True
    # elif fail:
    #     print(f"{m3id} has been run and unsuccessfully matched, with radiance file used in matching in Results/Failed/.")
    #     if f'{m3id}.txt' in os.listdir('Results/Failed'):
    #         return False, False, False
    #     else:
    #         return False, True, False

    #Input topography file to be used to generate hillshade
    inlola = "Topography/LunarTopography_60mpx.tif"

    #Ground resolution of M3 in m/px.
    #This is required for extraction of LOLA data over a reasonable area of the Moon to generate a hillshade for matching.
    m3resolution = 200

    #Define starting and ending bands in M3 for averaging.
    #The code averages M3 bands together to increase the quality of the matching product. Set the start band and end band (in units of band number)
    band_bnds=(71,81)

    #Input directory for radiance data. This is a volume of M3 data that have been clipped to 2677 rows to be more similar to HVM3 data. The middle 2677 rows were extracted for each M3 product.
    m3dir="Data_M3"

    #Create a directory to run all of the processes/make files in.
    workdir=f'{m3id}_work'
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    os.mkdir(workdir)

    #Source files
    inrdn=f"{m3dir}/{m3id}_RDN.IMG"
    inobs=f"{m3dir}/{m3id}_OBS.IMG"
    inloc=f"{m3dir}/{m3id}_LOC.IMG"

    loc_img = gdal.Open(inloc)
    rdn_img = gdal.Open(inrdn)
    obs_img = gdal.Open(inobs)

    # Check to make sure the rdn loc and obs images are all found
    if loc_img is None or rdn_img is None or obs_img is None:
        print(f"M3ID NOT FOUND: {m3id}")
        if os.path.isdir(workdir):
            shutil.rmtree(workdir)
        f = open(f"Results/Failed/{m3id}.txt", "a")
        f.write(f"{m3id} NOT FOUND")
        f.close()
        return True, False, False

    ###Calculate extent for hillshade. This calculates the center lat/lon from the LOC file for M3.
    ###For HVM3, the LOC file will not be available at this stage, so center coordinates should be extracted from the filename, or from a lookup table of targets.

    #Calculate center latitude/longitude
    latband = loc_img.GetRasterBand(2)
    minlat, maxlat = latband.ComputeStatistics(0)[:2]
    clat = (minlat + maxlat)/2

    lonband = loc_img.GetRasterBand(1)
    minlon, maxlon =  lonband.ComputeStatistics(0)[:2]
    clon = (minlon + maxlon)/2

    #Get x and y for clon and clat in meter space 
    projin = osr.SpatialReference()
    projout = osr.SpatialReference()
    projin.SetFromUserInput('+proj=longlat +a=1737400 +b=1737400 +no_defs')
    projout.SetFromUserInput(f'+proj=sinu +lon_0={clon} +x_0=0 +y_0=0 +a=1737400 +b=1737400 +units=m +no_defs')

    in2out_tf = osr.CoordinateTransformation(projin, projout)
    coords = in2out_tf.TransformPoint(clon, clat, 0)
    x, y = coords[:2]

    #Determine the number of lines and samples in the image.
    #These values are multiplied by the resolution to create the bounding box to use to clip LOLA to the approximate area of the image.
    samples, lines = rdn_img.RasterXSize, rdn_img.RasterYSize
    
    #Calculate the width and height of the image in meters using M3's resolution (defined at the beginning of the script)
    fullwidth = samples * m3resolution
    fullheight = lines * m3resolution

    #Set the bounding box. Add a kilometer on all sides for margin (may need more if pointing uncertainty is very high)
    xmin, xmax = x-fullwidth/2-1000, x+fullwidth/2+1000
    ymin, ymax = y-fullheight/2-1000, y+fullheight/2+1000

    #Get x and y in degree space. Parse the lines to define boudning box coordinates.
    out2in_tf = osr.CoordinateTransformation(projout, projin)
    minlon, maxlat = out2in_tf.TransformPoint(xmin, ymax, 0)[:2]
    maxlon = out2in_tf.TransformPoint(xmax, ymax, 0)[0]
    minlat = out2in_tf.TransformPoint(xmin, ymin, 0)[1]

    #Clip the global topography to the bounding box. The LOLA DEM is 60 m/px and does not need to be resampled for this procedure. Only clip.
    gdal.Warp(f'{workdir}/{m3id}_topo.tif', inlola,
            options = gdal.WarpOptions(
                format='GTiff',
                outputBounds=(minlon,minlat,maxlon,maxlat),
                outputBoundsSRS='+proj=longlat +a=1737400 +b=1737400 +no_defs'
            ))
	
    #Calculate the mean solar azimuth and solar incidence from the Observation data file
	#This will not be available for HVM3. Will need predicted values for center of the planned target from MOS/GDS or MdNav
    azmband = obs_img.GetRasterBand(1)
    azmmean = azmband.ComputeStatistics(0)[2]
    incband = obs_img.GetRasterBand(2)
    incmean = incband.ComputeStatistics(0)[2]

	#Reproject topography to sinusoidal to increase chances of a match. Can't get the $clon variable in the proj4 syntax, so sending it to temp file, then executing that.
    gdal.Warp(f'{workdir}/{m3id}_topo_sinu.tif',f'{workdir}/{m3id}_topo.tif',
            options=gdal.WarpOptions(
                format='GTiff',
                dstSRS=f'+proj=sinu +lon_0={clon} +x_0=0 +y_0=0 +a=1737400 +b=1737400 +no_defs'
            ))
    
    #Create a flat Radiance product with 10 bands averaged. For M3 testing, longer wavelengths produced better matching results.
	#Separate the individual bands into separate rasters
    bfns = [f'{workdir}/{m3id}_RDN_b{b}.tif' for b in range(*band_bnds)]
    for b in range(*band_bnds):
        gdal.Translate(bfns[b-band_bnds[0]], inrdn,
                    options=gdal.TranslateOptions(
                        format='GTiff',
                        bandList=[b]
                    ))
    gdal_calc.Calc(calc='mean(a, axis=0)', a=bfns, outfile=f'{workdir}/{m3id}_RDN_average.tif')

    #Create 8-bit raster of averaged radiance for matching
    med_im = np.array(Image.open(f'{workdir}/{m3id}_RDN_average.tif'))
    gdal.Translate(f"{workdir}/{m3id}_RDN_average_byte.tif", f"{workdir}/{m3id}_RDN_average.tif", 
                options=gdal.TranslateOptions(
                    format='GTiff',
                    outputType=gdalconst.GDT_Byte,
                    scaleParams=[[med_im.min(),med_im.max(),1,255]]
                ))

    # Clean up unused files
    os.remove(f"{workdir}/{m3id}_RDN_average.tif")
    for name in bfns: os.remove(name)

    def checkFM(alt, az):
        #Make temporary dir for match image
        os.mkdir(f"{workdir}/{m3id}")
        #Create a hillshade map from the topography
        gdal.DEMProcessing(f"{workdir}/{m3id}_hillshade_{alt:0.2f}_{az:0.2f}.tif", f"{workdir}/{m3id}_topo_sinu.tif", "hillshade",
                    options=gdal.DEMProcessingOptions(
                        format='GTiff',
                        alg='ZevenbergenThorne',
                        azimuth=azmmean,
                        altitude=incmean
                    ))
        #Define files for RDN, hillshade, match output
        inrdn_fm = f'{workdir}/{m3id}_RDN_average_byte.tif'
        inshd_fm = f'{workdir}/{m3id}_hillshade_{alt:0.2f}_{az:0.2f}.tif'
        out_fm = f'{workdir}/{m3id}/{m3id}_{alt:0.2f}_{az:0.2f}'
        #Run image match
        try:
            match_images(inrdn_fm, inshd_fm, out_fm)
        except Exception as e:
            print(e)
            print("MATCH FAILED")
            shutil.rmtree(f"{workdir}/{m3id}")
            return False

        #Return True if process worked, else remove the temp dir and return False
        if os.path.isfile(f"{out_fm}.png"):
            return True
        else:
            shutil.rmtree(f"{workdir}/{m3id}")
            return False
    
    def makeHSH(alt, az):
        gdal.DEMProcessing(f"{hshdir}/{m3id}_hillshade_{alt:0.2f}_{az:0.2f}.tif", f"{workdir}/{m3id}_topo_sinu.tif", "hillshade",
                    options=gdal.DEMProcessingOptions(
                        format='GTiff',
                        alg='ZevenbergenThorne',
                        azimuth=az,
                        altitude=alt
                    ))
        
    
    # Make Hillshades
    hshdir=f'{m3id}_hsh'
    if os.path.isdir(hshdir):
        shutil.rmtree(hshdir)
    os.mkdir(hshdir)

    makeHSH(incmean, azmmean)
    azm, inc = np.meshgrid(np.arange(azmmean-60, azmmean+61, 10),
                            np.arange( 10, 80, 10))
    coords = np.vstack((inc.flatten(), azm.flatten())).T
    for k in range(len(coords)):
        makeHSH(*coords[k])
    
    # # If there has been any match, move the matching directory to Results/Worked, else 
    # # write a file to Results/Failed indicating no match was found.
    # if matched:
    #     shutil.move(f"{workdir}/{m3id}", f"Results/Worked/{m3id}")
    # else:
    #     shutil.move(f"{workdir}/{m3id}_RDN_average_byte.tif", f"Results/Failed/{m3id}_RDN_average_byte.tif")
    shutil.move(f"{workdir}/{m3id}_RDN_average_byte.tif", f"{hshdir}/{m3id}_RDN_average_byte.tif")
    shutil.rmtree(workdir)
    # return True, True, matched


if __name__ == "__main__":
    if len(sys.argv) == 2:
        run_match(sys.argv[1])
        quit()
    if sys.argv[1] == '-f':
        m3ids = open(sys.argv[2], 'r').read().split('\n')
        data = np.zeros((len(m3ids), 3))
        
        firstrun = np.zeros(len(m3ids))
        filefound = np.zeros(len(m3ids))
        matchsuccess = np.zeros(len(m3ids))
        for k_m3id in range(len(m3ids)):
            firstrun[k_m3id], filefound[k_m3id], matchsuccess[k_m3id] = run_match(m3ids[k_m3id])
        
        outdf = pd.DataFrame({
            "M3ID": m3ids,
            "FIRST RUN": ["True" if k else "False" for k in firstrun],
            "FILE FOUND": ["True" if k else "False" for k in filefound],
            "MATCH SUCCEEDED": ["True" if k else "False" for k in matchsuccess]
        })
        # fnout = 'dataout.csv'
        fnout = 'dataout.csv' if len(sys.argv) < 4 else sys.argv[3]
        outdf.to_csv(fnout, index=False)
        # np.save('dataout.npy', data)
    else:
        print("INVALID PARAMS")
        print("SINGLE RUN:\t python LTB_FM_M3.py <M3ID>")
        print("BATCH RUN:\t python LTB_FM_M3.py -f <M3 list file>")
    

