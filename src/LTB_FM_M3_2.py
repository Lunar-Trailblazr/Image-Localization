# This code takes M3 data and matches it to LOLA topography

# First step: radiance images are read and hillshade topography is generated
# Second step: feature matching with SIFT performed to match M3 features to hillshade features
#               If this feature matching doesn't work, matching is reattempted with lighting conditions.

# Data exists in Data_M3 directory. All data has been clipped to the middle 2677 rows to be similar to HVM3.

# This script builds on the original LTB_FeatureMatching_M3.sh script, authored by Jay Dickson (jdickson@umn.edu)

# To execute, either run one M3 image as:
# python3 LTB_FM_M3.py M3G20090207T090807
# Or a list of many M3IDs as:
# python3 LTB_FM_M3.py -f filename

# If running with a list of m3ids, ensure Results/Worked/ and Results/Failed/ exist as directories for output.

# Depends on ImageReg.py and M3_Preprocess.py

# Kevin Gauld (kgauld@caltech.edu)
# June 2024


import os, sys, shutil, traceback

from osgeo import gdal, osr, gdalconst
from osgeo_utils import gdal_calc

import numpy as np
import pandas as pd
from PIL import Image
import cv2 as cv

from M3 import M3
from ImageReg import IterativeMatcher


FM_OBJ = IterativeMatcher()

#Input topography file to be used to generate hillshade
inlola = "Topography/LunarTopography_60mpx.tif"
#Ground resolution of M3 in m/px.
#This is required for extraction of LOLA data over a reasonable area of the Moon to generate a hillshade for matching.
m3resolution = 200
#Define starting and ending bands in M3 for averaging.
#The code averages M3 bands together to increase the quality of the matching product. 
# Set the start band and end band (in units of band number). Includes first band, excludes last band
band_bnds=(71,82)
#Input directory for radiance data. This is a volume of M3 data that have been clipped to 2677 rows to be more similar to HVM3 data. The middle 2677 rows were extracted for each M3 product.
m3dir="Data_M3"

DF_COLS = ['M3ID', 'WORKED', 
           'FIRST_TRY', 'FILE_FOUND',
           'LON', 'LON_MIN', 'LON_MAX',  
           'LAT', 'LAT_MIN', 'LAT_MAX',
           'P_X', 'P_X_MIN', 'P_X_MAX',
           'P_Y', 'P_Y_MIN', 'P_Y_MAX',
           'BND_LON', 'BND_LON_MIN', 'BND_LON_MAX',  
           'BND_LAT', 'BND_LAT_MIN', 'BND_LAT_MAX',
           'INC', 'AZM', 'INC_MATCH', 'AZM_MATCH', 'N_MATCHES']

def checkFM(M3_OBJ, workdir, inc=None, azm=None):
    m3id    = M3_OBJ.m3id
    az      = M3_OBJ.azm if azm is None else azm
    inc     = M3_OBJ.inc if inc is None else inc
    alt     = 90 - inc

    #Make temporary dir for match image
    os.mkdir(f"{workdir}/{m3id}")

    #Define files for RDN, hillshade, match output
    inrdn_fm = f'{workdir}/{m3id}_RDN_average_byte.tif'
    inshd_fm = f'{workdir}/{m3id}_hillshade_az{az:0.2f}_inc{inc:0.2f}.tif'
    out_fm = f'{workdir}/{m3id}/{m3id}_az{az:0.2f}_inc{inc:0.2f}'

    #Create a hillshade map from the topography
    gdal.DEMProcessing(
        inshd_fm, 
        f"{workdir}/{m3id}_topo_sinu.tif", 
        "hillshade",
        options=gdal.DEMProcessingOptions(
            format='GTiff',
            alg='ZevenbergenThorne',
            azimuth=az,
            altitude=alt,
            zFactor=2
        ))
    
    A = cv.imread(inrdn_fm, cv.IMREAD_ANYDEPTH)
    B = cv.imread(inshd_fm, cv.IMREAD_ANYDEPTH)
    # A = cv.resize(A, B.shape[::-1])
    success = False
    #Run image match
    try:
        H_f, kp_f, kp2, matches_f, success = FM_OBJ.match_and_plot([f'{out_fm}_match.tif', 
                                                                    f'{out_fm}_match2.tif', 
                                                                    f'{workdir}/{m3id}/{m3id}_RDN_WARP.tif',
                                                                    f'{workdir}/{m3id}/{m3id}_RDN_KPS.tif',
                                                                    f'{out_fm}_KPS.tif'], 
                                                                    A, B, 
                                                                    colormatch=True, 
                                                                    cmatch_fns=[f'{workdir}/{m3id}/{m3id}_RDN_NORM.tif',
                                                                                f'{out_fm}_NORM.tif',])
        # match_images(inrdn_fm, inshd_fm, out_fm)
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print("MATCH FAILED")
        shutil.rmtree(f"{workdir}/{m3id}")
        return False, inshd_fm, None, 0

    #Return True if process worked, else remove the temp dir and return False
    if success or os.path.isfile(f"{out_fm}_match.tif"):
        return True, inshd_fm, [f'{out_fm}_match.tif', f'{out_fm}_colormatched.tif'], len(matches_f)
    else:
        shutil.rmtree(f"{workdir}/{m3id}")
        return False, inshd_fm, None, len(matches_f)
    

def run_match(m3id):
    '''
        Runs feature matching for a given m3id
    '''
    infodict = {}
    for k in DF_COLS: infodict[k] = None
    infodict["M3ID"]=m3id

    # Check to see if this image has already been matched. If so, remove
    # the previous match and retry. This operates based on the assumption that
    # there is a pre-check to see if the dataframe contains this m3id, so
    # if the dataframe does not already contain this sample
    work = any(x.startswith(m3id) for x in os.listdir('Results/Worked'))
    fail = any(x.startswith(m3id) for x in os.listdir('Results/Failed'))
    if work:
        shutil.rmtree(f'Results/Worked/{m3id}')
    elif fail:
        if f'{m3id}.txt' in os.listdir('Results/Failed'):
            os.remove(f'Results/Failed/{m3id}.txt')
        else:
            os.remove(f'Results/Failed/{m3id}_RDN_average_byte.tif')
    
    #Create a directory to run all of the processes/make files in.
    workdir=f'{m3id}_work'
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    os.mkdir(workdir)

    # Create M3 data object
    try:
        M3_OBJ = M3(m3id=m3id, 
                    m3dir=m3dir, 
                    resolution=m3resolution)
    except FileNotFoundError as e:
        if os.path.isdir(workdir):
            shutil.rmtree(workdir)
        f = open(f"Results/Failed/{m3id}.txt", "a")
        f.write(f"{m3id} NOT FOUND")
        f.close()
        infodict['FILE_FOUND']  = False
        infodict['WORKED']      = False
        return infodict

    infodict['FILE_FOUND']  =   True

    ###Calculate extent for hillshade. This calculates the center lat/lon from the LOC file for M3.
    ###For HVM3, the LOC file will not be available at this stage, so center coordinates should be extracted from the filename, or from a lookup table of targets.

    # Get center/boundary lat/lon
    infodict['LAT']         =   M3_OBJ.clat
    infodict['LAT_MIN']     =   M3_OBJ.minlat
    infodict['LAT_MAX']     =   M3_OBJ.maxlat
    infodict['LON']         =   M3_OBJ.clon
    infodict['LON_MIN']     =   M3_OBJ.minlon
    infodict['LON_MAX']     =   M3_OBJ.maxlon

    #Calculate the mean solar azimuth and solar incidence from the Observation data file
	#This will not be available for HVM3. Will need predicted values for center of the planned target from MOS/GDS or MdNav
    infodict['INC']         =   M3_OBJ.inc
    infodict['AZM']         =   M3_OBJ.azm

    # Get center x y in meter space
    # Set the bounding box. Add a kilometer on all sides for margin 
    # (may need more if pointing uncertainty is very high)
    # M3_OBJ.set_bounding_box(1000)
    M3_OBJ.set_bounding_box_center(77000, 135000)

    infodict['P_X']         =   M3_OBJ.px
    infodict['P_X_MIN']     =   M3_OBJ.xmin
    infodict['P_X_MAX']     =   M3_OBJ.xmax
    infodict['P_Y']         =   M3_OBJ.py
    infodict['P_Y_MIN']     =   M3_OBJ.ymin
    infodict['P_Y_MAX']     =   M3_OBJ.ymax

    #Get x and y in degree space. Parse the lines to define bounding box coordinates.
    infodict['BND_LAT']    =   M3_OBJ.bound_lat
    infodict['BND_LAT_MIN']=   M3_OBJ.bound_minlat
    infodict['BND_LAT_MAX']=   M3_OBJ.bound_maxlat
    infodict['BND_LON']    =   M3_OBJ.bound_lon
    infodict['BND_LON_MIN']=   M3_OBJ.bound_minlon
    infodict['BND_LON_MAX']=   M3_OBJ.bound_maxlon

    # Get the radiance image from the M3 data, as an 8bit raster image.
    # rdn_image_fn = M3_OBJ.get_RDN_8bit(bands=np.arange(*band_bnds),
    #                                    outdir=workdir,
    #                                    outfn=f'{M3_OBJ.m3id}_RDN_average_byte.tif')
    # rdn_image_fn = M3_OBJ.get_RDN_8bit_square(bands=np.arange(*band_bnds),
    #                                    outdir=workdir,
    #                                    outfn=f'{M3_OBJ.m3id}_RDN_average_byte.tif')
    rdn_image_fn = M3_OBJ.get_RDN_8bit_crop(bands=np.arange(*band_bnds),
                                       w_x=60800, w_y=74600,
                                       outdir=workdir,
                                       outfn=f'{M3_OBJ.m3id}_RDN_average_byte.tif')
    
    #Clip the global topography to the bounding box. The LOLA DEM is 60 m/px and does not need to be resampled for this procedure. Only clip.
    gdal.Warp(f'{workdir}/{m3id}_topo.tif', inlola,
            options = gdal.WarpOptions(
                format='GTiff',
                outputBounds=(M3_OBJ.bound_minlon,
                              M3_OBJ.bound_minlat,
                              M3_OBJ.bound_maxlon,
                              M3_OBJ.bound_maxlat),
                outputBoundsSRS='+proj=longlat +a=1737400 +b=1737400 +no_defs'
            ))
    
	#Reproject topography to sinusoidal to increase chances of a match.
    gdal.Warp(f'{workdir}/{m3id}_topo_sinu.tif',f'{workdir}/{m3id}_topo.tif',
            options=gdal.WarpOptions(
                format='GTiff',
                dstSRS=f'+proj=sinu +lon_0={M3_OBJ.clon} +x_0=0 +y_0=0 +a=1737400 +b=1737400 +no_defs'
            ))
    
    # Check if a match was successful
    matched, hsh_fn, saved_fns, n_matches = checkFM(M3_OBJ, workdir)
    first_hshfn = hsh_fn
    infodict['FIRST_TRY']=matched
    if not matched:
        # Retry matches for azm=[azmmean-60, azmmean+60] and 
        # inc = [10,80] in intervals of 10
        print("RETRYING MATCH")
        azm, inc = np.meshgrid(np.arange(M3_OBJ.azm-60, M3_OBJ.azm+61, 10),
                               np.arange( 10, 90, 10))
        coords = np.vstack((inc.flatten(), azm.flatten())).T
        for k in range(len(coords)):
            matched, hsh_fn, saved_fns, n_matches = checkFM(M3_OBJ, workdir, inc=coords[k][0], azm=coords[k][1])
            if matched:
                infodict['INC_MATCH'] = coords[k][0]
                infodict['AZM_MATCH'] = coords[k][1]
                infodict['N_MATCHES'] = n_matches
                break
    else:
        infodict['INC_MATCH'] = infodict['INC']
        infodict['AZM_MATCH'] = infodict['AZM']
        infodict['N_MATCHES'] = n_matches
    
    # If there has been any match, move the matching directory to Results/Worked, else 
    # write a file to Results/Failed indicating no match was found.
    if matched:
        shutil.copy(saved_fns[0], f"Results/Matches/{m3id}_match.tif")
        shutil.move(f"{workdir}/{m3id}", f"Results/Worked/{m3id}")
        shutil.move(f"{workdir}/{m3id}_RDN_average_byte.tif", f"Results/Worked/{m3id}/{m3id}_RDN_average_byte.tif")
        shutil.move(hsh_fn, f"Results/Worked/{m3id}/{m3id}_hillshade_az{infodict['AZM_MATCH']:0.2f}_inc{infodict['INC_MATCH']:0.2f}.tif")
        if hsh_fn != first_hshfn:
            shutil.move(first_hshfn, f"Results/Worked/{m3id}/{first_hshfn.split('/')[-1]}")
    else:
        infodict['N_MATCHES'] = 0
        os.mkdir(f"Results/Failed/{m3id}")
        shutil.move(f"{workdir}/{m3id}_RDN_average_byte.tif", f"Results/Failed/{m3id}/{m3id}_RDN_average_byte.tif")
        shutil.move(first_hshfn, f"Results/Failed/{m3id}/{first_hshfn.split('/')[-1]}")
    infodict['WORKED'] = matched
    shutil.rmtree(workdir)
    return infodict


if __name__ == "__main__":

    if len(sys.argv) == 2:
        print(run_match(sys.argv[1]))
        quit()
     
    if len(sys.argv) > 1 and sys.argv[1] == '-f':

        fnout = 'dataout.csv' if len(sys.argv) < 4 else sys.argv[3]
        m3ids = open(sys.argv[2], 'r').read().split('\n')

        if fnout not in os.listdir('Results'):
            data = pd.DataFrame(columns=DF_COLS)
        else:
            data = pd.read_csv(f'Results/{fnout}')
        
        for k_m3id in range(len(m3ids)):
            if m3ids[k_m3id] in data['M3ID'].values:
                print(f"Skipping {m3ids[k_m3id]} - MATCH {'WORKED' if data[data['M3ID'] == m3ids[k_m3id]]['WORKED'].values[0] else 'FAILED'}")
                continue
            k_data = run_match(m3ids[k_m3id])
            data = pd.concat([data, pd.DataFrame(k_data, index=[0])])
            data.to_csv(f'Results/{fnout}',index=False)
    else:
        print("INVALID PARAMS")
        print("SINGLE RUN:\t python LTB_FM_M3.py <M3ID>")
        print("BATCH RUN:\t python LTB_FM_M3.py -f <M3 list file>")
    

