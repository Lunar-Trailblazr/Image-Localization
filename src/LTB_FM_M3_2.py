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

# Depends on ImageReg.py and M3.py

# Kevin Gauld (kgauld@caltech.edu)
# June 2024


import os, sys, shutil, traceback, logging

from osgeo import gdal, osr, gdalconst
from osgeo_utils import gdal_calc

import numpy as np
import pandas as pd
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt

from M3 import M3
from ImageReg import IterativeMatcher

logging.basicConfig(filename='Results/runlog.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logging.getLogger("PIL.TiffImagePlugin").disabled=True

# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

logging.info('STARTING NEW RUN')

FM_OBJ = IterativeMatcher()

# Input file for topography
inlola = "Topography/LunarTopography_60mpx.tif"

# Ground resolution of M3 in m/pix. Used to approximate the size of the
# M3 image used in matching, guaranteeing a sufficiently large hillshade.
m3resolution = 200

# Define the bands used for averaging the M3 radiance data. This includes
# the first band, excludes the last band. Averaging bands together increases
# quality of the image product used in matching.
band_bnds=(71,82)

# Input directory for the radiance data.
# The currently used data is clipped to 2677 rows. The implementation no longer
# relies on this feature, but it is noted here for completeness.
m3dir="Data_M3"

# These are the metrics recorded for each run of the matching script.
# Each of these metrics are documented later in the script.
DF_COLS = ['M3ID', 'WORKED',  
           'FIRST_TRY', 'FILE_FOUND',
           'LON', 'LON_MIN', 'LON_MAX',  
           'LAT', 'LAT_MIN', 'LAT_MAX',
           'P_X', 'P_X_MIN', 'P_X_MAX',
           'P_Y', 'P_Y_MIN', 'P_Y_MAX',
           'BND_LON', 'BND_LON_MIN', 'BND_LON_MAX',  
           'BND_LAT', 'BND_LAT_MIN', 'BND_LAT_MAX',
           'INC', 'AZM', 'INC_MATCH', 'AZM_MATCH', 'N_MATCHES']
def get_backplanes(rdn_fn, hsh_fn, M3_OBJ, H):
    hsh_info = gdal.Open(hsh_fn)

    hsh_im = cv.imread(hsh_fn, cv.IMREAD_ANYDEPTH)
    rdn_im = cv.imread(rdn_fn, cv.IMREAD_ANYDEPTH)

    hsh_tf = hsh_info.GetGeoTransform()
    proj_xy = lambda x_pix, y_pix, tf: (tf[0]+x_pix*tf[1], tf[3]+y_pix*tf[5], 0)

    mask = cv.warpPerspective(np.ones(rdn_im.shape), H, hsh_im.shape[::-1])
    mask = cv.dilate(mask, np.ones((5,5)), iterations=1)

    mask_imagezone = np.where(mask==1)
    maskpoints = [proj_xy(mask_imagezone[1][k], mask_imagezone[0][k], hsh_tf) for k in range(len(mask_imagezone[0]))]
    backplane_data = np.array(M3_OBJ.out2in_tf.TransformPoints(maskpoints))

    lonimg_hsh = np.ones(mask.shape)*500
    lonimg_hsh[np.where(mask==1)] = backplane_data[:,0]
    lon_backplane = cv.warpPerspective(lonimg_hsh, np.linalg.inv(H), rdn_im.shape[::-1])

    latimg_hsh = np.ones(mask.shape)*500
    latimg_hsh[np.where(mask==1)] = backplane_data[:,1]
    lat_backplane = cv.warpPerspective(latimg_hsh, np.linalg.inv(H), rdn_im.shape[::-1])

    return lon_backplane, lat_backplane

def checkFM(M3_OBJ, workdir, inc=None, azm=None):
    m3id    = M3_OBJ.m3id
    az      = M3_OBJ.azm if azm is None else azm
    inc     = M3_OBJ.inc if inc is None else inc
    alt     = 90 - inc

    #Make temporary dir for match image
    if os.path.isdir(f"{workdir}/{m3id}"):
        shutil.rmtree(f"{workdir}/{m3id}")
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
    
    # Read in the images
    RDN = cv.imread(inrdn_fm, cv.IMREAD_ANYDEPTH)
    HSH = cv.imread(inshd_fm, cv.IMREAD_ANYDEPTH)

    success = False
    #Run image match, providing the filenames to write output to.
    try:
        H_f, kp_f, kp2, matches_f, success = FM_OBJ.match_and_plot([f'{out_fm}_match.tif', 
                                                                    f'{out_fm}_match2.tif', 
                                                                    f'{workdir}/{m3id}/{m3id}_RDN_WARP.tif',
                                                                    f'{workdir}/{m3id}/{m3id}_RDN_KPS.tif',
                                                                    f'{out_fm}_KPS.tif'], 
                                                                    RDN, HSH, 
                                                                    colormatch=True, 
                                                                    cmatch_fns=[f'{workdir}/{m3id}/{m3id}_RDN_NORM.tif',
                                                                                f'{out_fm}_NORM.tif',])
    except Exception as e:
        # Uncomment the following line for debugging exceptions
        logging.debug(traceback.format_exc())
        logging.error(f"{m3id} FAILED: {e}")
        # shutil.rmtree(f"{workdir}/{m3id}")
        return False, inshd_fm, None, 0
    
    # If matching failed, return that the match failed
    if not success and not os.path.isfile(f"{out_fm}_match.tif"):
        return False, inshd_fm, None, len(matches_f)
    
    # Create the backplanes and save them to files
    logging.info(f'{m3id} SUCCEEDED! Generating backplanes...')
    lon_bp, lat_bp = get_backplanes(inrdn_fm, inshd_fm, M3_OBJ, H_f)
    np.save(f'{workdir}/{m3id}/{m3id}_LON.npy', lon_bp)
    plt.imsave(f'{workdir}/{m3id}/{m3id}_LON.png', lon_bp)
    np.save(f'{workdir}/{m3id}/{m3id}_LAT.npy', lat_bp)
    plt.imsave(f'{workdir}/{m3id}/{m3id}_LAT.png', lat_bp)
    np.save(f'{workdir}/{m3id}/{m3id}_HOMOGRAPHY.npy', H_f)

    if not (1<np.ptp(lon_bp)<10 and 1<np.ptp(lat_bp)<10):
        logging.error(f"{m3id} BACKPLANE ERROR! MATCH FAILED!")
        return False, inshd_fm, None, len(matches_f)
    

    return True, inshd_fm, [f'{out_fm}_match.tif', 
                            f'{out_fm}_colormatched.tif'], len(matches_f)

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
        shutil.rmtree(f'Results/Failed/{m3id}')
    
    #Create a directory to run all of the processes/make files in.
    workdir=f'{m3id}_work'
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    os.mkdir(workdir)

    ##################################################
    ############### M3 DATA OBJECT ###################
    ##################################################

    # Create M3 data object. This precomputes many of the metrics needed for
    # creation of a radiance and hillshade image, and allows for the recording
    # of these metrics.
    try:
        M3_OBJ = M3(m3id=m3id, 
                    m3dir=m3dir, 
                    resolution=m3resolution)
        infodict['FILE_FOUND']  =   True
    except FileNotFoundError as e:
        # If there is a filenotfound error, then log that the M3
        # data could not be accessed and return.
        if os.path.isdir(workdir):
            shutil.rmtree(workdir)
        os.mkdir(f'Results/Failed/{m3id}')
        with open(f"Results/Failed/{m3id}/{m3id}.txt", "rw") as f:
            f.write(f"{m3id} NOT FOUND")
        infodict['FILE_FOUND']  = False
        infodict['WORKED']      = False
        return infodict
    
    # Set the bounding box for the hillshade region. To match the expectation
    # of HVM3, the box is set from the lat/lon center,
    M3_OBJ.set_bounding_box_center(77000, 135000)

    # Get the radiance image from the M3 data, as an 8bit raster image. Set the
    # aspect ratio of the cropped raster to be as similar as possible to HVM3.
    rdn_image_fn = M3_OBJ.get_RDN_8bit_crop(bands=np.arange(*band_bnds),
                                       w_x=60800, w_y=74600,
                                       outdir=workdir,
                                       outfn=f'{M3_OBJ.m3id}_RDN_average_byte.tif',
                                       contrast_mode='none')

    ####### Populate the data logger
    
    # Get center/boundary lat/lon. For HVM3, there will only be a center lat
    # lon coordinate given, no min or max.
    infodict['LAT']         =   M3_OBJ.clat
    infodict['LAT_MIN']     =   M3_OBJ.minlat
    infodict['LAT_MAX']     =   M3_OBJ.maxlat
    infodict['LON']         =   M3_OBJ.clon
    infodict['LON_MIN']     =   M3_OBJ.minlon
    infodict['LON_MAX']     =   M3_OBJ.maxlon
    # Record the solar azimuth and incidence angles. This will not be available
    # for HVM3, we will need to use predicted values from MOS/GDS or MdNav
    infodict['INC']         =   M3_OBJ.inc
    infodict['AZM']         =   M3_OBJ.azm
    # Get the bounding box for the hillshade in both meter and lat/lon space
    infodict['P_X']         =   M3_OBJ.px
    infodict['P_X_MIN']     =   M3_OBJ.xmin
    infodict['P_X_MAX']     =   M3_OBJ.xmax
    infodict['P_Y']         =   M3_OBJ.py
    infodict['P_Y_MIN']     =   M3_OBJ.ymin
    infodict['P_Y_MAX']     =   M3_OBJ.ymax
    infodict['BND_LAT']     =   M3_OBJ.bound_lat
    infodict['BND_LAT_MIN'] =   M3_OBJ.bound_minlat
    infodict['BND_LAT_MAX'] =   M3_OBJ.bound_maxlat
    infodict['BND_LON']     =   M3_OBJ.bound_lon
    infodict['BND_LON_MIN'] =   M3_OBJ.bound_minlon
    infodict['BND_LON_MAX'] =   M3_OBJ.bound_maxlon

    ######################################################
    ############### HILLSHADE CREATION ###################
    ######################################################
    
    
    #Clip the global LOLA DEM topography to the bounding box.
    gdal.Warp(f'{workdir}/{m3id}_topo.tif', inlola,
            options = gdal.WarpOptions(
                format='GTiff',
                outputBounds=(M3_OBJ.bound_minlon,
                              M3_OBJ.bound_minlat,
                              M3_OBJ.bound_maxlon,
                              M3_OBJ.bound_maxlat),
                outputBoundsSRS='+proj=longlat +a=1737400 +b=1737400 +no_defs'
            ))
    
    # Reproject the topography to sinusoidal
    gdal.Warp(f'{workdir}/{m3id}_topo_sinu.tif',f'{workdir}/{m3id}_topo.tif',
            options=gdal.WarpOptions(
                format='GTiff',
                dstSRS=f'+proj=sinu +lon_0={M3_OBJ.clon} +x_0=0 +y_0=0 +a=1737400 +b=1737400 +no_defs'
            ))
    
    # Check if a match was successful
    matched, hsh_fn, saved_fns, n_matches = checkFM(M3_OBJ, workdir)
    first_hshfn = hsh_fn
    infodict['FIRST_TRY']=matched
    # if not matched:
    #     logging.info("Actual OBS failed, trying default OBS (inc 45 az 315)")
    #     matched, hsh_fn, saved_fns, n_matches = checkFM(M3_OBJ, workdir,
    #                                                     inc=45, azm=315)
    #     if matched:
    #         infodict['INC_MATCH'] = 45
    #         infodict['AZM_MATCH'] = 315
    # else:
    infodict['INC_MATCH'] = infodict['INC']
    infodict['AZM_MATCH'] = infodict['AZM']
        
    # if not matched:
    #     # Retry matches for azm=[azmmean-60, azmmean+60] and 
    #     # inc = [10,80] in intervals of 10
    #     print("RETRYING MATCH")
    #     azm, inc = np.meshgrid(np.arange(M3_OBJ.azm-60, M3_OBJ.azm+61, 10),
    #                            np.arange( 10, 90, 10))
    #     coords = np.vstack((inc.flatten(), azm.flatten())).T
    #     for k in range(len(coords)):
    #         matched, hsh_fn, saved_fns, n_matches = checkFM(M3_OBJ, workdir, inc=coords[k][0], azm=coords[k][1])
    #         if matched:
    #             infodict['INC_MATCH'] = coords[k][0]
    #             infodict['AZM_MATCH'] = coords[k][1]
    #             infodict['N_MATCHES'] = n_matches
    #             break
    # else:
    infodict['WORKED'] = matched
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
        shutil.move(f"{workdir}/{m3id}", f"Results/Failed/{m3id}")
        shutil.move(f"{workdir}/{m3id}_RDN_average_byte.tif", f"Results/Failed/{m3id}/{m3id}_RDN_average_byte.tif")
        # os.mkdir(f"Results/Failed/{m3id}")
        # shutil.move(f"{workdir}/{m3id}_RDN_average_byte.tif", f"Results/Failed/{m3id}/{m3id}_RDN_average_byte.tif")
        shutil.move(first_hshfn, f"Results/Failed/{m3id}/{first_hshfn.split('/')[-1]}")
        if hsh_fn != first_hshfn:
            shutil.move(first_hshfn, f"Results/Failed/{m3id}/{first_hshfn.split('/')[-1]}")
    
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
                logging.info(f"Skipping {m3ids[k_m3id]} - MATCH {'WORKED' if data[data['M3ID'] == m3ids[k_m3id]]['WORKED'].values[0] else 'FAILED'}")
                continue
            logging.info(f"Starting {m3ids[k_m3id]}")
            k_data = run_match(m3ids[k_m3id])
            data = pd.concat([data, pd.DataFrame(k_data, index=[0])])
            data.to_csv(f'Results/{fnout}',index=False)
    else:
        logging.error("INVALID PARAMS\nSINGLE RUN:\t python LTB_FM_M3.py <M3ID>\nBATCH RUN:\t python LTB_FM_M3.py -f <M3 list file>")
    

