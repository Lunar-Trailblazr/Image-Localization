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
from Hillshade_Generator import Hillshade_Generator
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
M3_RES = 140
HVM3_RES = 60

# Define the bands used for averaging the M3 radiance data. This includes
# the first band, excludes the last band. Averaging bands together increases
# quality of the image product used in matching.
band_bnds=(71,82)  #(5,15)

UNCERTAINTY_SCALE = 1 #2**0.5

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

def get_latlon_grid(i_grid, j_grid, hshimg, BND_LATS, BND_LONS):
    """
    Converts grids of pixel coordinates (i, j) in a sinusoidally projected image (centered at the middle of the image)
    to latitude/longitude grids.

    Parameters:
        i_grid (np.ndarray): Grid of row indices (y-coordinates).
        j_grid (np.ndarray): Grid of column indices (x-coordinates).
        hshimg (np.ndarray): Image array for dimensions.
        BND_LATS (list): [lat_min, lat_max] bounding box.
        BND_LONS (list): [lon_min, lon_max] bounding box.

    Returns:
        tuple: (lat_grid, lon_grid) - Grids of latitude and longitude values.
    """
    height, width = hshimg.shape
    
    dlat = BND_LATS[1]-BND_LATS[0]
    dlon = BND_LONS[1]-BND_LONS[0]
    
    # Compute latitude values (same as before)
    lat_grid = BND_LATS[1] - i_grid * (dlat/(height - 1))

    # Compute longitude values centered at the middle of the image
    lon_center = (dlon) / 2+BND_LONS[0]  # Center of the longitude range
    j_center = width / 2  # Center pixel of the image in x direction
    
    cos_grid = np.cos(np.radians(lat_grid))
    scale = 1/np.max(cos_grid)
    
    lon_grid = lon_center + (j_grid - j_center) * (dlon/(width - 1)) / (cos_grid*scale)

    return lat_grid, lon_grid

def get_backplanes(rdn_fn, hsh_fn, M3_OBJ, H):
    hsh_info = gdal.Open(hsh_fn)

    hsh_im = cv.imread(hsh_fn, cv.IMREAD_ANYDEPTH)
    rdn_im = cv.imread(rdn_fn, cv.IMREAD_ANYDEPTH)

    lat_bounds = (M3_OBJ.bound_minlat, M3_OBJ.bound_maxlat)
    lon_bounds = (M3_OBJ.bound_minlon, M3_OBJ.bound_maxlon)

    hshimg = cv.imread(hsh_fn, cv.IMREAD_ANYDEPTH)
    j,i = np.meshgrid(np.arange(hshimg.shape[1]), np.arange(hshimg.shape[0]))
    lat_grid, lon_grid = get_latlon_grid(i, j, hshimg, lat_bounds, lon_bounds)
    lat_grid[np.where(lat_grid>lat_bounds[1])] = 600#np.inf
    lat_grid[np.where(lat_grid<lat_bounds[0])] = 600#np.inf
    lon_grid[np.where(lon_grid>lon_bounds[1])] = 600#np.inf
    lon_grid[np.where(lon_grid<lon_bounds[0])] = 600#np.inf

    mask = cv.warpPerspective(np.ones(rdn_im.shape), H, hsh_im.shape[::-1])
    mask = cv.dilate(mask, np.ones((5,5)), iterations=1)

    lonimg_hsh = lon_grid*mask#[np.where(mask==1)]
    latimg_hsh = lat_grid*mask#[np.where(mask==1)]

    lon_backplane = cv.warpPerspective(lonimg_hsh, np.linalg.pinv(H), rdn_im.shape[::-1])
    lat_backplane = cv.warpPerspective(latimg_hsh, np.linalg.pinv(H), rdn_im.shape[::-1])

    return lon_backplane, lat_backplane

def checkFM(M3_OBJ, HSH_OBJ, workdir, 
            inc=None, azm=None, zfactor=1, inrdn_fm=None):
    m3id    = M3_OBJ.m3id
    azm     = M3_OBJ.azm if azm is None else azm
    inc     = M3_OBJ.inc if inc is None else inc

    inshd_fm = HSH_OBJ.get_hillshade(
        azm=azm, inc=inc, zfactor=zfactor,
        fnout=f'{workdir}/{m3id}_hillshade_az{azm:0.2f}_inc{inc:0.2f}.tif'
    )
    inrdn_fm = f'{workdir}/{m3id}_RDN_average_byte.tif' if inrdn_fm is None else inrdn_fm

    #Make temporary dir for match image
    if os.path.isdir(f"{workdir}/{m3id}"):
        shutil.rmtree(f"{workdir}/{m3id}")
    os.mkdir(f"{workdir}/{m3id}")
    out_fm = f'{workdir}/{m3id}/{m3id}_az{azm:0.2f}_inc{inc:0.2f}'

    # Read in the images
    RDN = cv.imread(inrdn_fm, cv.IMREAD_ANYDEPTH)
    HSH = cv.imread(inshd_fm, cv.IMREAD_ANYDEPTH)

    T_GUESS = (np.array(HSH.shape)-np.array(RDN.shape)*M3_RES/HVM3_RES)/2
    H_init = np.array([[M3_RES/HVM3_RES, 0.01,   T_GUESS[1]],
                       [0.01,   M3_RES/HVM3_RES, T_GUESS[0]],
                       [0,      0,      1]])
    # H_init = np.eye(3, dtype=np.float32)
    success = False
    #Run image match, providing the filenames to write output to.
    try:
        H_f, kp_f, kp2, matches_f, _, success = FM_OBJ.match_and_plot([f'{out_fm}_match.tif', 
                                                                    f'{out_fm}_match2.tif', 
                                                                    f'{workdir}/{m3id}/{m3id}_RDN_WARP.tif',
                                                                    f'{workdir}/{m3id}/{m3id}_RDN_KPS.tif',
                                                                    f'{out_fm}_KPS.tif'], 
                                                                    RDN, HSH, 
                                                                    colormatch=True, 
                                                                    cmatch_fns=[f'{workdir}/{m3id}/{m3id}_RDN_NORM.tif',
                                                                                f'{out_fm}_NORM.tif',],
                                                                    H_init=H_init)
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
    pts = np.array([[kp_f[k.queryIdx].pt for k in matches_f],
                    [kp2[k.trainIdx].pt for k in matches_f]])
    np.save(f'{workdir}/{m3id}/{m3id}_MATCHES.npy', pts)

    img_area = np.radians(np.ptp(lon_bp)) * np.radians(np.ptp(lat_bp)) * (1737.4)**2
    AREA_TARGET = 51.3*63
    logging.info(f"{np.ptp(lon_bp)=}, {np.ptp(lat_bp)=}")
    logging.info(f"{m3id} AREA: {img_area} TARGET: {AREA_TARGET} km^2")
    # if not (1<np.ptp(lon_bp)<10 and 1<np.ptp(lat_bp)<10):
    if not 0.5*AREA_TARGET < img_area < 5*AREA_TARGET:
        logging.error(f"{m3id} BACKPLANE ERROR! MATCH FAILED BY AREA THRESHOLD! {img_area/AREA_TARGET}")
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
                    resolution=M3_RES)
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
    M3_OBJ.set_bounding_box_center(64400*UNCERTAINTY_SCALE, 112700*UNCERTAINTY_SCALE)

    # Get the radiance image from the M3 data, as an 8bit raster image. Set the
    # aspect ratio of the cropped raster to be as similar as possible to HVM3.
    w_x = 22e3*M3_RES/HVM3_RES
    w_y = 27e3*M3_RES/HVM3_RES
    rdn_image_fn = M3_OBJ.get_RDN_8bit_crop(bands=np.arange(*band_bnds),
                                       w_x=w_x, w_y=w_y,
                                       outdir=workdir,
                                       outfn=f'{M3_OBJ.m3id}_RDN_average_byte.tif',
                                       contrast_mode='rescale')

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

    HSH_OBJ = Hillshade_Generator(M3_OBJ=M3_OBJ, 
                                  workdir=workdir, 
                                  inlola=inlola)
    
    # Check if a match was successful
    matched, hsh_fn, saved_fns, n_matches = checkFM(M3_OBJ, HSH_OBJ, workdir,
                                                    inrdn_fm=rdn_image_fn)
    first_hshfn = hsh_fn
    infodict['FIRST_TRY']=matched
    if not matched:
        logging.error('FIRST TRY FAILED, OPTIMIZING Z SCALE')
        zf = HSH_OBJ.get_best_zfactor()
        logging.info(f'Using zFactor {zf}')
        matched, hsh_fn, saved_fns, n_matches = checkFM(M3_OBJ, HSH_OBJ, workdir,
                                                    inrdn_fm=rdn_image_fn, zfactor=zf)
    if not matched:
        logging.error(f"Z SCALING FAILED. TRYING Z FACTOR 0.5")
        matched, hsh_fn, saved_fns, n_matches = checkFM(M3_OBJ, HSH_OBJ, workdir,
                                                    inrdn_fm=rdn_image_fn, zfactor=0.5)
        
    infodict['INC_MATCH'] = infodict['INC']
    infodict['AZM_MATCH'] = infodict['AZM']
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
    

