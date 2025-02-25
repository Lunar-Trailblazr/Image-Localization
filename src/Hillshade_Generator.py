import os, sys, shutil, traceback, logging

from osgeo import gdal, osr, gdalconst
from osgeo_utils import gdal_calc

import numpy as np
import pandas as pd
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt

class Hillshade_Generator:
    def __init__(self, M3_OBJ, workdir, inlola="Topography/LunarTopography_60mpx.tif"):
        self.M3_OBJ=M3_OBJ
        self.workdir=workdir
        self.inlola=inlola
        self.get_topo_sinu(regen_topo=True)
    
    def get_topo_clip(self, outfn=None):
        if outfn is None:
            outfn = f'{self.workdir}/{self.M3_OBJ.m3id}_topo.tif'
        #Clip the global LOLA DEM topography to the bounding box.
        gdal.Warp(outfn, self.inlola,
                options = gdal.WarpOptions(
                    format='GTiff',
                    outputBounds=(self.M3_OBJ.bound_minlon,
                                self.M3_OBJ.bound_minlat,
                                self.M3_OBJ.bound_maxlon,
                                self.M3_OBJ.bound_maxlat),
                    outputBoundsSRS='+proj=longlat +a=1737400 +b=1737400 +no_defs'
                ))
        self.clip_fn = outfn
    
    def get_topo_sinu(self, outfn=None, regen_topo=False):
        if outfn is None:
            outfn = f'{self.workdir}/{self.M3_OBJ.m3id}_topo_sinu.tif'
        
        if regen_topo:
            self.get_topo_clip()

        # Reproject the topography to sinusoidal
        gdal.Warp(outfn,self.clip_fn,
            options=gdal.WarpOptions(
                format='GTiff',
                dstSRS=f'+proj=sinu +lon_0={self.M3_OBJ.clon} +x_0=0 +y_0=0 +a=1737400 +b=1737400 +no_defs'
            ))
        self.sinu_fn = outfn
        sinu_img = cv.imread(self.sinu_fn, cv.IMREAD_ANYDEPTH)
        self.sinu_mask = np.where(sinu_img < 32767)
    
    def get_hillshade(self, azm=None, inc=None, zfactor=1, fnout=None, regen_topo=False):
        azm     = self.M3_OBJ.azm if azm is None else azm
        inc     = self.M3_OBJ.inc if inc is None else inc
        alt     = 90-inc

        if fnout is None:
            fnout = f'{self.workdir}/{self.M3_OBJ.m3id}_hillshade_az{azm:0.2f}_inc{inc:0.2f}_z{zfactor:0.2f}.tif'
        if regen_topo:
            self.get_topo_sinu(regen_topo=True)

        gdal.DEMProcessing(
            fnout, self.sinu_fn, "hillshade",
            options=gdal.DEMProcessingOptions(
                format='GTiff',
                alg='ZevenbergenThorne',
                azimuth=azm,
                altitude=alt,
                zFactor=zfactor
            ))
        return fnout
    
    def get_best_zfactor(self, azm=None, inc=None, initial_guess=1):
        azm     = self.M3_OBJ.azm if azm is None else azm
        inc     = self.M3_OBJ.inc if inc is None else inc
        alt     = 90-inc
        guess = initial_guess

        for i in range(10):
            prev_guess = guess
            hsh_fn = self.get_hillshade(azm=azm, inc=inc, zfactor=guess)
            hsh_image = cv.imread(hsh_fn, cv.IMREAD_ANYDEPTH)
            hsh_masked_region = hsh_image[self.sinu_mask]

            prop_high = len(np.where(hsh_masked_region>250)[0])/len(hsh_masked_region)
            prop_low = len(np.where(hsh_masked_region<5)[0])/len(hsh_masked_region)
            stdev = hsh_masked_region.std()
            mean = hsh_masked_region.mean()
            print(f"high saturation: {prop_high}")
            print(f"low saturation: {prop_low}")
            print(f"standard deviation: {hsh_masked_region.std()}")
            print(f"mean: {hsh_masked_region.mean()}")

            guess += round(prop_high*10)
            guess -= round(prop_low*10)
            if stdev < 45:
                guess += 0.5
            if mean < 100:
                guess -= 1 if mean < 50 else 0.5
            if mean > 150:
                guess += 1 if mean > 200 else 0.5
            if mean-stdev < 0:
                guess -= 1
            if mean+stdev > 255:
                guess += 1
            
            print(f"{prev_guess} --> {guess}")
            if guess == prev_guess:
                break
        return guess

