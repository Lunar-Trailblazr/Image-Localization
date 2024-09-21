# M3.py
# 
# Class enabling work with M3 data. Processes the input ID to easily access metadata
# and perform postprocessing
#
# Relies on gdal and osr for manipulating files, and errno/os for error handling
#
# Kevin Gauld kgauld@caltech.edu
# July 2024

from osgeo import gdal, osr, gdalconst
from osgeo_utils import gdal_calc
import numpy as np
from PIL import Image
import errno
import os
from skimage import exposure
import cv2 as cv

class M3:
    def __init__(self, m3id, m3dir='Data_M3', fn_postfix='', resolution=200):
        '''
        Initialize the M3 data with its ID, the directory where data is stored,
        and the resolution of the data in m/pixel.

        On initialization, will parse out the lat, lon, inc, azm, and coordinate
        transforms down to the lunar surface
        '''
        self.m3id = m3id
        self.m3dir= m3dir
        self.filename = f"{self.m3dir}/{self.m3id}{fn_postfix}"
        self.loc_img = gdal.Open(f"{self.filename}_LOC.IMG")
        self.rdn_img = gdal.Open(f"{self.filename}_RDN.IMG")
        self.obs_img = gdal.Open(f"{self.filename}_OBS.IMG")
        self.resolution = resolution
        
        # Ensure all data files exist in the m3dir directory
        if self.loc_img is None:
            raise FileNotFoundError(errno.ENOENT, 
                                    os.strerror(errno.ENOENT), 
                                    f"{self.filename}_LOC.IMG")
        if self.rdn_img is None:
            raise FileNotFoundError(errno.ENOENT, 
                                    os.strerror(errno.ENOENT), 
                                    f"{self.filename}_RDN.IMG")
        if self.obs_img is None:
            raise FileNotFoundError(errno.ENOENT, 
                                    os.strerror(errno.ENOENT), 
                                    f"{self.filename}_OBS.IMG")
        
        # Set attributes for the latitude and longitude
        self.set_lat_lon()
        # Set attributes for the incidence and azimuth
        self.set_inc_azm()
        # Set up coordinate frame transformations
        self.set_coord_tfs()

        # Set attributes for the center xy location
        self.px, self.py = self.in2out_tf.TransformPoint(self.clon, self.clat, 0)[:2]
        # Determine the number of samples and lines in the image
        self.samples    =   self.rdn_img.RasterXSize
        self.lines      =   self.rdn_img.RasterYSize
        #Calculate the width and height of the image in meters using M3's resolution
        self.fullwidth  =   self.samples * self.resolution
        self.fullheight =   self.lines   * self.resolution


    def set_lat_lon(self):
        '''
        Set attributes for the latitude/longitude range of the captured image.

        Attributes set: self. {lon, minlon, maxlon, lat, minlat, maxlat}

        Define the extent of the image in lat/lon space, and give a center lat lon.
        '''
        latband = self.loc_img.GetRasterBand(2)
        self.minlat, self.maxlat = latband.ComputeStatistics(0)[:2]
        self.clat = (self.minlat + self.maxlat)/2

        lonband = self.loc_img.GetRasterBand(1)
        self.minlon, self.maxlon =  lonband.ComputeStatistics(0)[:2]
        self.clon = (self.minlon + self.maxlon)/2

    def set_coord_tfs(self):
        '''
        Set up two spatial references:
        self.projin:  M3 lat/lon reference 
        self.projout: hillshade x/y reference

        self.in2out_tf: transform lat/lon -> x/y
        self.out2in_tf: transform x/y     -> lat/lon
        '''
        self.projin = osr.SpatialReference()
        self.projin.SetFromUserInput('+proj=longlat +a=1737400 +b=1737400 +no_defs')

        self.projout = osr.SpatialReference()
        self.projout.SetFromUserInput(f'+proj=sinu +lon_0={self.clon} +x_0=0 +y_0=0 +a=1737400 +b=1737400 +units=m +no_defs')

        self.in2out_tf = osr.CoordinateTransformation(self.projin, self.projout)
        self.out2in_tf = osr.CoordinateTransformation(self.projout, self.projin)

    def set_inc_azm(self):
        '''
        Use the OBS image to get the incidence and azimuth angles when 
        the data was observed.

        self.azm set to the azimuth angle (deg)
        self.inc set to the indicence angle (deg)
        '''
        azmband = self.obs_img.GetRasterBand(1)
        incband = self.obs_img.GetRasterBand(2)
        self.azm = azmband.ComputeStatistics(0)[2]
        self.inc = incband.ComputeStatistics(0)[2]

    def set_bounding_box(self, m_offset):
        self.xmin, self.xmax = self.px-self.fullwidth/2 -m_offset, self.px+self.fullwidth/2 +m_offset
        self.ymin, self.ymax = self.py-self.fullheight/2-m_offset, self.py+self.fullheight/2+m_offset

        self.bound_minlon, self.bound_maxlat = self.out2in_tf.TransformPoint(self.xmin, self.ymax, 0)[:2]
        self.bound_maxlon = self.out2in_tf.TransformPoint(self.xmax, self.ymax, 0)[0]
        self.bound_minlat = self.out2in_tf.TransformPoint(self.xmin, self.ymin, 0)[1]

        self.bound_lat = (self.bound_maxlat+self.bound_minlat)/2
        self.bound_lon = (self.bound_maxlon+self.bound_minlon)/2

    def set_bounding_box_center(self, bbox_w_x, bbox_w_y=None):
        if bbox_w_y is None:
            bbox_w_y = bbox_w_x
        
        self.xmin, self.xmax = self.px-bbox_w_x//2, self.px+bbox_w_x//2
        self.ymin, self.ymax = self.py-bbox_w_y//2, self.py+bbox_w_y//2

        self.bound_minlon, self.bound_maxlat = self.out2in_tf.TransformPoint(self.xmin, self.ymax, 0)[:2]
        self.bound_maxlon = self.out2in_tf.TransformPoint(self.xmax, self.ymax, 0)[0]
        self.bound_minlat = self.out2in_tf.TransformPoint(self.xmin, self.ymin, 0)[1]

        self.bound_lat = (self.bound_maxlat+self.bound_minlat)/2
        self.bound_lon = (self.bound_maxlon+self.bound_minlon)/2


    def get_RDN_img(self, bands, outdir=None, outfn=None):
        '''
        Create a radiance product averaging a list of provided bands

        Returns the filename for the final product location

        If no output directory is given, defaults to current directory
        If no filename is given, defaults to {m3id}_RDN_average.tif
        '''
        if outfn is None:
            outfn = f'{self.m3id}_RDN_average.tif'
        if outdir is not None:
            outfn = outdir + '/' + outfn
        # Separate the bands into separate rasters
        bfns = [f'{self.m3id}_RDN_band{b}.tif' for b in bands]
        for b in range(len(bands)):
            gdal.Translate(bfns[b], f"{self.filename}_RDN.IMG",
                        options=gdal.TranslateOptions(
                            format='GTiff',
                            bandList=[bands[b]]
                        ))
        # Average all of the rasters together
        gdal_calc.Calc(calc='mean(a, axis=0)', a=bfns, outfile=outfn)
        # Remove the individual rasters
        for name in bfns: os.remove(name)
        # Return the output filename
        return outfn
    

    def get_RDN_8bit_crop(self, bands, w_x=None, w_y=None, outdir=None, outfn=None, contrast_mode='none'):
        if outfn is None:
            outfn = f'{self.m3id}_RDN_average_byte.tif'
        if outdir is not None:
            outfn = outdir + '/' + outfn

        rdn_orig = self.get_RDN_img(bands, outdir=outdir, outfn='rdnorig.tif')
        
        orig = np.array(Image.open(rdn_orig))

        if w_x is None:
            w_x = orig.shape[1]*self.resolution
        if w_y is None:
            w_x = w_y
            
        center = np.array(orig.shape)//2

        pix = [min(w_y//self.resolution, orig.shape[0])//2,
               min(w_x//self.resolution, orig.shape[1])//2]
        newimg = orig[center[0]-pix[0]:center[0]+pix[0],
                      center[1]-pix[1]:center[1]+pix[1]]
        
        if contrast_mode == 'equalize':
            newimg = exposure.rescale_intensity(newimg, out_range=(0,1))
            newimg = exposure.equalize_adapthist(newimg)
            newimg = np.array(exposure.rescale_intensity(newimg, out_range=(0,255))).astype(np.uint8)
        else:
            if contrast_mode == 'rescale':
                inbounds = (np.percentile(newimg, 1), np.percentile(newimg, 99))
            else:
                inbounds = (np.min(newimg), np.max(newimg))
            newimg = np.array(exposure.rescale_intensity(newimg,
                                                in_range=inbounds,
                                                out_range=(0,255))).astype(np.uint8)
        
        Image.fromarray(newimg).save(outfn)
        os.remove(rdn_orig)
        return outfn
