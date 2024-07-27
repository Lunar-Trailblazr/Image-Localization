# Kevin Gauld
# June 2024

# Implements SIFT-based Feature Matching

import cv2 as cv
import numpy as np
import warnings

# Default settings for SIFT detector.
detector_settings = {
    'nfeatures': 0,
    'nOctaveLayers': 4,
    'contrastThreshold': 0.04,
    'edgeThreshold': 10,
    'sigma': 1.6
}

class FeatureMatcher:
    '''
    Provides SIFT matcher and detector. Allows for computation of matches betweeen
    two images. 
    '''
    def __init__(self, 
                 detector=cv.SIFT_create(**detector_settings),
                 matcher=cv.BFMatcher()):
        '''
        Initialize FeatureMatcher with a default detector and matcher, or
        optionally pass in a detector and matcher OpenCV object. Defaults for
        the detector are at the top of file, defaults for the matcher are the 
        OpenCV defaults for a BFMatcher.
        '''
        self.detector = detector
        self.matcher = matcher

    def knn_ratio(self, des1, des2, k=2, r=0.65):
        '''
        Runs KNN using the matcher on the two descriptor groups, then filters to those
        with a distance metric below `r` (default value 0.65)
        '''
        matches = self.matcher.knnMatch(des1, des2, k=k)
        thresh = filter(lambda match: match[0].distance <= r*match[1].distance, matches)
        return list(thresh)

    def match(self, im1, im2, k=2, r=0.65):
        '''
        Provides the keypoints and match information for two passed images, using the
        saved detector and matcher objects
        '''
        # Get keypoints and descriptors for both images
        kp1, des1 = self.detector.detectAndCompute(im1, None)
        kp2, des2 = self.detector.detectAndCompute(im2, None)

        # Due to some assumptions within OpenCV, clip to first 2^18 keypoints
        # See https://github.com/opencv/opencv/issues/5700
        if len(des1) >= 1<<18:
            warnings.warn(f"Too many kps (hillshade {len(des1)}), only using first 2^18 kps")
            des1 = des1[:1<<18-1]
            kp1 = tuple(np.array(kp1)[:1<<18-1])
        if len(des2) >= 1<<18:
            warnings.warn(f"Too many kps (m3 {len(des2)}), only using first 2^18 kps")
            des2 = des2[:(1<<18)-1]
            kp2 = tuple(np.array(kp2)[:(1<<18)-1])
        
        # Get the matches using the descriptors and a KNN match
        matches = self.knn_ratio(des1, des2, k=k, r=r)
        
        # Return keypoints and the descriptor matches
        return kp1, kp2, matches


def colorTransfer(source, dest):
    # Transfers the tone of source to dest
    # source_gray = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
    # dest_gray = cv.cvtColor(dest, cv.COLOR_BGR2GRAY)

    s_mean, s_std = cv.meanStdDev(source, mask=cv.inRange(source, 0, 254))
    d_mean, d_std = cv.meanStdDev(dest, mask=cv.inRange(dest, 0, 254))

    s_mean, s_std = np.hstack(s_mean)[0], np.hstack(s_std)[0]
    d_mean, d_std = np.hstack(d_mean)[0], np.hstack(d_std)[0]

    return np.clip(((dest-d_mean)*(s_std/d_std))+s_mean, 0, 255).astype(np.uint8)

class IterativeMatcher:
    '''
    Iteratively improves matching between images.
    Only does one iteration for now
    TODO: Add more iterations (if necessary)
    '''

    def __init__(self, fmobj=FeatureMatcher()):
        self.fmobj = fmobj
    
    def match_and_plot(self, im1, im2, ofn='match', M_in=np.eye(3), colormatch=True):
        print('m1')
        kp1, kp2, matches, H = self.match(im1, im2,
                                          M_in=M_in,
                                          colormatch=colormatch)
        print('m2')
        # im1prime = cv.warpPerspective(im1, H, im1.shape[::-1])
        # kp1, kp2, matches, H = self.match(im1prime, im2)
        out_img = cv.drawMatchesKnn(im1, kp1, 
                                    im2, kp2, 
                                    matches, None, 
                                    flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        cv.imwrite(f'{ofn}.png', out_img)

    def match(self, im1, im2, M_in=np.eye(3), colormatch=True):
        if colormatch:
            im2 = colorTransfer(im1, im2)
        kp1, kp2, matches = self.fmobj.match(im1, im2)

        if len(matches) < 4:
            raise Exception("UNDER 4 MATCHES TO CALCULATE HOMOGRAPHY - MATCH FAILED")

        ptsA = np.array([kp1[k[0].queryIdx].pt for k in matches])
        ptsB = np.array([kp2[k[0].trainIdx].pt for k in matches])

        homography, mask = cv.findHomography(ptsA, ptsB, cv.RANSAC, 3.)
        print(type(homography))
        return kp1,kp2,matches,homography.dot(M_in)
    
