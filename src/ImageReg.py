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

    # def match(self, im1, im2, k=2, r=0.65):
        
    #     # Get keypoints and descriptors for both images
    #     kp1, des1 = self.detector.detectAndCompute(im1, None)
    #     kp2, des2 = self.detector.detectAndCompute(im2, None)

    #     return self.match_from_descriptors(kp1, des1, kp2, des2, k=k, r=r)

    def match(self, kp1, des1, kp2, des2, k=2, r=0.65):
        '''
        Provides the keypoints and match information for two passed keypoint/descriptor pairs, 
        using the saved detector and matcher objects
        '''
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
        
        # Get the matches from descriptors using KNN matcher and return
        # keypoints/descriptor matches.
        return self.knn_ratio(des1, des2, k=k, r=r)

def colorTransfer(source, dest):
    # Transfers the tone of source to dest
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
    
    def match(self, im1, kp2, des2, M_in=np.eye(3)):
        im1prime = cv.warpPerspective(im1, M_in, im1.shape[::-1])
        kp1, des1 = self.fmobj.detector.detectAndCompute(im1prime, None)
        matches = self.fmobj.match(kp1, des1, kp2, des2)
        assert len(matches) >= 4, "UNDER 4 MATCHES TO CALCULATE HOMOGRAPHY - MATCH FAILED"

        ptsA = np.array([kp1[k[0].queryIdx].pt for k in matches])
        ptsB = np.array([kp2[k[0].trainIdx].pt for k in matches])
        try:
            homography, mask = cv.findHomography(ptsA, ptsB, cv.RANSAC, 3.)
        except:
            homography, mask = cv.findHomography(ptsA, ptsB, 0)
        H = homography.dot(M_in)

        return kp1,des1,matches,H
    
    def step(self, im1, kp2, des2, M_in, k, kp_f, des_f, matches_f):
        prev_H = M_in.copy()

        try:
            (kp1, des1, matches, M_out) = self.match(im1, kp2, des2, M_in=M_in)
        except Exception as e:
            if k==0: raise e
            return False, False, False, k, False

        new_matches = [cv.DMatch(m[0].queryIdx + len(kp_f), 
                                 m[0].trainIdx, 
                                 m[0].distance) for m in matches]
        tidxs = [m.trainIdx for m in new_matches]
        for idx, prev_M in enumerate(matches_f):
            if prev_M.trainIdx in tidxs:
                loc = tidxs.index(prev_M.trainIdx)
                if new_matches[loc].distance < prev_M.distance:
                    matches_f[idx] = cv.DMatch(new_matches[loc].queryIdx,
                                               new_matches[loc].trainIdx,
                                               new_matches[loc].distance)
                del tidxs[loc]
                del new_matches[loc]
        
        for kp in kp1:
            kp.pt = tuple(cv.perspectiveTransform(
                np.array([[kp.pt]], dtype=np.float64),
                np.linalg.pinv(prev_H)
            )[0,0])
        return M_out, list(kp1), list(des1), list(new_matches), True

    def iterative_match(self, im1, im2, colormatch=True, n_iters=10):
        if colormatch:
            im1 = colorTransfer(im2, im1)
        
        kp2, des2 = self.fmobj.detector.detectAndCompute(im2, None)
        H_cur = np.eye(3)
        H_f = np.eye(3)
        kp_f = []
        des_f = []
        matches_f = []

        for k in range(n_iters):
            # Try to take another step in the iterative FBM
            H_cur, kp1, des1, new_matches, s = self.step(im1, kp2, des2, H_cur, k, kp_f, des_f, matches_f)
            
            if s == False:
                break

            kp_f = kp_f+kp1
            des_f = des_f+des1
            matches_f = matches_f+new_matches
            
            if len(matches_f) > 4:
                ptsA = np.array([kp_f[k.queryIdx].pt for k in matches_f])
                ptsB = np.array([kp2[k.trainIdx].pt for k in matches_f])
                try:
                    H_f, mask = cv.findHomography(ptsA, ptsB, cv.RANSAC, 3)
                except:
                    H_f, mask = cv.findHomography(ptsA, ptsB, 0)
                
                H_f, kp1, des1, new_matches, s = self.step(im1, kp2, des2, H_f, k, kp_f, des_f, matches_f)
                if s == False:
                    break
                
                kp_f = kp_f+kp1
                des_f = des_f+des1
                matches_f = matches_f+new_matches
        assert len(matches_f) > 0, "NO MATCHES FOUND"
        return H_f, kp_f, kp2, matches_f
    
    def match_and_plot(self, outfn, im1, im2):
        (H_f, kp_f, kp2, matches_f) = self.iterative_match(im1, im2)
        img3 = cv.drawMatches(im1,kp_f,im2,kp2,matches_f,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imwrite(outfn, img3)
        return outfn
