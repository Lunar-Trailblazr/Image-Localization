# Kevin Gauld
# June 2024

# Implements SIFT-based Feature Matching
import cv2 as cv
import numpy as np
import warnings
from skimage import exposure
import logging

#################################################
############### SIFT DETECTOR ###################
#################################################

# Default settings for SIFT detector.
detector_settings = {
    'nfeatures': 0,
    'nOctaveLayers': 4,
    'contrastThreshold': 0.03,
    'edgeThreshold': 15,
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

    def knn_ratio(self, des1, des2, k=2, r=0.8):
        '''
        Runs KNN using the matcher on the two descriptor groups, then filters to those
        with a distance metric below `r` (default value 0.8)
        '''
        matches = self.matcher.knnMatch(des1, des2, k=k)
        thresh = filter(lambda match: match[0].distance <= r*match[1].distance, matches)
        return list(thresh)

    def match(self, kp1, des1, kp2, des2, k=2, r=0.8):
        '''
        Provides the keypoints and match information for two passed keypoint/descriptor pairs, 
        using the saved detector and matcher objects
        '''
        # Due to some assumptions within OpenCV, clip to first 2^18 keypoints
        # See https://github.com/opencv/opencv/issues/5700
        if len(des1) >= 1<<18:
            logging.warning(f"Too many kps (hillshade {len(des1)}), only using first 2^18 kps")
            des1 = des1[:1<<18-1]
            kp1 = tuple(np.array(kp1)[:1<<18-1])
        
        if len(des2) >= 1<<18:
            logging.warning(f"Too many kps (m3 {len(des2)}), only using first 2^18 kps")
            des2 = des2[:(1<<18)-1]
            kp2 = tuple(np.array(kp2)[:(1<<18)-1])
        
        # Get the matches from descriptors using KNN matcher and return
        # keypoints/descriptor matches.
        return self.knn_ratio(des1, des2, k=k, r=r)
    
    def match_image_to_kp(self, im1, shape2, kp2, des2, M_in=np.eye(3), k=2, r=0.8):
        # Warp the input based on the current homography and the shape of the output
        im1prime = cv.warpPerspective(im1, M_in, shape2[::-1])

        # Detect the keypoints for the warped image, throwing an error if 
        # there are no keypoints detected.
        kp1, des1 = self.detector.detectAndCompute(im1prime, None)
        assert len(kp1) > 0, f"NO KEYPOINTS GENERATED!"
        assert des1 is not None, f"NO DESCRIPTORS GENERATED!"

        # Attempt to match the keypoints for the first and second images,
        # throwing an error if there are <4 matches
        matches = self.match(kp1, des1, kp2, des2, k=k, r=r)
        assert len(matches) >= 4, f"UNDER 4 MATCHES TO CALCULATE HOMOGRAPHY (FOUND {len(matches)}) - MATCH FAILED"

        # Compute a homography for the match. If no homography can be
        # computed with RANSAC, retry with least squares.
        ptsA = np.array([kp1[k[0].queryIdx].pt for k in matches])
        ptsB = np.array([kp2[k[0].trainIdx].pt for k in matches])
        try: # RANSAC
            homography, mask = cv.findHomography(ptsA, ptsB, cv.RANSAC, 3.)
            if homography is None:
                logging.error(f"RANSAC FAILURE! Trying Least squares. N_MATCHES: {len(ptsA)}")
                raise Exception()
        except: # Least squares
            homography, mask = cv.findHomography(ptsA, ptsB, 0)

        # Combine the output homography with the input and return
        H = homography.dot(M_in)
        return kp1,des1,matches,H
    
    def match_image_to_image(self, im1, im2, M_in=np.eye(3), k=2, r=0.8):
        kp2, des2 = self.detector.detectAndCompute(im2, None)
        return self.match_image_to_kp(im1, kp2, im2.shape, des2, M_in=M_in, k=k, r=r)


###################################################
############### Iterative MATCH ###################
###################################################


def colortransfer(src, dst):
    mask = np.where(dst > 0)
    get_bnds = lambda src : (np.percentile(src, 1), np.percentile(src,99))
    # srcremap = np.array(exposure.rescale_intensity(src, in_range=get_bnds(src))).astype(np.uint8)
    dst_mask = np.array(exposure.rescale_intensity(dst[mask], in_range=get_bnds(dst[mask]))).astype(np.uint8)
    dst_mask = np.array(exposure.match_histograms(dst_mask, src.flatten())).astype(np.uint8)
    dstremap = dst.copy()
    dstremap[mask] = dst_mask
    return src, dstremap

class IterativeMatcher:
    '''
    Iteratively improves matching between images.
    '''
    def __init__(self, fmobj=FeatureMatcher()):
        self.fmobj = fmobj
    
    def step(self, im1, M_in, k, kp_f, des_f, matches_f):
        '''
        Takes one step in the iterative FBM process, with an input image and homography alongside
        the keypoints and descriptors for the match image and the total list of keypoints for
        the process.
        '''
        # Retains a copy of the previous homography
        prev_H = M_in.copy()

        # Finds the matches between the given image and the destination 
        # keypoints using the matcher object. If this match fails, `None` is
        # returned for all fields.
        try:
            (kp1, des1, matches, M_out) = self.fmobj.match_image_to_kp(im1, self.shape2, self.kp2, self.des2, M_in=M_in)
        except Exception as e:
            if k==0: raise e
            return M_in, [], [], []
        
        # Create list of the new matches, with query indexes shifted based on
        # the currently matched keypoints. As a note, each keypoint contains
        # the query index in the source image, the train index in the destination
        # image, and the distance/similarity of that match. The lower the distance,
        # the more similar the matched points are.
        new_matches = [cv.DMatch(m[0].queryIdx + len(kp_f), 
                                 m[0].trainIdx, 
                                 m[0].distance) for m in matches]
        
        # Removes duplicate matches, prioritizing those with a lower distance. 
        # If there is a duplicate match, update the pre-existing match with the new values 
        # if the new distance is lower. Either way, delete the match from the list to be added 
        # to the cumulative matches.
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
        
        # Transform each keypoint back to the original warping (undo the initial homography)
        for kp in kp1:
            kp.pt = tuple(cv.perspectiveTransform(
                np.array([[kp.pt]], dtype=np.float64),
                np.linalg.pinv(prev_H)
            )[0,0])
        
        # Return the new homography, keypoints, descriptors, and matches
        return M_out, list(kp1), list(des1), list(new_matches)

    def iterative_match(self, im1, im2, 
                        colormatch=True, cmatch_fns=None,
                        n_iters=10, H_init=np.eye(3)):
        '''
        Performs iterative matching between two images
        '''
        # Optionally choose to color-match image 1 and 2 to potentially
        # improve the match quality. On by default. If a filename is provided, will also
        # save the color matched image to disk.
        if colormatch:
            im1, im2 = colortransfer(im1, im2)
            if cmatch_fns is not None:
                cv.imwrite(cmatch_fns[0], im1)
                cv.imwrite(cmatch_fns[1], im2)
        
        # Detect keypoints for the second image. Every iteration will only
        # warp the input first image, so we ensure consistency by precomputing
        # and saving the keypoints for the second image for each use.
        self.kp2, self.des2 = self.fmobj.detector.detectAndCompute(im2, None)
        self.shape2 = im2.shape

        # Initialize homography and keypoint match storage
        H_cur = H_init
        H_f = H_init
        kp_f = []
        des_f = []
        matches_f = []

        # Perform matching for however many iterations are requested (10 by default)
        for k in range(n_iters):
            # Take the first step of iterative FBM. For this step, the current homography will be used to find 
            # new matches. From here, we have this iteration's new keypoints and descriptors.
            H_cur, kp1, des1, new_matches = self.step(im1, H_cur, k, kp_f, des_f, matches_f)
            if H_cur is None:
                break

            # Here, we combine the new keypoints and descriptors with those that are already known
            kp_f = kp_f+kp1
            des_f = des_f+des1
            matches_f = matches_f+new_matches
            
            # If there are enough new matches, find a new homography which uses all matches up to this point
            # in the iterative FBM process.
            if len(matches_f) > 4:
                ptsA = np.array([kp_f[k.queryIdx].pt for k in matches_f])
                ptsB = np.array([self.kp2[k.trainIdx].pt for k in matches_f])
                try:
                    H_f, mask = cv.findHomography(ptsA, ptsB, cv.RANSAC, 3)
                    if H_f is None:
                        logging.error(f"RANSAC FAILURE! Trying Least squares. N_MATCHES: {len(ptsA)}")
                        raise Exception()
                except:
                    H_f, mask = cv.findHomography(ptsA, ptsB, 0)
                
                # Use this new cumulative homography to find tie points in the image much like in the first
                # step, adding these as well to the lists of cumulative matches.
                H_f, kp1, des1, new_matches = self.step(im1, H_f, k, kp_f, des_f, matches_f)
                if H_f is None:
                    break
                
                kp_f = kp_f+kp1
                des_f = des_f+des1
                matches_f = matches_f+new_matches

        # Throw an error if there are no matches, otherwise return the detected matches.
        assert len(matches_f) > 0, "NO MATCHES FOUND"
        ptsA = np.array([kp_f[k.queryIdx].pt for k in matches_f])
        ptsB = np.array([self.kp2[k.trainIdx].pt for k in matches_f])
        try:
            H_f, mask = cv.findHomography(ptsA, ptsB, cv.RANSAC, 3)
            if H_f is None:
                logging.error(f"RANSAC FAILURE! Trying Least squares. N_MATCHES: {len(ptsA)}")
                raise Exception()
        except:
            H_f, mask = cv.findHomography(ptsA, ptsB, 0)
        return H_f, kp_f, self.kp2, matches_f, mask
    
    def match_and_plot(self, outfns, im1, im2, 
                       colormatch=True, cmatch_fns=None, 
                       n_iters=10, H_init=np.eye(3)):
        # Perform the iterative matching
        (H_f, kp_f, kp2, matches_f, mask) = self.iterative_match(im1, im2, 
                                                           colormatch=colormatch, cmatch_fns=cmatch_fns, 
                                                           n_iters=n_iters, H_init=H_init)
        
        if colormatch:
            im1, im2 = colortransfer(im1,im2)
        
        # Create images for output to assess match quality
        img3 = cv.warpPerspective(im1, H_f, im2.shape[::-1])//2
        cv.imwrite(outfns[2], img3)
        img3 += im2//2
        img4 = cv.drawMatches(im1,kp_f,im2,kp2,matches_f,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img5 = cv.drawKeypoints(im1, kp_f, im1, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img6 = cv.drawKeypoints(im2, kp2, im2, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imwrite(outfns[0], img3)
        cv.imwrite(outfns[1], img4)
        cv.imwrite(outfns[3], img5)
        cv.imwrite(outfns[4], img6)

        return H_f, kp_f, kp2, matches_f, mask, outfns
