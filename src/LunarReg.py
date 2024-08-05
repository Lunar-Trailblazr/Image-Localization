import cv2 as cv
import numpy as np

from os import listdir
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

class SIFTDetector:
    def __init__(self):
        # Defining SIFT parameters
        nFeatures = 0
        nOctaveLayers = 4
        contrastThreshold = 0.04
        edgeThreshold = 10
        sigma = 1.6

        # Initializing SIFT detector
        self.sift = cv.SIFT_create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)

    def detect(self, im):
        # Convert input image to gray scale
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors
        kp, des = self.sift.detectAndCompute(gray, None)

        return kp, des

def ratioTest(knnMatches, ratio):
    # Lowe's ratio test to discard insufficiently strong matches
    return list(filter(lambda match: match[0].distance <= match[1].distance*ratio, knnMatches))

def symmetryTest(knnMatchesA, knnMatchesB):
    # Discard matches that are not symmetric
    matches = []
    for matchA in knnMatchesA:
        for matchB in knnMatchesB:
            if matchA[0].queryIdx == matchB[0].trainIdx and matchA[0].trainIdx == matchB[0].queryIdx:
                matches.append(cv.DMatch(matchA[0].queryIdx, matchA[0].trainIdx, matchA[0].distance))
    return matches

class BFMatcher:
    def __init__(self):
        normType = cv.NORM_L2
        crossCheck = False
        self.bf = cv.BFMatcher(normType, crossCheck=crossCheck)
        self.ratio = 0.65

    def match(self, desA, desB):
        # Ratio test using 2 nearest neighbors
        knnMatchesA = ratioTest(self.bf.knnMatch(desA, desB, 2), self.ratio)
        knnMatchesB = ratioTest(self.bf.knnMatch(desB, desA, 2), self.ratio)

        # Symmetry test
        matches = symmetryTest(knnMatchesA, knnMatchesB)

        return matches


def colorTransfer(source, dest):
    # Transfers the tone of source to dest
    source_gray = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
    dest_gray = cv.cvtColor(dest, cv.COLOR_BGR2GRAY)

    s_mean, s_std = cv.meanStdDev(source_gray, mask=cv.inRange(source_gray, 0, 254))
    d_mean, d_std = cv.meanStdDev(dest_gray, mask=cv.inRange(dest_gray, 0, 254))

    s_mean, s_std = np.hstack(s_mean)[0], np.hstack(s_std)[0]
    d_mean, d_std = np.hstack(d_mean)[0], np.hstack(d_std)[0]

    return cv.cvtColor(
            np.clip(((dest_gray-d_mean)*(s_std/d_std))+s_mean, 0, 255).astype(np.uint8),
            cv.COLOR_GRAY2BGR)

class IterativeFBM:
    def __init__(self, detector, matcher):
        # Defining FBM parameters
        self.detector = detector
        self.matcher = matcher
        self.maxIterations = 15
        self.reprojTolerance = 1.5
        self.redundancyTolerance = 3.

    def reprojTest(self, ptA, ptB, homography):
        # Tests the reprojection of ptA to ptB using homography
        # Returns True if the reprojection is within tolerance of ptB
        ptA = cv.perspectiveTransform(
                np.array([[ptA]], dtype=np.float64),
                homography
                )[0,0]
        ptB = np.array(ptB)

        dPt = ptA - ptB
        maxSqr = self.reprojTolerance*self.reprojTolerance
        distSqr = dPt[0]*dPt[0] + dPt[1]*dPt[1]

        return distSqr <= maxSqr

    def redundancyCheck(self, matches):
        # Checks if there are redundant matches based off of the match index on the ground truth
        # Returns the unique matches and retains the best match our of redundant matches
        uniqueMatches = []
        for i, matchA in enumerate(matches):
            isMin = True
            for j, matchB in enumerate(matches, start=i+1):
                if matchA.trainIdx == matchB.trainIdx and matchA.distance > matchB.distance:
                    isMin = False
                    break
            if isMin:
                uniqueMatches.append(matchA)
        return uniqueMatches

    def step(self, imA, imB, M, kpB, desB):
        imAprime = cv.warpPerspective(imA, M, imA.shape[:-1][::-1])
        kpAprime, desAprime = self.detector.detect(imAprime)
        matches = self.matcher.match(desAprime, desB)

        # Find homography
        ptsA, ptsB = [], []
        for match in matches:
            ptsA.append(kpAprime[match.queryIdx].pt)
            ptsB.append(kpB[match.trainIdx].pt)
        try:
            homography, mask = cv.findHomography(np.array(ptsA), np.array(ptsB), cv.RANSAC, 3.)
        except:
            try:
                homography, mask = cv.findHomography(np.array(ptsA), np.array(ptsB)) # Default to least-squares
            except Exception as e:
                assert len(ptsA) >= 4, 'Less than 4 matches found for homography!'
                raise e
        M = homography.dot(M)

        # Test points against reprojection
        matches = list(filter(lambda match:
            self.reprojTest(
                kpAprime[match.queryIdx].pt, kpB[match.trainIdx].pt, homography),
            matches))

        return (M, matches, kpAprime, desAprime)

    def match(self, imA, imB):
        # Color transfer
        imA = colorTransfer(imB, imA)

        # Match image heights
        height = min(imA.shape[0], imB.shape[0])
        imA = cv.resize(imA, dsize=(round((height/imA.shape[0])*imA.shape[1]), height))
        imB = cv.resize(imB, dsize=(round((height/imB.shape[0])*imB.shape[1]), height))

        # Initialize data
        kpB, desB = self.detector.detect(imB)

        chaoticHomography = np.identity(3)
        orderlyHomography = np.identity(3)
        orderlyKeypoints = []
        orderlyDescriptors = []
        orderlyMatches = []

        # Iteration
        i = 0
        while i <= self.maxIterations:
            i = i + 1

            # Chaotic step
            prevChaoticHomography = chaoticHomography.copy()
            try:
                (chaoticHomography, chaoticMatches,
                        kpAprime, desAprime) = self.step(imA, imB, chaoticHomography, kpB, desB)
            except Exception as e:
                if i == 1: raise e
                break

            # Redundancy check
            for match in chaoticMatches:
                orderlyMatches.append(cv.DMatch(
                    match.queryIdx + len(orderlyKeypoints), match.trainIdx, match.distance))
            orderlyMatches = self.redundancyCheck(orderlyMatches)

            # Append chaotic data
            for kp in kpAprime:
                # Invert keypoints to original image space
                kp.pt = tuple(
                        cv.perspectiveTransform(
                            np.array([[kp.pt]], dtype=np.float64),
                            np.linalg.inv(prevChaoticHomography)
                            )[0,0])
            chaoticKeypoints = list(kpAprime)
            chaoticDescriptors = list(desAprime)
            orderlyKeypoints = orderlyKeypoints + chaoticKeypoints
            orderlyDescriptors = orderlyDescriptors + orderlyKeypoints

            if len(orderlyMatches) > 0:
                # Find orderly homography
                ptsA, ptsB = [], []
                for match in orderlyMatches:
                    ptsA.append(orderlyKeypoints[match.queryIdx].pt)
                    ptsB.append(kpB[match.trainIdx].pt)
                try:
                    # Attempt to use RANSAC
                    orderlyHomography, mask = cv.findHomography(np.array(ptsA), np.array(ptsB), cv.RANSAC, 3.)
                except:
                    try:
                        # Default to least-squares if there aren't enough points
                        orderlyHomography, mask = cv.findHomography(np.array(ptsA), np.array(ptsB))
                    except Exception as e:
                        if i > 1: break
                        assert len(ptsA) >= 4, 'Less than 4 matches found for homography!'
                        raise e

                # Orderly step
                prevOrderlyHomography = orderlyHomography.copy()
                try:
                    (orderlyHomography, newOrderlyMatches,
                            kpAprime, desAprime) = self.step(imA, imB, orderlyHomography, kpB, desB)
                except Exception as e:
                    if i == 1: raise e
                    break

                # Redundancy check
                for match in newOrderlyMatches:
                    orderlyMatches.append(cv.DMatch(
                        match.queryIdx + len(orderlyKeypoints), match.trainIdx, match.distance))
                orderlyMatches = self.redundancyCheck(orderlyMatches)

                # Append orderly data
                for kp in kpAprime:
                    # Invert keypoints to original image space
                    kp.pt = tuple(
                            cv.perspectiveTransform(
                                np.array([[kp.pt]], dtype=np.float64),
                                np.linalg.inv(orderlyHomography)
                                )[0,0])
                orderlyKeypoints = orderlyKeypoints + list(kpAprime)
                orderlyDescriptors = orderlyDescriptors + list(desAprime)

        assert len(orderlyMatches) > 0, 'No matches found during iterative FBM!'

        return (chaoticHomography, orderlyHomography,
                chaoticKeypoints, orderlyKeypoints,
                chaoticDescriptors, orderlyDescriptors,
                kpB, desB,
                chaoticMatches, orderlyMatches,
                imA, imB, i)

#This code matches features between M3 data and LOLA hillshade data.
#This script calls upon detector and feature based matching scripts that can be found here: ../src/lunarreg/
#Those scripts are loaded as modules within lunarreg in this code.
#This is only the code that calls those scripts and generates outputs. Changes to the actual feature matching and detecting should be done on those scripts.

#The code expects three arguments: 1) a radiance file, 2) a hillshade file and 3) a name for the output files.
#The code is called by using python, i.e.: python M3-LOLA.py in-radiance.tif in-hillshade.tif outfile


#This function records a CSV file with match coordinates and saves a PNG file of the image-to-image plot
def record_plot(
        imA, imB,
        chaoticKeypoints, orderlyKeypoints, kpB,
        chaoticMatches, orderlyMatches,
        chaoticHomography, orderlyHomography, output):
    
    fig, axs = plt.subplots(3,2,figsize=(3,15))

    imAprimeC = cv.warpPerspective(imA, chaoticHomography, imA.shape[:-1][::-1])
    imAprimeO = cv.warpPerspective(imA, orderlyHomography, imA.shape[:-1][::-1])
    
    imgs = (imA, imB, imAprimeC, imB, imAprimeO, imB)

    for axidx in range(len(axs.flatten())):
        axs.flatten()[axidx].imshow(imgs[axidx])
        axs.flatten()[axidx].axis('off')

    #Create a CSV file and write a header line with the column names
    f = open(output + ".csv", "a")
    f.write("HillshadeColumn,HillshadeRow,M3Column,M3Row\n")

    for match in chaoticMatches:
        ptA = tuple(
                cv.perspectiveTransform(
                    np.array([[chaoticKeypoints[match.queryIdx].pt]], dtype=np.float64),
                    chaoticHomography)[0,0])
        con = ConnectionPatch(
                xyA=ptA, xyB=kpB[match.trainIdx].pt,
                coordsA='data', coordsB='data',
                axesA=axs[1,0], axesB=axs[1,1], 
                color='red', linewidth=.1)
        axs[1,1].add_artist(con)

    for match in orderlyMatches:
        ptA = tuple(
                cv.perspectiveTransform(
                    np.array([[orderlyKeypoints[match.queryIdx].pt]], dtype=np.float64),
                    orderlyHomography)[0,0])

        #Write the match coordinates for the orderly matches to the CSV file
        f.write(str(ptA[0]) + "," + str(ptA[1]) + "," + str(kpB[match.trainIdx].pt[0]) + "," + str(kpB[match.trainIdx].pt[1]) + "\n")

        con = ConnectionPatch(
                xyA=ptA, xyB=kpB[match.trainIdx].pt,
                coordsA='data', coordsB='data',
                axesA=axs[2,0], axesB=axs[2,1], color='red', linewidth=.1)
        axs[2,1].add_artist(con)

    plt.savefig(output + '.png', dpi=1200)
    f.close()

def match_images(A, B, ofn):
    imA = cv.imread(A)
    imB = cv.imread(B)
    # Load detectors and matchers
    detector = SIFTDetector()
    matcher = BFMatcher()
    matcher.ratio = 0.65
    # Load iterative FBM
    fbm = IterativeFBM(detector, matcher)
    fbm.maxIterations = 10

    (chaoticHomography,     orderlyHomography,
     chaoticKeypoints,      orderlyKeypoints,
     chaoticDescriptors,    orderlyDescriptors,
     kpB,                   desB,
     chaoticMatches,        orderlyMatches,
     imA, imB, i) = fbm.match(imA, imB)
    
    record_plot(imA, imB,
                chaoticKeypoints, orderlyKeypoints, kpB,
                chaoticMatches, orderlyMatches, 
                chaoticHomography, orderlyHomography, ofn)