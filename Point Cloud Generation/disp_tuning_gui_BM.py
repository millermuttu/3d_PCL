"""
This program generates Depth Map of an Image using Stereovision
Authors: Frederic Uhrweiller & Stephane Vujasinovic
"""

import numpy as np
import cv2
from blockmatchers import StereoBM, StereoSGBM
from calibration import StereoCalibration
from stereo_cameras import CalibratedPair

kernel = np.ones((3, 3), np.uint8)  # Filtering

white_pixel_count = 0  # Variable to store the total strength of all pixels
white_pixels = 0.0
offset = 21951047  # Summation of pixel strength for empty box

# Create StereoSGBM and prepare all parameters
stereo = cv2.StereoBM_create()

# Used for the filtered image
stereoR = cv2.ximgproc.createRightMatcher(stereo)
# Create another stereo for right

# WLS FILTER Parameters (I don't know how to decide this)
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

CamL = r"C:\muttu\Customer\Dr_ganesh\stereo_vision\1.Point Cloud Generation\stereo_images\left_8.ppm"
CamR = r"C:\muttu\Customer\Dr_ganesh\stereo_vision\1.Point Cloud Generation\stereo_images\right_8.ppm"
calib_folder = "calibration_files"

camera_pair = CalibratedPair(None,
                                StereoCalibration(input_folder=calib_folder),
                                stereo)

def nothing(x):
    pass

cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',600,600)

cv2.createTrackbar('numDisparities','disp',1,17,nothing)
cv2.createTrackbar('blockSize','disp',5,50,nothing)
cv2.createTrackbar('preFilterType','disp',1,1,nothing)
cv2.createTrackbar('preFilterSize','disp',2,25,nothing)
cv2.createTrackbar('preFilterCap','disp',5,62,nothing)
cv2.createTrackbar('textureThreshold','disp',10,100,nothing)
cv2.createTrackbar('uniquenessRatio','disp',15,100,nothing)
cv2.createTrackbar('speckleRange','disp',0,100,nothing)
cv2.createTrackbar('speckleWindowSize','disp',3,25,nothing)
cv2.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
cv2.createTrackbar('minDisparity','disp',5,25,nothing)


while True:
    # Start Reading Camera images
    frameR = cv2.imread(CamR)
    frameL = cv2.imread(CamL)

    if frameR is not None:
        image_pair = [frameL,frameR]
        Left_nice, Right_nice = camera_pair.calibration.rectify(image_pair)

        # Updating the parameters based on the trackbar positions
        numDisparities = cv2.getTrackbarPos('numDisparities', 'disp') * 16
        blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 5
        preFilterType = cv2.getTrackbarPos('preFilterType', 'disp')
        preFilterSize = cv2.getTrackbarPos('preFilterSize', 'disp') * 2 + 5
        preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
        textureThreshold = cv2.getTrackbarPos('textureThreshold', 'disp')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
        speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp') * 2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
        minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')

        # Setting the updated parameters before computing disparity map
        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterType(preFilterType)
        stereo.setPreFilterSize(preFilterSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setTextureThreshold(textureThreshold)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)

        grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
        grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)

        # Compute the 2 images for the Depth_image
        disp = stereo.compute(grayL, grayR)  # as type(np.float32)/ 16
        dispL = disp
        dispR = stereoR.compute(grayR, grayL)
        dispL = np.int16(dispL)
        dispR = np.int16(dispR)

        # Using the WLS filter
        filteredImg = wls_filter.filter(dispL, grayR, None, dispR)
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0,
                                    alpha=255, norm_type=cv2.NORM_MINMAX)
        filteredImg = np.uint8(filteredImg)
        # cv2.imshow('Disparity Map', filteredImg)
        disp = ((disp.astype(np.float32) / 16) - minDisparity) / numDisparities

        # Resize the image for faster executions
        dispR = cv2.resize(disp, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)

        # Filtering the Results with a closing filter
        # Apply an morphological filter for closing little "black" holes in the
        #  picture(Remove noise)
        closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, kernel)

        dispc = (closing - closing.min()) * 255
        # Convert the type of the matrix from float32 to uint8, to show the results
        # with the function cv2.imshow()
        dispC = dispc.astype(np.uint8)
        disp_Color = cv2.applyColorMap(dispC, cv2.COLORMAP_OCEAN)
        filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_OCEAN)
        jet_color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_JET)
        bone_color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_BONE)

        cv2.imshow('Hot Color Mapping', jet_color)
        cv2.imshow("disparity", bone_color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        CamL = cv2.VideoCapture(CamL)
        CamR = cv2.VideoCapture(CamR)

# wb.save("output_data.xlsx")  # Save excel file
CamR.release()
CamL.release()
cv2.destroyAllWindows()
