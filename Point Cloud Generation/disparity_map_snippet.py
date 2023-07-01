import cv2
from matplotlib import pyplot as plt
import numpy as np

imgL = cv2.imread(r'C:\muttu\Customer\Dr_ganesh\Arducam\stereo_images\left_image.jpg',0)
imgR = cv2.imread(r'C:\muttu\Customer\Dr_ganesh\Arducam\stereo_images\right_image.jpg',0)

min_disp = 11
num_disp = 96
kernel = np.ones((3, 3), np.uint8)  # Filtering

stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=7)

# stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp,
#                                blockSize=3, uniquenessRatio=10,
#                                speckleWindowSize=100, speckleRange=32,
#                                disp12MaxDiff=5, P1=8 * 3 * 3 ** 2,
#                                P2=32 * 3 * 3 ** 2)

stereoR = cv2.ximgproc.createRightMatcher(stereo)
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

disp = stereo.compute(imgL,imgR)
dispL = disp
dispR = stereoR.compute(imgR, imgL)
dispL = np.int16(dispL)
dispR = np.int16(dispR)
disp = np.uint8(disp)

filteredImg = wls_filter.filter(dispL, imgL, None, dispR)
filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0,
                            alpha=255, norm_type=cv2.NORM_MINMAX)
filteredImg = np.uint8(filteredImg)
# cv2.imshow('Disparity Map', filteredImg)
disp = ((disp.astype(np.float32) / 16) - min_disp) / num_disp

# Resize the image for faster executions
dispR = cv2.resize(disp, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)

# Filtering the Results with a closing filter
# Apply an morphological filter for closing little "black" holes in the
#  picture(Remove noise)
closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, kernel)

# Colors map
dispc = (closing - closing.min()) * 255
# Convert the type of the matrix from float32 to uint8, to show the results
# with the function cv2.imshow()
dispC = dispc.astype(np.uint8)
disp_Color = cv2.applyColorMap(dispC, cv2.COLORMAP_OCEAN)
filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_OCEAN)
jet_color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_JET)
bone_color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_BONE)

# Show the result for the Depth_image
# cv2.imshow('Disparity', disp)
# cv2.imshow('Closing', closing)
# cv2.imshow('Color Depth', disp_Color)
# cv2.imshow('Filtered Color Depth', filt_Color)
jet_color = cv2.resize(jet_color, (640, 480))
bone_color = cv2.resize(bone_color, (640, 480))
cv2.imshow('Jet Color Mapping', jet_color[0:480, 130:640])
cv2.imshow('bone color', bone_color)
img = cv2.cvtColor(bone_color, cv2.COLOR_BGR2GRAY)
img = img[0:480, 130:640]   # Cropping the unwanted  vertical strip
cv2.imshow('Grayscaled Image', img)
cv2.waitKey()
