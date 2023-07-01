#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
from time import sleep
import datetime
import argparse
import open3d as o3d
import os

from projector_3d import PointCloudVisualizer
import matplotlib.pyplot as plt

'''
a main file to read the camera from arducam (mono cameras) with calibration done manually using the scripts given by Depthai.
here we are using stereo images to convert them to point cloud and save them as .pcd files which can be visualized uisng open3d package
'''


parser = argparse.ArgumentParser()
parser.add_argument("-pcl", "--pointcloud", help="enables point cloud convertion and visualization", default=False, action="store_true")
parser.add_argument("-static", "--static_frames", default=False, action="store_true",
                    help="Run stereo on static frames passed from host 'dataset' folder")
parser.add_argument("-save","--save_pcl", help="enable if you want to save the point cloud", default=False, action="store_true")
args = parser.parse_args()

point_cloud    = args.pointcloud   # Create point cloud visualizer. Depends on 'out_rectified'
save_pcl = args.save_pcl


# StereoDepth config options. TODO move to command line options
source_camera  = not args.static_frames
out_depth      = True  # Disparity by default
out_rectified  = True   # Output and display rectified streams
lrcheck  = True   # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = False   # Better accuracy for longer distance, fractional disparity 32-levels
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
median   = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

# Sanitize some incompatible options
if lrcheck or extended or subpixel:
    median   = dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF # TODO

print("StereoDepth config options:")
print("    Left-Right check:  ", lrcheck)
print("    Extended disparity:", extended)
print("    Subpixel:          ", subpixel)
print("    Median filtering:  ", median)

# TODO add API to read this from device / calib data
# right_intrinsic = [[860.0, 0.0, 1280.0], [0.0, 860.0, 360.0], [0.0, 0.0, 1.0]]
# right_intrinsic = [[807.0106811523438,0.0,627.4284057617188],[0.0,806.4772338867188,395.004150390625],[0.0,0.0,1.0]]

pcl_converter = None
if point_cloud:
    if out_rectified:
        try:
            from projector_3d import PointCloudVisualizer
        except ImportError as e:
            raise ImportError(f"\033[1;5;31mError occured when importing PCL projector: {e}. Try disabling the point cloud \033[0m ")
        # pcl_converter = PointCloudVisualizer(right_intrinsic, 1280, 720)
    else:
        print("Disabling point-cloud visualizer, as out_rectified is not set")

def create_rgb_cam_pipeline():
    print("Creating pipeline: RGB CAM -> XLINK OUT")
    pipeline = dai.Pipeline()

    cam          = pipeline.createColorCamera()
    xout_preview = pipeline.createXLinkOut()
    xout_video   = pipeline.createXLinkOut()

    cam.setPreviewSize(540, 540)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)

    xout_preview.setStreamName('rgb_preview')
    xout_video  .setStreamName('rgb_video')

    cam.preview.link(xout_preview.input)
    cam.video  .link(xout_video.input)

    streams = ['rgb_preview', 'rgb_video']

    return pipeline, streams

def create_mono_cam_pipeline():
    print("Creating pipeline: MONO CAMS -> XLINK OUT")
    pipeline = dai.Pipeline()

    cam_left   = pipeline.createMonoCamera()
    cam_right  = pipeline.createMonoCamera()
    xout_left  = pipeline.createXLinkOut()
    xout_right = pipeline.createXLinkOut()

    cam_left .setBoardSocket(dai.CameraBoardSocket.LEFT)
    cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    for cam in [cam_left, cam_right]: # Common config
        cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        #cam.setFps(20.0)

    xout_left .setStreamName('left')
    xout_right.setStreamName('right')

    cam_left .out.link(xout_left.input)
    cam_right.out.link(xout_right.input)

    streams = ['left', 'right']

    return pipeline, streams

def create_stereo_depth_pipeline(from_camera=True):
    print("Creating Stereo Depth pipeline: ", end='')
    if from_camera:
        print("MONO CAMS -> STEREO -> XLINK OUT")
    else:
        print("XLINK IN -> STEREO -> XLINK OUT")
    pipeline = dai.Pipeline()

    if from_camera:
        cam_left      = pipeline.createMonoCamera()
        cam_right     = pipeline.createMonoCamera()
    else:
        cam_left      = pipeline.createXLinkIn()
        cam_right     = pipeline.createXLinkIn()
    stereo            = pipeline.createStereoDepth()
    xout_left         = pipeline.createXLinkOut()
    xout_right        = pipeline.createXLinkOut()
    xout_depth        = pipeline.createXLinkOut()
    xout_disparity    = pipeline.createXLinkOut()
    xout_rectif_left  = pipeline.createXLinkOut()
    xout_rectif_right = pipeline.createXLinkOut()

    if from_camera:
        cam_left .setBoardSocket(dai.CameraBoardSocket.LEFT)
        cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        for cam in [cam_left, cam_right]: # Common config
            cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            #cam.setFps(20.0)
    else:
        cam_left .setStreamName('in_left')
        cam_right.setStreamName('in_right')

    stereo.initialConfig.setConfidenceThreshold(200)
    stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout
    stereo.initialConfig.setMedianFilter(median) # KERNEL_7x7 default
    stereo.setLeftRightCheck(lrcheck)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(subpixel)
    if from_camera:
        # Default: EEPROM calib is used, and resolution taken from MonoCamera nodes
        # stereo.loadCalibrationFile(path)
        pass
    else:
        stereo.setEmptyCalibration() # Set if the input frames are already rectified
        stereo.setInputResolution(1280, 720)

    xout_left        .setStreamName('left')
    xout_right       .setStreamName('right')
    xout_depth       .setStreamName('depth')
    xout_disparity   .setStreamName('disparity')
    xout_rectif_left .setStreamName('rectified_left')
    xout_rectif_right.setStreamName('rectified_right')

    cam_left .out        .link(stereo.left)
    cam_right.out        .link(stereo.right)
    stereo.syncedLeft    .link(xout_left.input)
    stereo.syncedRight   .link(xout_right.input)
    stereo.depth         .link(xout_depth.input)
    stereo.disparity     .link(xout_disparity.input)
    if out_rectified:
        stereo.rectifiedLeft .link(xout_rectif_left.input)
        stereo.rectifiedRight.link(xout_rectif_right.input)

    streams = ['left', 'right']
    if out_rectified:
        streams.extend(['rectified_left', 'rectified_right'])
    streams.extend(['disparity', 'depth'])

    return pipeline, streams

# The operations done here seem very CPU-intensive, TODO
def convert_to_cv2_frame(name, image,right_intrinsic,pcl_converter):
    global last_rectif_right
    baseline = 25 #mm
    focal = right_intrinsic[0][0]
    max_disp = 96
    disp_type = np.uint8
    disp_levels = 1
    pcd = None
    if (extended):
        max_disp *= 2
    if (subpixel):
        max_disp *= 32;
        disp_type = np.uint16  # 5 bits fractional disparity
        disp_levels = 32

    data, w, h = image.getData(), image.getWidth(), image.getHeight()
    print(w,h)
    # TODO check image frame type instead of name
    if name == 'rgb_preview':
        frame = np.array(data).reshape((3, h, w)).transpose(1, 2, 0).astype(np.uint8)
    elif name == 'rgb_video': # YUV NV12
        yuv = np.array(data).reshape((h * 3 // 2, w)).astype(np.uint8)
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    elif name == 'depth':
        # TODO: this contains FP16 with (lrcheck or extended or subpixel)
        frame = np.array(data).astype(np.uint8).view(np.uint16).reshape((h, w))
    elif name == 'disparity':
        disp = np.array(data).astype(np.uint8).view(disp_type).reshape((h, w))

        # Compute depth from disparity (32 levels)
        with np.errstate(divide='ignore'): # Should be safe to ignore div by zero here
            depth = (disp_levels * baseline * focal / disp).astype(np.uint16)

        if 1: # Optionally, extend disparity range to better visualize it
            frame = (disp * 255. / max_disp).astype(np.uint8)

        if 1: # Optionally, apply a color map
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
            #frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

        if point_cloud:
            if 0: # Option 1: project colorized disparity
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _,pcd = pcl_converter.rgbd_to_projection(depth, frame_rgb, True, save_pcl)
            else: # Option 2: project rectified right
                pcl_converter.rgbd_to_projection(depth, last_rectif_right, False, save_pcl)
            pcl_converter.visualize_pcd()
            # if save_pcl:
                # pcl_converter.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                # point_cloud_filename = "point_cloud.pcd"
                # o3d.io.write_point_cloud(point_cloud_filename, pcl_converter)
                # pcl_converter.save_pcd("point_cloud.ply")


    else: # mono streams / single channel
        frame = np.array(data).reshape((h, w)).astype(np.uint8)
        if name == 'rectified_right':
            last_rectif_right = frame
    return frame


def test_pipeline():
    print("Creating DepthAI device")
    with dai.Device() as device:
        print('Read calib data')
        calibData = device.readCalibration()
        print(calibData)
        M_right = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, 1280, 720))
        print("RIGHT Camera resized intrinsics...")
        print(M_right)

        cams = device.getConnectedCameras()
        depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
        if depth_enabled:
            pipeline, streams = create_stereo_depth_pipeline(source_camera)
        else:
            pipeline, streams = create_rgb_cam_pipeline()
        #pipeline, streams = create_mono_cam_pipeline()

        print("Starting pipeline")
        device.startPipeline(pipeline)

        # starting point cloud
        pcl_converter = PointCloudVisualizer(M_right, 1280, 720)

        in_streams = []
        if not source_camera:
            # Reversed order trick:
            # The sync stage on device side has a timeout between receiving left
            # and right frames. In case a delay would occur on host between sending
            # left and right, the timeout will get triggered.
            # We make sure to send first the right frame, then left.
            in_streams.extend(['in_right', 'in_left'])
        in_q_list = []
        inStreamsCameraID = []
        for s in in_streams:
            q = device.getInputQueue(s)
            in_q_list.append(q)
            inStreamsCameraID = [dai.CameraBoardSocket.RIGHT, dai.CameraBoardSocket.LEFT]

        # Create a receive queue for each stream
        q_list = []
        for s in streams:
            q = device.getOutputQueue(s, 8, blocking=False)
            q_list.append(q)

        # Need to set a timestamp for input frames, for the sync stage in Stereo node
        timestamp_ms = 0
        index = 1
        while True:
            # Handle input streams, if any
            if in_q_list:
                dataset_size = 2  # Number of image pairs
                frame_interval_ms = 33
                for i, q in enumerate(in_q_list):
                    name = q.getName()
                    path = 'dataset/'+ name + '.png'
                    print("path :", [path])
                    data = cv2.imread(path, cv2.IMREAD_GRAYSCALE).reshape(1280*720)
                    tstamp = datetime.timedelta(seconds = timestamp_ms // 1000,
                                                milliseconds = timestamp_ms % 1000)
                    img = dai.ImgFrame()
                    img.setData(data)
                    img.setTimestamp(tstamp)
                    img.setInstanceNum(inStreamsCameraID[i])
                    img.setType(dai.ImgFrame.Type.RAW8)
                    img.setWidth(1280)
                    img.setHeight(720)
                    q.send(img)
                    if timestamp_ms == 0:  # Send twice for first iteration
                        q.send(img)
                    print("Sent frame: {:25s}".format(path), 'timestamp_ms:', timestamp_ms)
                timestamp_ms += frame_interval_ms
                index = (index + 1) % dataset_size
                if 1: # Optional delay between iterations, host driven pipeline
                    sleep(frame_interval_ms / 1000)
            # Handle output streams

            for q in q_list:
                name  = q.getName()
                image = q.get()
                #print("Received frame:", name)
                # Skip some streams for now, to reduce CPU load
                if name in ['left', 'right', 'depth']: continue
                frame = convert_to_cv2_frame(name, image,M_right,pcl_converter)
                cv2.imshow(name, frame)

            if cv2.waitKey(1) == ord('q'):

                break


test_pipeline()
