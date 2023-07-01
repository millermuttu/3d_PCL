import cv2
import depthai as dai
import numpy as np
import os


def getFrame(queue):
    # Get frame from queue
    frame = queue.get()
    # Convert frame to OpenCV format and return
    return frame.getCvFrame()


def getMonoCamera(pipeline, isLeft):
    # Configure mono camera
    mono = pipeline.createMonoCamera()

    # Set Camera Resolution
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    if isLeft:
        # Get left camera
        mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
    else:
        # Get right camera
        mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    return mono


if __name__ == '__main__':
    pipeline = dai.Pipeline()

    os.makedirs("dataset", exist_ok=True)    

    # Set up left and right cameras
    monoLeft = getMonoCamera(pipeline, isLeft=True)
    monoRight = getMonoCamera(pipeline, isLeft=False)

    # Set output Xlink for left camera
    xoutLeft = pipeline.createXLinkOut()
    xoutLeft.setStreamName("left")

    # Set output Xlink for right camera
    xoutRight = pipeline.createXLinkOut()
    xoutRight.setStreamName("right")

    # Attach cameras to output Xlink
    monoLeft.out.link(xoutLeft.input)
    monoRight.out.link(xoutRight.input)

    with dai.Device(pipeline) as device:
        # Get output queues.
        leftQueue = device.getOutputQueue(name="left", maxSize=1)
        rightQueue = device.getOutputQueue(name="right", maxSize=1)

        # Set display window name
        cv2.namedWindow("Stereo Pair")
        # Variable used to toggle between side by side view and oneframe view.
        sideBySide = True

        while True:
            # Get left frame
            leftFrame = getFrame(leftQueue)
            # Get right frame
            rightFrame = getFrame(rightQueue)

            if sideBySide:
                # Show si....................de by side view
                imOut = np.hstack((leftFrame, rightFrame))
            else:
                # Show overlapping frames
                imOut = np.uint8(leftFrame / 2 + rightFrame / 2)

            # Display output image
            cv2.imshow("Stereo Pair", imOut)

            ## writing images
            cv2.imwrite("dataset/strero.png", imOut)
            cv2.imwrite("dataset/in_right.png", rightFrame)
            cv2.imwrite("dataset/in_left.png", leftFrame)

            # Check for keyboard input
            key = cv2.waitKey(1)
            if key == ord('q'):
                # Quit when q is pressed
                break
            elif key == ord('t'):
                # Toggle display when t is pressed
                sideBySide = not sideBySide