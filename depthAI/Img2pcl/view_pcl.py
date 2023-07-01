import os

import cv2 as cv2
import numpy as np
import open3d as o3d


"""
File to visualize the point cloud using open3d package
and also save .ocd files into .ply file so that .ply files can be viewed in meshlab software
"""


if __name__ == "__main__":
    path = r"./p.pcd"
    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud(path)
    print(pcd)
    print(np.asarray(pcd.points))
    basename = os.path.basename(path)
    filename, file_extension = os.path.splitext(basename)
    o3d.io.write_point_cloud(os.path.join(os.path.dirname(path),basename+".ply"), pcd)
    o3d.visualization.draw_geometries([pcd])




