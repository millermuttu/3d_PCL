import cv2 as cv2
import numpy as np
import open3d as o3d

# for i in range(1,90):
#     img_left = cv2.imread(r"C:\muttu\Customer\Dr_ganesh\Arducam\calibration\left_{:02d}.ppm".format(i))
#     img_right = cv2.imread(r"C:\muttu\Customer\Dr_ganesh\Arducam\calibration\right_{:02d}.ppm".format(i))
#     # print(img_left.shape)
#     # print(img_right.shape)
#     print(i)
#     cv2.imshow("left", img_left)
#     cv2.imshow("right",img_right)
#     cv2.waitKey(2000)


if __name__ == "__main__":
    # path = r"C:\muttu\Customer\Dr_ganesh\stereo_vision\Point Cloud Generation\stereo_images\out.ply"
    # path = r"C:\muttu\Customer\Dr_ganesh\stereo_vision\Distance Measurement\out_from_images.ply"
    path = r"C:\Users\flyin\Downloads\set1\out.ply"
    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud(path)
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])



