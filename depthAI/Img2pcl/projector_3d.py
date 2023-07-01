import numpy as np
import open3d as o3d

class PointCloudVisualizer():
    def __init__(self, intrinsic_matrix, width, height):
        self.depth_map = None
        self.rgb = None
        self.pcl = None

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width,
                                                                         height,
                                                                         intrinsic_matrix[0][0],
                                                                         intrinsic_matrix[1][1],
                                                                         intrinsic_matrix[0][2],
                                                                         intrinsic_matrix[1][2])
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.isstarted = False

        self.ply_header = (
            '''ply
            format ascii 1.0
            element vertex {vertex_count}
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header
            ''')

    def rgbd_to_projection(self, depth_map, rgb, is_rgb, is_save):
        self.depth_map = depth_map
        self.rgb = rgb
        rgb_o3d = o3d.geometry.Image(self.rgb)
        depth_o3d = o3d.geometry.Image(self.depth_map)
        pcd = None
        # TODO: query frame shape to get this, and remove the param 'is_rgb'
        if is_rgb:
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d, convert_rgb_to_intensity=False)
        else:
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d)
        if self.pcl is None:
            self.pcl = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)
        else:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)
            self.pcl.points = pcd.points
            self.pcl.colors = pcd.colors
            if is_save:
                pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                o3d.io.write_point_cloud('p.pcd', pcd)
        return self.pcl,pcd

    def visualize_pcd(self):
        if not self.isstarted:
            self.vis.add_geometry(self.pcl)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            self.vis.add_geometry(origin)
            self.isstarted = True
        else:
            self.vis.update_geometry(self.pcl)
            self.vis.poll_events()
            self.vis.update_renderer()

    def close_window(self):
        self.vis.destroy_window()

    def save_pcd(self,output_file):
        """Export ``PointCloud`` to PLY file for viewing in MeshLab."""
        points = np.hstack([self.pcl.points, self.pcl.colors])
        with open(output_file, 'w') as outfile:
            outfile.write(self.ply_header.format(vertex_count=len(self.pcl.points)))
            np.savetxt(outfile, points, '%f %f %f %d %d %d')

