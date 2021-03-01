import open3d as o3d
import numpy as np
import os
import pandas as pd
from laspy.file import File
import torch
from numpy import transpose
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
import random

labels = {'nochange': 0, 'added': 1, 'removed': 2, 'change': 3, 'color_change': 4}



def get_files(dir_1, dir_2, classified_dir):
    files_dir_1 = [os.path.join(dir_1, f) for f in os.listdir(dir_1) if os.path.isfile(os.path.join(dir_1, f))
                   and f.split(".")[-1] == 'las']
    files_dir_2 = [os.path.join(dir_2, f) for f in os.listdir(dir_2) if os.path.isfile(os.path.join(dir_2, f))
                   and f.split(".")[-1] == 'las']

    files_dir_1 = {int(os.path.basename(x).split("_")[0]): x for x in files_dir_1}
    files_dir_2 = {int(os.path.basename(x).split("_")[0]): x for x in files_dir_2}

    classified_point_list_files = [os.path.join(classified_dir, f) for f in os.listdir(classified_dir)
                                   if os.path.isfile(os.path.join(classified_dir, f))]
    scene_numbers = [int(os.path.basename(x).split('_')[0]) for x in classified_point_list_files]
    classified_point_list_dfs = {scene_num: pd.read_csv(path)
                                 for scene_num, path in zip(scene_numbers, classified_point_list_files)}
    return scene_numbers, files_dir_1, files_dir_2, classified_point_list_dfs


def extract_area(full_cloud, center, color, clearance, shape='cylinder'):
    mask = []
    if shape == 'square':
        x_mask = ((center[0] + clearance) > full_cloud[:, 0]) & (full_cloud[:, 0] > (center[0] - clearance))
        y_mask = ((center[1] + clearance) > full_cloud[:, 1]) & (full_cloud[:, 1] > (center[1] - clearance))
        mask = x_mask & y_mask
    elif shape == 'cylinder':
        mask = np.linalg.norm(full_cloud[:, :2] - center, axis=1) < clearance
    return full_cloud[mask], color[mask]


def load_las(path):
    input_las = File(path, mode='r')
    point_records = input_las.points.copy()
    las_scale_x = input_las.header.scale[0]
    las_offset_x = input_las.header.offset[0]
    las_scale_y = input_las.header.scale[1]
    las_offset_y = input_las.header.offset[1]
    las_scale_z = input_las.header.scale[2]
    las_offset_z = input_las.header.offset[2]

    # calculating coordinates
    p_x = np.array((point_records['point']['X'] * las_scale_x) + las_offset_x)
    p_y = np.array((point_records['point']['Y'] * las_scale_y) + las_offset_y)
    p_z = np.array((point_records['point']['Z'] * las_scale_z) + las_offset_z)

    points = np.vstack((p_x, p_y, p_z, input_las.red, input_las.green, input_las.blue)).T

    return points


def remove_below(cloud, color, center):
    mask = (center[2] <= cloud[:, 2])
    return cloud[mask], color[mask]


def traslation(cloud, center):
    cloud = cloud - center
    return cloud


def features(pcd, voxel_size=0.05):
    radius_normal = voxel_size * 2
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_feature, max_nn=100))
    return pcd_fpfh


class KNNGraph(object):
    def __init__(self,
                 k=6,
                 loop=False,
                 force_undirected=False,
                 flow='source_to_target'):
        self.k = k
        self.loop = loop
        self.force_undirected = force_undirected
        self.flow = flow

    def __call__(self, data):
        data.edge_attr = None
        batch = data.batch if 'batch' in data else None
        edge_index = knn_graph(data.pos, self.k, batch, loop=self.loop, flow=self.flow)

        if self.force_undirected:
            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

        data.edge_index = edge_index

        return data

    def __repr__(self):
        return '{}(k={})'.format(self.__class__.__name__, self.k)



def statistical_outlier_removal(pcd,std):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10,std_ratio =std)
    return cl

def generate_dataset(dir_1, dir_2, classified_dir):
    dataset16 = []
    dataset20 = []
    dataset_col = []
    scene_numbers, files_dir_1, files_dir_2, classified_point_list_dfs = get_files(dir_1, dir_2, classified_dir)
    for scene in np.sort(scene_numbers):
        file1 = files_dir_1[scene]
        file2 = files_dir_2[scene]
        points_rgb_2016 = load_las(file1)
        points_rgb_2020 = load_las(file2)

        object_2016 = points_rgb_2016[:, :3]
        object_2016_rgb = points_rgb_2016[:, 3:]
        object_2016_rgb = np.rint(np.divide(object_2016_rgb, object_2016_rgb.max(axis=0)) * 255).astype(np.uint8)

        object_2020 = points_rgb_2020[:, :3]
        object_2020_rgb = points_rgb_2020[:, 3:]
        object_2020_rgb = np.rint(np.divide(object_2020_rgb, object_2020_rgb.max(axis=0)) * 255).astype(np.uint8)

        center = []
        y = []
        # labels and point of interest
        for index, row in classified_point_list_dfs[scene].iterrows():
            center.append(np.array([row['x'], row['y'], row['z']]))
            y.append(row['classification'])
        for point in range(len(center)):
            flag16 = 0
            flag20 = 0
            col_flag = 0
            pc2016, color_16 = extract_area(np.asarray(object_2016), center[point][:2], object_2016_rgb, 1)
            pc2020, color_20 = extract_area(np.asarray(object_2020), center[point][:2], object_2020_rgb, 1)

            pc2016, color_16 = remove_below(pc2016, color_16, center[point])
            pc2020, color_20 = remove_below(pc2020, color_20, center[point])

            pc2016 = traslation(pc2016, center[point])
            pc2020 = traslation(pc2020, center[point])

            if len(pc2016) == 0:
                pc2016 = [center[point]]
                color_16 = [[0,0,0]]
                col_flag = 1
            if len(pc2020) == 0:
                pc2020 = [center[point]]
                color_20 = [[0,0,0]]
                col_flag = 1
            pc16 = o3d.geometry.PointCloud()
            pc16.points = o3d.utility.Vector3dVector(np.asarray(pc2016))
            pc16.paint_uniform_color([1, 1, 1])
            for i in range(len(np.asarray(pc16.points))):
                pc16.colors[i] = [color_16[i][0] / 255, color_16[i][1] / 255, color_16[i][2] / 255]
            # o3d.visualization.draw_geometries([pc16])

            pc20 = o3d.geometry.PointCloud()
            pc20.points = o3d.utility.Vector3dVector(np.asarray(pc2020))
            pc20.paint_uniform_color([1, 1, 1])
            for i in range(len(np.asarray(pc20.points))):
                pc20.colors[i] = [color_20[i][0] / 255, color_20[i][1] / 255, color_20[i][2] / 255]
            # o3d.visualization.draw_geometries([pc20])

            pc16_g = statistical_outlier_removal(pc16,2.5)
            pc20_g = statistical_outlier_removal(pc20,2.5)

            # geomertrical part
            dist16 = pc16.compute_point_cloud_distance(pc16_g)
            dist20 = pc20.compute_point_cloud_distance(pc20_g)

            # creating dataset for 2020 stream
            change_points_20 = []
            dist_20= []
            for i in range(len(dist20)):
                if dist20[i] > 0.2:
                    change_points_20.append(pc2020[i])
                    dist_20.append(dist20[i])
            if len(change_points_20) > 0:
                if len(change_points_20) >= 50000:
                    random.shuffle(change_points_20)
                    f20 = o3d.geometry.PointCloud()
                    f20.points = o3d.utility.Vector3dVector(np.asarray(change_points_20[:50000]))
                else:
                    f20 = o3d.geometry.PointCloud()
                    f20.points = o3d.utility.Vector3dVector(np.asarray(change_points_20))
                # o3d.visualization.draw_geometries([f20])
            else:
                change_points_20.append(center[point])
                f20 = o3d.geometry.PointCloud()
                f20.points = o3d.utility.Vector3dVector(np.asarray(change_points_20))
                flag16 = 1

            change_points_16 = []
            dist_16 = []
            for i in range(len(dist16)):
                if dist16[i] > 0.2:
                    change_points_16.append(pc2016[i])
                    dist_16.append(dist16[i])
            if len(change_points_16) > 0:
                if len(change_points_16)>=50000:
                    random.shuffle(change_points_16)
                    f16 = o3d.geometry.PointCloud()
                    f16.points = o3d.utility.Vector3dVector(np.asarray(change_points_16[:50000]))
                else:
                    f16 = o3d.geometry.PointCloud()
                    f16.points = o3d.utility.Vector3dVector(np.asarray(change_points_16))
            else:
                change_points_16.append(center[point])
                f16 = o3d.geometry.PointCloud()
                f16.points = o3d.utility.Vector3dVector(np.asarray(np.asarray(change_points_16)))
                flag20 =1


            fpfh16 = transpose(features(f16).data)
            fpfh16 = torch.tensor(fpfh16, dtype=torch.float32)
            node_vals = torch.tensor(np.asarray(f16.points), dtype=torch.float32)
            label = labels[y[point]]
            label = torch.tensor(label)
            data = Data(y=label, pos=node_vals, features=fpfh16, scene=scene, point=point, flag=flag16,
                        truelab=y[point])
            knn = KNNGraph(10, loop=True)
            knn.__call__(data)
            dataset16.append(data)


            fpfh20 = transpose(features(f20).data)
            fpfh20 = torch.tensor(fpfh20, dtype=torch.float32)
            node_vals_1 = torch.tensor(np.asarray(f20.points), dtype=torch.float32)
            label = labels[y[point]]
            label = torch.tensor(label)
            data_1 = Data(y=label, pos=node_vals_1, features=fpfh20, scene=scene, point=point, flag=flag20,
                          truelab=y[point])
            knn = KNNGraph(10, loop=True)
            knn.__call__(data_1)
            dataset20.append(data_1)

            if col_flag == 0 :

                # color part
                col16 = o3d.geometry.PointCloud()
                col16.points = o3d.utility.Vector3dVector(np.asarray(color_16))
                #o3d.visualization.draw_geometries([col16])

                col20 = o3d.geometry.PointCloud()
                col20.points = o3d.utility.Vector3dVector(np.asarray(color_20))
                #o3d.visualization.draw_geometries([col20])

                dist16 = col16.compute_point_cloud_distance(col20)
                dist20 = col20.compute_point_cloud_distance(col16)

                f16_tree = o3d.geometry.KDTreeFlann(pc16)
                f20_tree = o3d.geometry.KDTreeFlann(pc20)

                cc = []
                col_dist = []
                col_dist2 = []
                val1 = (max(dist20) + np.mean(dist20)) / 3
                val2 = (max(dist20) + np.mean(dist20)) / 3
                for i in range(len(dist20)):
                    if dist20[i] > val1:
                        cc.append(np.asarray(pc20.points)[i])
                        col_dist.append(color_20[i])
                        [k, idx, _] = f16_tree.search_knn_vector_3d(pc20.points[i], 1)
                        col_dist2.append(color_16[idx[0]])
                for i in range(len(dist16)):
                    if dist16[i] > val2:
                        cc.append(np.asarray(pc16.points)[i])
                        col_dist.append(color_16[i])
                        [k, idx, _] = f20_tree.search_knn_vector_3d(pc16.points[i], 1)
                        col_dist2.append(color_20[idx[0]])

                # o3d.visualization.draw_geometries([pcf])
                if len(cc) > 0:
                    pcf = o3d.geometry.PointCloud()
                    pcf.points = o3d.utility.Vector3dVector(np.asarray(cc))
                else:
                    col_flag = 1

            if col_flag == 1:
                cc = []
                cc.append(center[point])
                pcf = o3d.geometry.PointCloud()
                pcf.points = o3d.utility.Vector3dVector(np.asarray(cc))
                col_dist = []
                col_dist2 = []
                col_dist.append([0,0,0])
                col_dist2.append([0, 0, 0])

            feats = transpose(features(pcf).data)
            feats = torch.tensor(feats)
            color_point = torch.tensor(np.hstack((feats,col_dist,col_dist2)), dtype=torch.float32)
            node_vals = torch.tensor(np.asarray(pcf.points), dtype=torch.float32)
            # label = torch.tensor([labdic[y[point]]] * len(np.asarray(f16.points)),dtype=torch.float32)
            label = labels[y[point]]
            label = torch.tensor(label)
            data = Data(y=label, pos=node_vals, features =color_point, scene=scene, point=point, flag=col_flag,
                        truelab=y[point])
            knn = KNNGraph(10, loop=True)
            knn.__call__(data)
            dataset_col.append(data)
    return dataset16, dataset20, dataset_col


