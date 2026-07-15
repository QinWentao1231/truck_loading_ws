import open3d as o3d
import numpy as np
import math
import time
import copy


# 0630 车厢内 9 份点云的地面法向 Ry 中位数，作为固定零点。
# method=4 时返回相对该零点的 Ry；method=1/2/3 始终返回 Ry=0。
RY_INTERIOR_BASELINE_DEG = 0.03
RY_OUTPUT_DEADBAND_DEG = 0.2


def _wrap_angle_deg(angle):
    """将角度归一化到 [-180, 180)。"""
    return (float(angle) + 180.0) % 360.0 - 180.0


def relative_ry_from_0630_baseline(raw_ry_deg):
    """返回相对 0630 车厢内固定零点的 Ry 变化。"""
    delta = _wrap_angle_deg(raw_ry_deg - RY_INTERIOR_BASELINE_DEG)
    return 0.0 if abs(delta) < RY_OUTPUT_DEADBAND_DEG else delta


def ry_from_ground_normal(ground_model, lidar_to_base):
    """仅由地面法向计算机器人基坐标系下的绝对 Ry（度）。

    ground_model 在角点算法中被统一为向下法向，因此取反向得到“上”向量；
    只用外参旋转部分转到机器人基坐标，避免垛面倾斜和箱体边缘污染 Ry。
    """
    normal = np.asarray(ground_model[:3], dtype=float)
    norm = np.linalg.norm(normal)
    if norm == 0:
        raise ValueError("地面法向为零，无法计算 Ry")
    up_lidar = -normal / norm
    up_base = np.asarray(lidar_to_base[:3, :3], dtype=float) @ up_lidar
    up_base /= np.linalg.norm(up_base)
    return math.degrees(math.atan2(up_base[0], up_base[2]))


def fixed_axis_plane(model, axis, sign):
    """保留平面在主轴上的交点，将法向固定为指定坐标轴方向。"""
    coefficient = float(model[axis])
    if abs(coefficient) < 1e-6:
        raise ValueError(f"平面在轴 {axis} 上的法向分量过小，无法固定法向")
    coordinate = -float(model[3]) / coefficient
    normal = [0.0, 0.0, 0.0]
    normal[axis] = float(sign)
    return [normal[0], normal[1], normal[2], -float(sign) * coordinate]


def process_point_cloud(pcd, method):
    # #保存pcd点云到路径
    # target_file_path = "/home/simplenav/Test_ws/corn_poits.pcd"
    # success = o3d.io.write_point_cloud(target_file_path, pcd, write_ascii=True)
    # if success:
    #     print(f"传入点云已成功保存到 {target_file_path}")
    # else:
    #     print("点云保存失败")
    # file_path = "/home/fanuc/data_nav/carriage/trun_cloud_20250714_130945.pcd"
    # pcd = o3d.io.read_point_cloud(file_path)
    # method = 3
    ori_points = np.asarray(pcd.points)
    theta = np.pi / 2
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    rotated_points = np.dot(ori_points, R_z.T)
    pcd.points = o3d.utility.Vector3dVector(rotated_points)
    # 打印点云数量
    num_points = len(np.asarray(pcd.points))
    print(f"******输入点云数量为 : {num_points}")


    # target_file_path_ply = "/home/simplenav/Test_ws/corn.ply"
    # #保存ply点云到目标路径
    # success = o3d.io.write_point_cloud(target_file_path_ply, pcd)
    # if success:
    #     print(f"传入点云已成功保存到 {target_file_path_ply}")
    # else:
    #     print("点云保存失败")


    def segment_plane(pcd, distance_threshold=0.005, ransac_n=3, num_iterations=10000):
        """
        分割点云中的平面并在原始点云上以不同颜色标记显示。
        Args:
            pcd: Open3D 点云对象
            distance_threshold: RANSAC 的距离阈值
            ransac_n: RANSAC 拟合平面所需的最小点数
            num_iterations: RANSAC 的迭代次数
        Returns:
            planes: 分割出的平面点云列表
            remaining_cloud: 剩余点云
        """
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        return plane_model, inlier_cloud, outlier_cloud


    def show_plane(model, color):
        """
        平面拟合可视化。
        Args:
            model: 平面方程
            color: 绘制颜色
        Returns:
            plane_mesh: Open3D Box对象
        """
        normal = np.array([model[0], model[1], model[2]])
        # 计算平面法向量的旋转
        initial_normal = np.array([0, 0, 1])  # 初始法向量是 Z 轴方向
        axis = np.cross(initial_normal, normal)  # 旋转轴是初始法向量与目标法向量的叉积
        axis = axis / np.linalg.norm(axis)  # 归一化旋转轴
        cos_angle = np.dot(initial_normal, normal)  # 计算夹角的余弦值
        angle = np.arccos(cos_angle)  # 计算夹角
        # 创建一个平面网格
        plane_mesh = o3d.geometry.TriangleMesh.create_box(width=6, height=4, depth=0.001)
        # 计算旋转矩阵
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        # 旋转平面网格
        plane_mesh.rotate(R, center=(0, 0, 0))
        # 根据平面方程中的 d 来计算平面的位置
        plane_mesh.translate(-plane_mesh.get_center())
        if max(range(len(normal)), key=lambda i: abs(normal[i])) == 0:
            plane_mesh.translate(np.array([-model[3] / model[0], 0, 0]))
        if max(range(len(normal)), key=lambda i: abs(normal[i])) == 1:
            plane_mesh.translate(np.array([0, -model[3] / model[1], 0]))
        if max(range(len(normal)), key=lambda i: abs(normal[i])) == 2:
            plane_mesh.translate(np.array([0, 0, -model[3] / model[2]]))
        plane_mesh.paint_uniform_color(color)

        return plane_mesh


    def intersection_of_planes(plane1, plane2, plane3):
        """
        拟合三个平面的交点。
        Args:
            plane1, plane2, plane3: 平面方程
        Returns:
            intersection_point: 交点
        """
        # 解线性方程组 Ax = b，求解三个平面的交点
        A = np.array([
            [plane1[0], plane1[1], plane1[2]],
            [plane2[0], plane2[1], plane2[2]],
            [plane3[0], plane3[1], plane3[2]]
        ])
        b = np.array([-plane1[3], -plane2[3], -plane3[3]])
        # 使用np.linalg.solve求解
        intersection_point = np.linalg.solve(A, b)
        return intersection_point


    def matrix2euler(r):
        """
        旋转矩阵转欧拉角xyzwpr。
        Args:
            r: 4*4旋转矩阵
        Returns:
            xyzwpr
        """
        # 确保传入的是3x3旋转矩阵
        assert r.shape == (4, 4)
        # 计算欧拉角 (ZYX 顺序)
        yaw = np.arctan2(r[1, 0], r[0, 0])  # z轴旋转
        pitch = np.arctan2(-r[2, 0], np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2))
        roll = np.arctan2(r[2, 1], r[2, 2])  # x轴旋转
        return [r[0, 3], r[1, 3], r[2, 3], roll * 180 / math.pi, pitch * 180 / math.pi, yaw * 180 / math.pi]


    def point_to_plane_distance(point, a, b, c, d):
        # 计算点到平面的符号
        return a * point[0] + b * point[1] + c * point[2] + d


    def fiterCloud(pcd):
        # vox_pcd = pcd
        # vox_pcd = pcd.voxel_down_sample(voxel_size=0.005)
        down_pcd = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)[0]
        return down_pcd


    def clustFrontBoard(pcd):
        labels = np.array(pcd.cluster_dbscan(eps=0.03, min_points=10))
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_count_dict = dict(zip(unique_labels, counts))
        min_cluster_size = 1000
        valid_labels = [label for label in unique_labels
                        if label != -1 and label_count_dict[label] >= min_cluster_size]
        cluster_planes = []
        d_list = []
        if view:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Front clustered", width=800, height=600)
            for label in valid_labels:
                indices = np.where(labels == label)[0]
                cluster_points = np.asarray(pcd.points)[indices]
                cluster_pcd = o3d.geometry.PointCloud()
                cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
                cluster_pcd.paint_uniform_color(np.random.rand(3))
                vis.add_geometry(cluster_pcd)
            vis.run()
            vis.destroy_window()
        for label in valid_labels:
            indices = np.where(labels == label)[0]
            cluster_points = np.asarray(pcd.points)[indices]
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
            cluster_pcd.paint_uniform_color(np.random.rand(3))
            plane_model, plane_inliers, _ = segment_plane(cluster_pcd)
            cluster_planes.append(plane_inliers)
            d_list.append(abs(plane_model[3]))
        d_array = np.array(d_list)
        d_threshold = d_array.min() + (d_array.max() - d_array.min()) * (1 / 4)
        selected_clusters = [
            cluster_planes[i] for i, d in enumerate(d_list) if d <= d_threshold
        ]
        other_clusters = [
            cluster_planes[i] for i, d in enumerate(d_list) if d >= d_threshold
        ]
        if not selected_clusters:
            raise ValueError("FrontBoard未找到满足 d 阈值条件的聚类")
        merged_points = selected_clusters[0]
        for cloud in selected_clusters[1:]:
            merged_points += cloud
        # merged_points.paint_uniform_color([1, 1, 0])
        other_points = other_clusters[0]
        for cloud in other_clusters[1:]:
            other_points += cloud
        if view:
            pcd_show = copy.deepcopy(pcd)
            pcd_tree = o3d.geometry.KDTreeFlann(pcd_show)
            pcd_show.paint_uniform_color([0.7, 0.7, 0.7])
            colors = np.asarray(pcd_show.colors)
            visited = set()
            for point in merged_points.points:
                _, idx, _ = pcd_tree.search_knn_vector_3d(point, 1)
                if idx[0] not in visited:
                    colors[idx[0]] = [1.0, 0.0, 0.0]
                    visited.add(idx[0])
            for point in other_points.points:
                _, idx, _ = pcd_tree.search_knn_vector_3d(point, 1)
                if idx[0] not in visited:
                    colors[idx[0]] = [0.0, 1.0, 0.0]
                    visited.add(idx[0])
            pcd_show.colors = o3d.utility.Vector3dVector(colors)
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="front d filtered", width=800, height=600, left=500, top=200)
            vis.add_geometry(pcd_show)
            vis.run()
            vis.destroy_window()
        final_plane_model, _, _ = segment_plane(merged_points)
        aabb = merged_points.get_axis_aligned_bounding_box()
        bounding_box_final = aabb.get_extent()
        other_plane_model, _, _ = segment_plane(other_points)
        aabb = other_points.get_axis_aligned_bounding_box()
        bounding_box_other = aabb.get_extent()
        # final_plane_model[3] = -d_array.min()
        if len(merged_points.points) <= len(other_points.points):
            return [[final_plane_model[0], final_plane_model[1], final_plane_model[2], final_plane_model[3],
                    bounding_box_final[1]],
                    [other_plane_model[0], other_plane_model[1], other_plane_model[2], other_plane_model[3],
                    bounding_box_other[1]]]
        else:
            return [[other_plane_model[0], other_plane_model[1], other_plane_model[2], other_plane_model[3],
                    bounding_box_other[1]],
                    [final_plane_model[0], final_plane_model[1], final_plane_model[2], final_plane_model[3],
                    bounding_box_final[1]]]
    
    start_time = time.time()
    global view
    view = False
    view_normal = False
    corner_list = []
    # 1: 车头波纹板；2: I垛面；3: L垛面；
    # 4: I垛面角点，并根据地面法向返回相对车厢内基准的 Ry。
    print(f'method: {method}')
    down_pcd = pcd
    if view:
        down_pcd = fiterCloud(down_pcd)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Down", width=800, height=600, left=500, top=200)
        vis.add_geometry(down_pcd)
        vis.run()
        vis.destroy_window()
        down_pcd.paint_uniform_color([0.2, 0.2, 0.2])

    # 垛面点云直通滤波
    points = np.asarray(down_pcd.points)
    x_filtered = (points[:, 0] <= 1.9) & (points[:, 0] >= 0.5)
    y_filtered = (points[:, 1] <= 2.0) & (points[:, 1] >= -2.0)
    z_filtered = (points[:, 2] <= 1.0) & (points[:, 2] >= -1.0)
    filtered_points = points[x_filtered & y_filtered & z_filtered]
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd = fiterCloud(filtered_pcd)
    filtered_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))
    normals = np.asarray(filtered_pcd.normals)
    points = np.asarray(filtered_pcd.points)

    if method in (1, 2, 4):
        flip_mask = normals[:, 0] < 0
        normals[flip_mask] *= -1
        filtered_pcd.normals = o3d.utility.Vector3dVector(normals)
        if view_normal:
            o3d.visualization.draw_geometries([filtered_pcd], point_show_normal=True)
        normals = np.asarray(filtered_pcd.normals)
        target_normal = np.array([1, 0, 0])
        cos_theta = np.abs(np.dot(normals, target_normal))
        angle_threshold = np.cos(np.radians(10))
        front_indices = np.where(cos_theta > angle_threshold)[0]
        front_points = np.asarray(filtered_pcd.points)[front_indices]
        front_pcd = o3d.geometry.PointCloud()
        front_pcd.points = o3d.utility.Vector3dVector(front_points)
        front_pcd = fiterCloud(front_pcd)
        if view:
            front_pcd.paint_uniform_color([0, 1, 1])
            pcd_show = copy.deepcopy(filtered_pcd)
            pcd_tree = o3d.geometry.KDTreeFlann(pcd_show)
            pcd_show.paint_uniform_color([0.7, 0.7, 0.7])
            colors = np.asarray(pcd_show.colors)
            visited = set()
            for point in front_pcd.points:
                _, idx, _ = pcd_tree.search_knn_vector_3d(point, 1)
                if idx[0] not in visited:
                    colors[idx[0]] = [1.0, 0.0, 0.0]
                    visited.add(idx[0])
            pcd_show.colors = o3d.utility.Vector3dVector(colors)
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="front filtered", width=800, height=600, left=500, top=200)
            vis.add_geometry(pcd_show)
            vis.run()
            vis.destroy_window()
        [a, b, c, d], inlier_cloud, outlier_cloud = segment_plane(front_pcd)
        if a < 0:
            a, b, c, d = -a, -b, -c, -d
        temp_points = np.asarray(outlier_cloud.points)
        if len(temp_points) != 0:
            n_main = np.array([a, b, c])
            n_main = n_main / np.linalg.norm(n_main)
            distances = temp_points @ n_main + d
            if method == 1:
                delta = min(np.percentile(distances, 5), 0)
            else:
                delta = min(np.percentile(distances, 50), 0)
            delta_min = min(np.percentile(distances, 0), 0)
            i_model = [a, b, c, d - delta]
            i_model_ = [a, b, c, d - delta_min]
        else:
            i_model = [a, b, c, d]
            i_model_ = [a, b, c, d]
        if view:
            front_plane1 = show_plane([a, b, c, d], [1, 0, 0])
            front_plane2 = show_plane(i_model, [0, 1, 0])
            front_plane3 = show_plane(i_model_, [0, 0, 1])
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Front result", width=800, height=600, left=500, top=200)
            vis.add_geometry(filtered_pcd)
            vis.add_geometry(front_plane1)
            vis.add_geometry(front_plane2)
            vis.add_geometry(front_plane3)
            vis.run()
            vis.destroy_window()
    elif method == 3:
        flip_mask = normals[:, 0] < 0
        normals[flip_mask] *= -1
        filtered_pcd.normals = o3d.utility.Vector3dVector(normals)
        if view_normal:
            o3d.visualization.draw_geometries([filtered_pcd], point_show_normal=True)
        normals = np.asarray(filtered_pcd.normals)
        target_normal = np.array([1, 0, 0])
        cos_theta = np.abs(np.dot(normals, target_normal))
        angle_threshold = np.cos(np.radians(10))
        front_indices = np.where(cos_theta > angle_threshold)[0]
        front_points = np.asarray(filtered_pcd.points)[front_indices]
        front_pcd = o3d.geometry.PointCloud()
        front_pcd.points = o3d.utility.Vector3dVector(front_points)
        front_pcd = fiterCloud(front_pcd)
        # 滤波结果可视化
        if view:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Front filtered", width=800, height=600)
            vis.add_geometry(front_pcd)
            vis.run()
            vis.destroy_window()
        l_model, i_model = clustFrontBoard(front_pcd)
        if l_model[0] < 0:
            l_model[0] = -l_model[0]
            l_model[1] = -l_model[1]
            l_model[2] = -l_model[2]
            l_model[3] = -l_model[3]
        if i_model[0] < 0:
            i_model[0] = -i_model[0]
            i_model[1] = -i_model[1]
            i_model[2] = -i_model[2]
            i_model[3] = -i_model[3]
        front_planeR = show_plane([l_model[0], l_model[1], l_model[2], l_model[3]], [1, 0, 0])
        front_planeL = show_plane([i_model[0], i_model[1], i_model[2], i_model[3]], [0, 1, 0])
        if view:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Front result", width=800, height=600, left=500, top=200)
            vis.add_geometry(filtered_pcd)
            vis.add_geometry(front_planeR)
            vis.add_geometry(front_planeL)
            vis.run()
            vis.destroy_window()
    else:
        raise Exception(f'Wrong method value: {method}; expected 1, 2, 3, or 4')

    # 左侧面点云滤波
    flip_mask = (normals[:, 1] * points[:, 1]) < 0
    # 批量翻转
    normals[flip_mask] *= -1
    filtered_pcd.normals = o3d.utility.Vector3dVector(normals)
    if view_normal:
        o3d.visualization.draw_geometries([filtered_pcd], point_show_normal=True)
    normals = np.asarray(filtered_pcd.normals)
    target_normal = np.array([0, 1, 0])
    cos_theta = np.dot(normals, target_normal)
    angle_threshold = np.cos(np.radians(45))
    if method == 1:
        x_threshold = -i_model[3] - 0.1
    else:
        x_threshold = -i_model[3] - 0.05
    z_threshold = -0.7
    x_threshold_ = 1.0
    mask = (cos_theta > angle_threshold) & (points[:, 0] < x_threshold) & (points[:, 0] > x_threshold_) & (points[:, 2] > z_threshold)
    left_points = np.asarray(filtered_pcd.points)[mask]
    left_pcd = o3d.geometry.PointCloud()
    left_pcd.points = o3d.utility.Vector3dVector(left_points)
    left_pcd = fiterCloud(left_pcd)
    left_points = np.asarray(left_pcd.points)
    # 可视化滤波后的点云
    if view:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="left_pcd", width=800, height=600, left=500, top=200)
        vis.add_geometry(left_pcd)
        vis.run()
        vis.destroy_window()
    [a, b, c, d], inlier_cloud, outlier_cloud = segment_plane(left_pcd)
    if b < 0:
        a, b, c, d = -a, -b, -c, -d
    temp_points = np.asarray(outlier_cloud.points)
    if len(temp_points) != 0:
        # n_main = np.array([a, b, c])
        n_main = np.array([0, 1, 0])
        n_main = n_main / np.linalg.norm(n_main)
        # 计算所有点到主平面的有符号距离
        # distances = temp_points @ n_main + d
        distances = left_points @ n_main
        # distances = distances[distances < 0]
        distances = distances[distances > 0]
        # delta = max(min(np.percentile(distances, 40), 0), -0.05)
        delta = np.percentile(distances, 5) # 调整左侧
        # delta_min = min(np.percentile(distances, 0), 0)
        delta_min = np.percentile(distances, 0)
        # side_left_model = [a, b, c, d - delta]
        side_left_model = [0, 1, 0, - delta]
        # side_left_model = [a, b, c, d]
        # side_left_model_ = [a, b, c, d - delta_min]
        side_left_model_ = [0, 1, 0, - delta_min]
    else:
        side_left_model = [a, b, c, d]
        side_left_model_ = [a, b, c, d]
    if view:
        side_l_plane1 = show_plane([a, b, c, d], [1, 0, 0])
        side_l_plane2 = show_plane(side_left_model, [0, 1, 0])
        side_l_plane3 = show_plane(side_left_model_, [0, 0, 1])
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="side_left", width=800, height=600, left=500, top=200)
        vis.add_geometry(filtered_pcd)
        vis.add_geometry(side_l_plane1)
        vis.add_geometry(side_l_plane2)
        vis.add_geometry(side_l_plane3)
        vis.run()
        vis.destroy_window()

    # 右侧面点云滤波
    if view_normal:
        o3d.visualization.draw_geometries([filtered_pcd], point_show_normal=True)
    target_normal = np.array([0, -1, 0])
    cos_theta = np.dot(normals, target_normal)
    angle_threshold = np.cos(np.radians(45))
    # right_indices = np.where(cos_theta > angle_threshold)[0]
    x_threshold = -i_model[3]-0.05 if method != 3 else -l_model[3]
    z_threshold = -0.7
    mask = (cos_theta > angle_threshold) & (points[:, 0] < x_threshold) & (points[:, 2] > z_threshold)
    right_points = np.asarray(filtered_pcd.points)[mask]
    right_pcd = o3d.geometry.PointCloud()
    right_pcd.points = o3d.utility.Vector3dVector(right_points)
    right_pcd = fiterCloud(right_pcd)
    right_points = np.asarray(right_pcd.points)
    # 可视化滤波后的点云
    if view:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="side_right", width=800, height=600, left=500, top=200)
        vis.add_geometry(right_pcd)
        vis.run()
        vis.destroy_window()
    [a, b, c, d], inlier_cloud, outlier_cloud = segment_plane(right_pcd)
    if b > 0:
        a, b, c, d = -a, -b, -c, -d
    temp_points = np.asarray(outlier_cloud.points)
    if len(temp_points) != 0:
        # n_main = np.array([a, b, c])
        n_main = np.array([0, -1, 0])
        n_main = n_main / np.linalg.norm(n_main)
        # 计算所有点到主平面的有符号距离
        # distances = temp_points @ n_main + d
        distances = right_points @ n_main
        # delta = min(np.percentile(distances, 5), 0)
        delta = np.percentile(distances, 10)
        # delta_min = min(np.percentile(distances, 0), 0)
        delta_min = np.percentile(distances, 0)
        # side_right_model = [a, b, c, d - delta]
        side_right_model = [0, -1, 0, -delta]
        # side_right_model = [a, b, c, d]
        # side_right_model_ = [a, b, c, d - delta_min]
        side_right_model_ = [0, -1 ,0, -delta_min]
    else:
        side_right_model = [a, b, c, d]
        side_right_model_ = [a, b, c, d]
    if view:
        side_r_plane1 = show_plane([a, b, c, d], [1, 0, 0])
        side_r_plane2 = show_plane(side_right_model, [0, 1, 0])
        side_r_plane3 = show_plane(side_right_model_, [0, 0, 1])
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="side_right", width=800, height=600, left=500, top=200)
        vis.add_geometry(filtered_pcd)
        vis.add_geometry(side_r_plane1)
        vis.add_geometry(side_r_plane2)
        vis.add_geometry(side_r_plane3)
        vis.run()
        vis.destroy_window()

    #  底面点云滤波
    flip_mask = normals[:, 2] > 0
    normals[flip_mask] *= -1
    filtered_pcd.normals = o3d.utility.Vector3dVector(normals)
    if view_normal:
        o3d.visualization.draw_geometries([filtered_pcd], point_show_normal=True)
    normals = np.asarray(filtered_pcd.normals)
    target_normal = np.array([0, 0, -1])
    cos_theta = np.dot(normals, target_normal)
    angle_threshold = np.cos(np.radians(10))
    ground_indices = np.where(cos_theta > angle_threshold)[0]
    ground_points = np.asarray(filtered_pcd.points)[ground_indices]
    # 地面 z 会随雷达安装/外参变化，不使用固定 z 阈值。
    # 从较低 40% 的水平候选点中找最密集的 10mm 高度峰，
    # 再取该高度 ±60mm 带宽拟合局部地面。
    # 不复用前/左/右平面筛选过程中被多次翻转过的 normals；直接从空间 ROI
    # 的低位高度峰提取地面，再由独立 RANSAC 确定法向。
    ground_candidates = points[points[:, 0] > 1.0]
    if len(ground_candidates) < 30:
        raise ValueError(f"地面水平候选点不足: {len(ground_candidates)}")
    z_values = ground_candidates[:, 2]
    z_upper = float(np.percentile(z_values, 40))
    z_low = z_values[z_values <= z_upper]
    z_min, z_max = float(z_low.min()), float(z_low.max())
    if z_max - z_min < 0.01:
        ground_z = float(np.median(z_low))
    else:
        bin_count = max(1, int(np.ceil((z_max - z_min) / 0.01)))
        hist, edges = np.histogram(z_low, bins=bin_count)
        peak = int(hist.argmax())
        ground_z = float((edges[peak] + edges[peak + 1]) * 0.5)
    ground_points = ground_candidates[
        np.abs(ground_candidates[:, 2] - ground_z) <= 0.06]
    if len(ground_points) < 30:
        raise ValueError(f"动态地面高度带内点数不足: {len(ground_points)}")
    ground_pcd = o3d.geometry.PointCloud()
    ground_pcd.points = o3d.utility.Vector3dVector(ground_points)
    # ground_pcd = fiterCloud(ground_pcd)
    # 可视化滤波后的点云
    if view:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="ground_pcd", width=800, height=600, left=500, top=200)
        vis.add_geometry(ground_pcd)
        vis.run()
        vis.destroy_window()
    o3d.utility.random.seed(0)
    [a, b, c, d], ground_inliers, ground_outliers = segment_plane(ground_pcd)
    if c > 0:
        a, b, c, d = -a, -b, -c, -d
    # # 获取底面点云
    temp_points = np.asarray(ground_outliers.points)
    if len(temp_points) != 0:
        n_main = np.array([a, b, c])
        n_main = n_main / np.linalg.norm(n_main)
        # 计算所有点到主平面的有符号距离
        distances = temp_points @ n_main + d
        delta = min(np.percentile(distances, 10), 0)
        delta_min = min(np.percentile(distances, 0), 0)
        ground_model = [a, b, c, d - delta]
        ground_model_ = [a, b, c, d - delta_min]
    else:
        ground_model = [a, b, c, d]
        ground_model_ = [a, b, c, d]
    # 校验底面法线
    v1 = np.array([i_model[0], i_model[1], i_model[2]])
    v2 = np.array([side_left_model[0], side_left_model[1], side_left_model[2]])
    v3 = np.array([ground_model[0], ground_model[1], ground_model[2]])
    v12 = np.cross(v1, v2)
    if v12[2] > 0:
        v12 = -v12
    dot_product = np.dot(v12, v3)
    magnitude_v1 = np.linalg.norm(v12)
    magnitude_v2 = np.linalg.norm(v3)
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    angle_rad = np.arccos(np.clip(cos_theta, -1, 1))
    angle_with_plane_rad = np.pi / 2 - angle_rad
    angle_with_plane_deg = np.degrees(angle_with_plane_rad)
    # if angle_with_plane_deg < 85 or np.max(temp_points[:, 2]) + ground_model[3] > 0.08:
    #     raise Exception(f"地面法线夹角: {angle_with_plane_deg:.2f}, "
    #                     f"地面凸起： {np.max(filtered_points[:, 2]) + ground_model[3]:.2f}, 请检查具体情况")
    if view:
        ground_plane = show_plane(ground_model, [1, 0, 0])
        ground_plane_ = show_plane(ground_model_, [0, 0, 1])
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Ground result", width=800, height=600, left=500, top=200)
        vis.add_geometry(filtered_pcd)
        vis.add_geometry(ground_plane)
        vis.add_geometry(ground_plane_)
        vis.run()
        vis.destroy_window()

    lid2base_m = np.array([[0.999, -0.011, -0.021, 352.101],
                           [0.011, 0.999, 0.000, -561.688],
                           [0.021, 0.000, 0.999, -392.580],
                           [0.000, 0.000, 0.000, 1.000]])

    # 角点位置始终由固定法向的三个平面求交：
    # 前面 +X，左/右面 ±Y，底面 -Z。实测地面法向只用于 method=4 的 Ry，
    # 不反向修改左右角点坐标。
    measured_ground_model = list(ground_model)
    ground_position_model = fixed_axis_plane(ground_model, axis=2, sign=-1)
    front_position_model = fixed_axis_plane(i_model, axis=0, sign=1)
    side_left_position_model = fixed_axis_plane(side_left_model, axis=1, sign=1)
    side_right_position_model = fixed_axis_plane(side_right_model, axis=1, sign=-1)
    front_right_position_model = (
        fixed_axis_plane(l_model, axis=0, sign=1) if method == 3 else None)

    if method in (1, 2, 4):
        corner_point1 = intersection_of_planes(front_position_model,
                                               ground_position_model, side_left_position_model)
        corner_point2 = intersection_of_planes(front_position_model,
                                               ground_position_model, side_right_position_model)
        o_be = np.array([(corner_point2[0] - corner_point1[0]),
                         (corner_point2[1] - corner_point1[1]),
                         (corner_point2[2] - corner_point1[2])])
        o_be = o_be / np.linalg.norm(o_be)
        o_no = np.array([-side_left_model[0], -side_left_model[1], -side_left_model[2]])
        # 计算点积
        dot_product = np.dot(o_be, o_no)
        # 计算模
        magnitude_a = np.linalg.norm(o_be)
        magnitude_b = np.linalg.norm(o_no)
        # 计算夹角的余弦值
        cos_theta = dot_product / (magnitude_a * magnitude_b)
        # 计算夹角，返回值是弧度
        theta = np.arccos(cos_theta)
        # 将弧度转换为角度
        theta_degrees = np.degrees(theta)
        if theta_degrees < 10:
            o_re = o_be
        else:
            o_re = o_be
        # corner_point1[0] = min(corner_point1[0], corner_point2[0])
        if method == 1:
            corner_point1[0] -= 0.08
        a_re = [0.0, 0.0, 1.0]
        n_re = np.cross(o_re, a_re)
        corner_point1_m = np.array([[n_re[0], o_re[0], a_re[0], corner_point1[0] * 1000],
                                    [n_re[1], o_re[1], a_re[1], corner_point1[1] * 1000],
                                    [n_re[2], o_re[2], a_re[2], corner_point1[2] * 1000],
                                    [0, 0, 0, 1]])
        corner_point2_m = np.array([[0, 0, 0, corner_point2[0] * 1000],
                                    [0, 0, 0, corner_point2[1] * 1000],
                                    [0, 0, 0, corner_point2[2] * 1000],
                                    [0, 0, 0, 1]])

        # 通过标定数据，转换至机器人基坐标系下
        point1_m = lid2base_m @ corner_point1_m
        point2_m = lid2base_m @ corner_point2_m
        point1 = matrix2euler(point1_m)
        point2 = matrix2euler(point2_m)
        if method == 4:
            raw_ry = ry_from_ground_normal(measured_ground_model, lid2base_m)
            point1[4] = relative_ry_from_0630_baseline(raw_ry)
        else:
            point1[4] = 0.0
        corner_list.append(point1)
        corner_list.append(point2)
        print(f'corner1 in lidar: x: {corner_point1[0] * 1000:.3f}, '
              f'y: {corner_point1[1] * 1000:.3f}, '
              f'z: {corner_point1[2] * 1000:.3f}')
        print(f'corner1 in robot base: x: {point1[0]:.3f}, y: {point1[1]:.3f}, z: {point1[2]:.3f}, '
              f'rx: {point1[3]:.3f}, ry: {point1[4]:.3f}, rz: {point1[5]:.3f}')
        print(f'corner2 in lidar: x: {corner_point2[0] * 1000:.3f},'
              f'y: {corner_point2[1] * 1000:.3f},'
              f'z: {corner_point2[2] * 1000:.3f}')
        print(f'corner2 in robot base: x: {point2[0]:.3f}, y: {point2[1]:.3f}, z: {point2[2]:.3f}')
        if view:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Two corner result", width=800, height=600, left=500, top=200)
            vis.add_geometry(filtered_pcd)
            vis.add_geometry(front_plane2)
            vis.add_geometry(side_l_plane2)
            vis.add_geometry(side_r_plane2)
            vis.add_geometry(ground_plane)
            vis.run()
            vis.destroy_window()

    if method == 3:
        corner_point1 = intersection_of_planes(front_position_model,
                                               ground_position_model, side_left_position_model)
        corner_point4 = intersection_of_planes(front_position_model,
                                               ground_position_model, side_right_position_model)
        o_be = np.array([(corner_point4[0] - corner_point1[0]),
                         (corner_point4[1] - corner_point1[1]),
                         (corner_point4[2] - corner_point1[2])])
        o_be = o_be / np.linalg.norm(o_be)
        o_no = np.array([-side_left_model[0], -side_left_model[1], -side_left_model[2]])
        # 计算点积
        dot_product = np.dot(o_be, o_no)
        # 计算模
        magnitude_a = np.linalg.norm(o_be)
        magnitude_b = np.linalg.norm(o_no)
        # 计算夹角的余弦值
        cos_theta = dot_product / (magnitude_a * magnitude_b)
        # 计算夹角，返回值是弧度
        theta = np.arccos(cos_theta)
        # 将弧度转换为角度
        theta_degrees = np.degrees(theta)
        if theta_degrees < 10:
            o_re = o_be
        else:
            o_re = o_be
        # corner_point1[0] = min(corner_point1[0], corner_point4[0])
        a_re = [0.0, 0.0, 1.0]
        n_re = np.cross(o_re, a_re)
        corner_point1_m = np.array([[n_re[0], o_re[0], a_re[0], corner_point1[0] * 1000],
                                    [n_re[1], o_re[1], a_re[1], corner_point1[1] * 1000],
                                    [n_re[2], o_re[2], a_re[2], corner_point1[2] * 1000],
                                    [0, 0, 0, 1]])
        corner_point4 = intersection_of_planes(front_right_position_model,
                                               ground_position_model, side_right_position_model)
        corner_point4_m = np.array([[0, 0, 0, corner_point4[0] * 1000],
                                    [0, 0, 0, corner_point4[1] * 1000],
                                    [0, 0, 0, corner_point4[2] * 1000],
                                    [0, 0, 0, 1]])
        corner_point2 = [corner_point1[0], corner_point1[1] - i_model[4], corner_point1[2]]
        corner_point2_m = np.array([[0, 0, 0, corner_point2[0] * 1000],
                                    [0, 0, 0, corner_point2[1] * 1000],
                                    [0, 0, 0, corner_point2[2] * 1000],
                                    [0, 0, 0, 1]])
        corner_point3 = [corner_point4[0], corner_point4[1] + l_model[4], corner_point4[2]]
        corner_point3_m = np.array([[0, 0, 0, corner_point3[0] * 1000],
                                    [0, 0, 0, corner_point3[1] * 1000],
                                    [0, 0, 0, corner_point3[2] * 1000],
                                    [0, 0, 0, 1]])
        # 通过标定数据，转换至机器人基坐标系下
        point1_m = lid2base_m @ corner_point1_m
        point2_m = lid2base_m @ corner_point2_m
        point3_m = lid2base_m @ corner_point3_m
        point4_m = lid2base_m @ corner_point4_m
        point1 = matrix2euler(point1_m)
        point2 = matrix2euler(point2_m)
        point3 = matrix2euler(point3_m)
        point4 = matrix2euler(point4_m)
        point1[4] = 0.0
        corner_list.append(point1)
        corner_list.append(point2)
        corner_list.append(point3)
        corner_list.append(point4)
        print(f'corner1 in lidar: x: {corner_point1[0] * 1000:.3f}, '
              f'y: {corner_point1[1] * 1000:.3f}, '
              f'z: {corner_point1[2] * 1000:.3f}')
        print(f'corner1 in robot base: x: {point1[0]:.3f}, y: {point1[1]:.3f}, z: {point1[2]:.3f}, '
              f'rx: {point1[3]:.3f}, ry: {point1[4]:.3f}, rz: {point1[5]:.3f}')
        print(f'corner2 in lidar: x: {corner_point2[0] * 1000:.3f},'
              f'y: {corner_point2[1] * 1000:.3f},'
              f'z: {corner_point2[2] * 1000:.3f}')
        print(f'corner2 in robot base: x: {point2[0]:.3f}, y: {point2[1]:.3f}, z: {point2[2]:.3f}')
        print(f'corner3 in lidar: x: {corner_point3[0] * 1000:.3f},'
              f'y: {corner_point3[1] * 1000:.3f},'
              f'z: {corner_point3[2] * 1000:.3f}')
        print(f'corner3 in robot base: x: {point3[0]:.3f}, y: {point3[1]:.3f}, z: {point3[2]:.3f}')
        print(f'corner4 in lidar: x: {corner_point4[0] * 1000:.3f},'
              f'y: {corner_point4[1] * 1000:.3f},'
              f'z: {corner_point4[2] * 1000:.3f}')
        print(f'corner4 in robot base: x: {point4[0]:.3f}, y: {point4[1]:.3f}, z: {point4[2]:.3f}')
    end_time = time.time()
    print(f'corner_list return: {corner_list}')
    print(f'cost time: {(end_time - start_time):.2f}')
    return corner_list

# file_path = "/home/fanuc/data_nav/carriage/trun_cloud_20250821_093142.pcd"
# pcd = o3d.io.read_point_cloud(file_path)
# process_point_cloud(pcd, 2)
