import copy
import open3d as o3d
import numpy as np
import math
import time
import rclpy


def segment_plane(pcd, distance_threshold=0.001, ransac_n=3, num_iterations=100000):
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
        cluster_planes[i] for i, d in enumerate(d_list) if d < d_threshold
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

    return [[final_plane_model[0], final_plane_model[1], final_plane_model[2], final_plane_model[3],
             bounding_box_final[1]],
            [other_plane_model[0], other_plane_model[1], other_plane_model[2], other_plane_model[3],
             bounding_box_other[1]]]

def point_to_plane_distance(x, y, z, a, b, c, d):
    """
    计算点 (x, y, z) 到平面 ax + by + cz + d = 0 的有符号距离
    """
    numerator = a * x + b * y + c * z + d
    denominator = math.sqrt(a * a + b * b + c * c)

    if denominator == 0:
        return 0.0  # 避免除零

    return abs(numerator / denominator)

def stitching_pcd(pcd1, pcd2, angle):
    # 假设你有一个变换矩阵，将第二个点云转换到第一个点云的坐标系下
    # 这里我们用一个简单的旋转和位移矩阵作为示例
    # 变换矩阵形式为 4x4 矩阵，其中前三列为旋转矩阵，第四列为平移向量
    transformation_matrix = np.array([[0.866, -0.5, 0, 1],
                                    [0.5, 0.866, 0, 2],
                                    [0, 0, 1, 3],
                                    [0, 0, 0, 1]])

    # 应用变换矩阵到第二个点云
    pcd2.transform(transformation_matrix)

    # 合并两个点云
    combined_pcd = pcd1 + pcd2

    # 可视化合并后的点云
    o3d.visualization.draw_geometries([combined_pcd])

def rotation_pcd(matrix, pcd):
    """
    生成绕X轴、Y轴、Z轴旋转的合成旋转矩阵。
    参数:
    roll -- 绕X轴旋转的角度（弧度）
    pitch -- 绕Y轴旋转的角度（弧度）
    yaw -- 绕Z轴旋转的角度（弧度）
    """
    translation_vector = np.array([matrix[0], matrix[1], matrix[2]])  # 例如：平移 (1, 2, 3)
    roll = math.radians(matrix[3])
    pitch = math.radians(matrix[4])
    yaw = math.radians(matrix[5])

    # 绕 X 轴旋转矩阵
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(roll), -math.sin(roll)],
                    [0, math.sin(roll), math.cos(roll)]])
    
    # 绕 Y 轴旋转矩阵
    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]])
    
    # 绕 Z 轴旋转矩阵
    R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1]])
    
    # 复合旋转矩阵：先绕 Z 轴旋转，再绕 Y 轴旋转，再绕 X 轴旋转
    rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))

    # 构建4x4的变换矩阵
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix  # 旋转部分
    transformation_matrix[:3, 3] = translation_vector  # 平移部分

    # 应用变换矩阵到第二个点云
    pcd.transform(transformation_matrix)
    return pcd

def valid_pcd(pcd):
    pts = np.asarray(pcd.points)
    # 掩码：去掉 NaN、inf
    mask = np.isfinite(pts).all(axis=1)
    clean_pts = pts[mask]
    pcd.points = o3d.utility.Vector3dVector(clean_pts)
    return pcd

def process_point_cloud(pcdLow, pcdHigh, length):
    # #保存pcd点云到路径
    # target_file_path = "/home/simplenav/Test_ws/corn_poits.pcd"
    # success = o3d.io.write_point_cloud(target_file_path, pcd, write_ascii=True)
    # if success:
    #     print(f"传入点云已成功保存到 {target_file_path}")
    # else:
    #     print("点云保存失败")
    pcdLow = valid_pcd(pcdLow)
    pcdHigh = valid_pcd(pcdHigh)
    pcd = pcdLow + pcdHigh
    # 打印点云数量
    num_points = len(np.asarray(pcd.points))
    print(f"******输入点云数量为 : {num_points}")
    start_time = time.time()
    global view
    view = True
    view_normal = True
    down_pcd = pcd
    if view:
        # down_pcd = fiterCloud(down_pcd)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Down", width=800, height=600, left=500, top=200)
        vis.add_geometry(down_pcd)
        vis.run()
        vis.destroy_window()
        down_pcd.paint_uniform_color([0.2, 0.2, 0.2])

    # 垛面点云直通滤波
    points = np.asarray(down_pcd.points)
    x_filtered = (points[:, 0] <= 0.5) & (points[:, 0] >= -1.5)
    y_filtered = (points[:, 1] <= -1.0) & (points[:, 1] >= -2.0)
    # z_filtered = (points[:, 2] <= 2.0) & (points[:, 2] >= -1.5)
    z_filtered = (points[:, 2] <= 0) & (points[:, 2] >= -1.05)
    filtered_points = points[x_filtered & y_filtered & z_filtered]
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    # filtered_pcd = fiterCloud(filtered_pcd)
    filtered_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))
    normals = np.asarray(filtered_pcd.normals)
    points = np.asarray(filtered_pcd.points)

    # 左侧面点云滤波
    flip_mask = (normals[:, 0] * (points[:, 0])) < 0
    # 批量翻转
    normals[flip_mask] *= -1
    filtered_pcd.normals = o3d.utility.Vector3dVector(normals)
    if view_normal:
        o3d.visualization.draw_geometries([filtered_pcd], point_show_normal=True)
    normals = np.asarray(filtered_pcd.normals)
    target_normal = np.array([1, 0, 0])
    cos_theta = np.dot(normals, target_normal)
    angle_threshold = np.cos(np.radians(15))
    mask = (cos_theta > angle_threshold)
    # mask = (points[:, 0] < x_threshold) & (points[:, 2] > -0.5)
    left_points = np.asarray(filtered_pcd.points)[mask]
    left_pcd = o3d.geometry.PointCloud()
    left_pcd.points = o3d.utility.Vector3dVector(left_points)
    print(f"左侧点云数量：{len(left_pcd.points)}")
    # left_pcd = fiterCloud(left_pcd)
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
        n_main = np.array([1, 0, 0])
        n_main = n_main / np.linalg.norm(n_main)
        # 计算所有点到主平面的有符号距离
        # distances = temp_points @ n_main + d
        distances = temp_points @ n_main
        # distances = distances[distances<0]
        distances = distances[distances > 0]
        # delta = max(min(np.percentile(distances, 40), 0), -0.05)
        delta = np.percentile(distances, 50)
        # delta_min = min(np.percentile(distances, 0), 0)
        delta_min = np.percentile(distances, 0)
        # side_left_model = [a, b, c, d - delta]
        side_left_model = [1, 0, 0, -delta_min]
        # side_left_model = [0, 1, 0, d - delta]
        # side_left_model = [a, b, c, d]
        side_left_model_ = [a, b, c, d]
    else:
        side_left_model = [a, b, c, d]
        side_left_model_ = [a, b, c, d]
    if view:
        # side_l_plane1 = show_plane([a, b, c, d], [1, 0, 0])
        # side_l_plane2 = show_plane(side_left_model, [0, 1, 0])
        side_l_plane3 = show_plane(side_left_model_, [0, 0, 1])
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="side_left", width=800, height=600, left=500, top=200)
        vis.add_geometry(filtered_pcd)
        # vis.add_geometry(side_l_plane1)
        # vis.add_geometry(side_l_plane2)
        vis.add_geometry(side_l_plane3)
        vis.run()
        vis.destroy_window()

    # 右侧面点云滤波
    target_normal = np.array([-1, 0, 0])
    cos_theta = np.dot(normals, target_normal)
    angle_threshold = np.cos(np.radians(15))
    # right_indices = np.where(cos_theta > angle_threshold)[0]
    mask = (cos_theta > angle_threshold)
    # mask = (points[:, 0] < x_threshold)
    right_points = np.asarray(filtered_pcd.points)[mask]
    right_pcd = o3d.geometry.PointCloud()
    right_pcd.points = o3d.utility.Vector3dVector(right_points)
    print(f"右侧点云数量：{len(right_pcd.points)}")
    # right_pcd = fiterCloud(right_pcd)
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
        n_main = np.array([-1, 0, 0])
        n_main = n_main / np.linalg.norm(n_main)
        # 计算所有点到主平面的有符号距离
        distances = temp_points @ n_main
        distances = distances[distances > 0]
        delta = np.percentile(distances, 50)
        delta_min = np.percentile(distances, 0)
        side_right_model = [-1, 0, 0, -delta]
        # side_right_model = [a, b, c, d]
        side_right_model_ = [a, b, c, d]
    else:
        side_right_model = [a, b, c, d]
        side_right_model_ = [a, b, c, d]
    if view:
        # side_r_plane1 = show_plane([a, b, c, d], [0, -1, 0])
        side_r_plane2 = show_plane(side_right_model, [0, 1, 0])
        # side_r_plane3 = show_plane(side_right_model_, [0, 0, 1])
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="side_right", width=800, height=600, left=500, top=200)
        vis.add_geometry(filtered_pcd)
        # vis.add_geometry(side_r_plane1)
        vis.add_geometry(side_r_plane2)
        # vis.add_geometry(side_r_plane3)
        vis.run()
        vis.destroy_window()
    print(f'distance1: {(point_to_plane_distance(1.5,0,0.25, side_left_model[0],side_left_model[1],side_left_model[2],side_left_model[3])+point_to_plane_distance(1.5, 0,0.25, side_right_model[0],side_right_model[1],side_right_model[2],side_right_model[3]))*1000:.1f}mm')
    print(f'distance2: {abs(side_left_model[3]+side_right_model[3])*1000:.1f}mm')
    end_time = time.time()
    print(f'cost time: {(end_time - start_time):.2f}')
    return end_time-start_time, int(point_to_plane_distance(1.5,0,0.25, side_left_model[0],side_left_model[1],side_left_model[2],side_left_model[3])+point_to_plane_distance(1.5, 0,0.25, side_right_model[0],side_right_model[1],side_right_model[2],side_right_model[3]))*1000

file_path = "/home/qinwentao/workcells/truck_loading_ws/src/pointcloud_subscriber/data/lidar_low.pcd"
pcdLow = o3d.io.read_point_cloud(file_path)
file_path = "/home/qinwentao/workcells/truck_loading_ws/src/pointcloud_subscriber/data/lidar_high.pcd"
pcdHigh = o3d.io.read_point_cloud(file_path)
process_point_cloud(pcdLow, pcdHigh, 392*2)
# ctime_list = []
# dis_list = []
# for i in range(1):
#     pcd_ = copy.deepcopy(pcd)
#     ctime, dis = process_point_cloud(pcd_)
#     ctime_list.append(ctime)
#     dis_list.append(dis)
# print(f"{len(ctime_list)}次平均耗时: {sum(ctime_list)/len(ctime_list):.2f}")
# print(f"{len(dis_list)}次平均距离: {sum(dis_list)/len(dis_list):.2f}")
# print(f"{len(dis_list)}次最大偏差: {max(abs(max(dis_list)-696), abs(min(dis_list)-975)):.2f}")