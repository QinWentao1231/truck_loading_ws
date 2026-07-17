import open3d as o3d
import numpy as np
import math
import time
import copy


# 0630 车厢内点云的地面法向 Ry 中位数，作为固定零点。
# method=4 返回相对该零点的 Ry；method=1/2/3/5 返回 Ry=0。
RY_INTERIOR_BASELINE_DEG = -2.08
RY_OUTPUT_DEADBAND_DEG = 0.2
MIN_PLANE_CANDIDATE_POINTS = 30
GROUND_SLICE_LOW_PERCENTILE = 40
GROUND_SLICE_BIN_WIDTH_M = 0.01
GROUND_SLICE_HALF_BAND_M = 0.06
GROUND_HEIGHT_OFFSET_M = 0.03
SPECIAL_FRONT_MIN_ANGLE_DEG = 20.0
SPECIAL_FRONT_MAX_ANGLE_DEG = 70.0
SPECIAL_FRONT_MAX_VERTICAL_TILT_DEG = 15.0
SPECIAL_FRONT_MIN_CLUSTER_POINTS = 300
SPECIAL_FRONT_MIN_PLANE_POINTS = 300
SPECIAL_FRONT_MIN_HEIGHT_M = 0.60
SPECIAL_FRONT_MIN_WIDTH_M = 0.08
SPECIAL_FRONT_MIN_INLIER_RATIO = 0.45
SIDE_ANGLE_MIN_X_M = 0.0
SIDE_ANGLE_FRONT_MARGIN_M = 0.03
SIDE_ANGLE_Y_HALF_BAND_M = 0.08
SIDE_ANGLE_MIN_Z_M = -0.5
SIDE_ANGLE_MAX_Z_M = 1.0
SIDE_ANGLE_X_BIN_WIDTH_M = 0.05
SIDE_ANGLE_MIN_BIN_POINTS = 100
SIDE_ANGLE_MIN_BINS = 10


class CornerDetectionCandidateError(ValueError):
    """角点检测候选点不足或无效。"""


def require_candidate_points(pcd, name, min_points=MIN_PLANE_CANDIDATE_POINTS):
    point_count = len(pcd.points)
    if point_count < min_points:
        raise CornerDetectionCandidateError(
            f"{name}候选点不足: {point_count} < {min_points}")
    return point_count


def require_candidate_values(values, name, min_values=1):
    value_count = len(values)
    if value_count < min_values:
        raise CornerDetectionCandidateError(
            f"{name}有效数据不足: {value_count} < {min_values}")
    return value_count


def slice_ground_height_layer(points, name="地面高度切片"):
    """从低位高度主峰附近截取地面层，排除侧边上空的水平点。"""
    candidate_points = np.asarray(points, dtype=float)
    require_candidate_values(
        candidate_points, name, min_values=MIN_PLANE_CANDIDATE_POINTS)

    z_values = candidate_points[:, 2]
    z_upper = float(np.percentile(
        z_values, GROUND_SLICE_LOW_PERCENTILE))
    low_z_values = z_values[z_values <= z_upper]
    require_candidate_values(
        low_z_values, f"{name}低位候选", min_values=MIN_PLANE_CANDIDATE_POINTS)

    z_min = math.floor(
        float(low_z_values.min()) / GROUND_SLICE_BIN_WIDTH_M
    ) * GROUND_SLICE_BIN_WIDTH_M
    z_max = math.ceil(
        float(low_z_values.max()) / GROUND_SLICE_BIN_WIDTH_M
    ) * GROUND_SLICE_BIN_WIDTH_M
    if z_max - z_min < GROUND_SLICE_BIN_WIDTH_M:
        peak_z = float(np.median(low_z_values))
    else:
        edges = np.arange(
            z_min,
            z_max + GROUND_SLICE_BIN_WIDTH_M * 1.5,
            GROUND_SLICE_BIN_WIDTH_M)
        histogram, edges = np.histogram(low_z_values, bins=edges)
        peak_index = int(np.argmax(histogram))
        peak_z = float(
            (edges[peak_index] + edges[peak_index + 1]) * 0.5)

    slice_mask = (
        np.abs(z_values - peak_z) <= GROUND_SLICE_HALF_BAND_M)
    sliced_points = candidate_points[slice_mask]
    require_candidate_values(
        sliced_points, name, min_values=MIN_PLANE_CANDIDATE_POINTS)
    return sliced_points, peak_z


def estimate_ground_normal_consensus(points, name="Ry地面", trials=31):
    """多次拟合水平面，使用法向中位数抑制单次跳变。"""
    candidate_points = np.asarray(points, dtype=float)
    if len(candidate_points) < MIN_PLANE_CANDIDATE_POINTS:
        raise CornerDetectionCandidateError(
            f"{name}候选点不足: {len(candidate_points)} < "
            f"{MIN_PLANE_CANDIDATE_POINTS}")

    candidate_pcd = o3d.geometry.PointCloud()
    candidate_pcd.points = o3d.utility.Vector3dVector(candidate_points)
    horizontal_cosine = math.cos(math.radians(10.0))
    fitted_normals = []
    for trial in range(trials):
        o3d.utility.random.seed(1009 + trial * 97)
        model, inliers = candidate_pcd.segment_plane(
            distance_threshold=0.005,
            ransac_n=3,
            num_iterations=1000)
        normal = np.asarray(model[:3], dtype=float)
        normal /= np.linalg.norm(normal)
        if normal[2] > 0:
            normal = -normal
        if -normal[2] < horizontal_cosine:
            continue
        fitted_normals.append((len(inliers), normal))

    if len(fitted_normals) < max(5, trials // 3):
        raise CornerDetectionCandidateError(
            f"{name}有效水平拟合不足: "
            f"{len(fitted_normals)} / {trials}")
    fitted_normals.sort(key=lambda item: item[0], reverse=True)
    best_normals = [item[1] for item in fitted_normals[:9]]
    normal = np.median(np.asarray(best_normals), axis=0)
    normal /= np.linalg.norm(normal)
    return [normal[0], normal[1], normal[2], 0.0]


def _wrap_angle_deg(angle):
    return (float(angle) + 180.0) % 360.0 - 180.0


def relative_ry_from_0630_baseline(raw_ry_deg):
    delta = _wrap_angle_deg(raw_ry_deg - RY_INTERIOR_BASELINE_DEG)
    return 0.0 if abs(delta) < RY_OUTPUT_DEADBAND_DEG else delta


def ry_from_ground_normal(ground_model, corner1, corner2):
    """按角点坐标系由地面法向计算 Ry。

    corner1 为原点，corner1 -> corner2 为 +Y，+Z 向上，
    并按右手系使用 +X = +Y x +Z。
    """
    normal = np.asarray(ground_model[:3], dtype=float)
    norm = np.linalg.norm(normal)
    if norm == 0:
        raise ValueError("地面法向为零，无法计算 Ry")
    up = -normal / norm

    y_axis = np.asarray(corner2, dtype=float) - np.asarray(corner1, dtype=float)
    y_norm = np.linalg.norm(y_axis)
    if y_norm == 0:
        raise ValueError("左右角点重合，无法建立 +Y 轴")
    y_axis /= y_norm
    z_axis = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(y_axis, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm == 0:
        raise ValueError("+Y 轴与 +Z 轴平行，无法建立 +X 轴")
    x_axis /= x_norm

    return math.degrees(math.atan2(np.dot(up, x_axis),
                                   np.dot(up, z_axis)))


def fixed_axis_plane(model, axis, sign):
    """保留平面在主轴上的交点，将法向固定为指定坐标轴。"""
    coefficient = float(model[axis])
    if abs(coefficient) < 1e-6:
        raise ValueError(f"平面在轴 {axis} 上的法向分量过小，无法固定法向")
    coordinate = -float(model[3]) / coefficient
    normal = [0.0, 0.0, 0.0]
    normal[axis] = float(sign)
    return [normal[0], normal[1], normal[2], -float(sign) * coordinate]


def expanded_side_direction_angle_to_x(points, reference_y, front_x,
                                       side_name):
    """在雷达到垛面之间按X分箱，稳健估计侧壁水平走向。"""
    cloud_points = np.asarray(points, dtype=float)
    max_x = float(front_x) - SIDE_ANGLE_FRONT_MARGIN_M
    if max_x <= SIDE_ANGLE_MIN_X_M:
        raise CornerDetectionCandidateError(
            f"{side_name}扩大角度范围无效: "
            f"[{SIDE_ANGLE_MIN_X_M:.3f}, {max_x:.3f}]m")
    candidate_mask = (
        np.isfinite(cloud_points).all(axis=1) &
        (cloud_points[:, 0] >= SIDE_ANGLE_MIN_X_M) &
        (cloud_points[:, 0] <= max_x) &
        (np.abs(cloud_points[:, 1] - reference_y) <=
         SIDE_ANGLE_Y_HALF_BAND_M) &
        (cloud_points[:, 2] >= SIDE_ANGLE_MIN_Z_M) &
        (cloud_points[:, 2] <= SIDE_ANGLE_MAX_Z_M)
    )
    candidate_points = cloud_points[candidate_mask]
    require_candidate_values(
        candidate_points, f"{side_name}扩大角度候选",
        min_values=MIN_PLANE_CANDIDATE_POINTS)

    edges = np.arange(
        SIDE_ANGLE_MIN_X_M,
        max_x + SIDE_ANGLE_X_BIN_WIDTH_M * 1.5,
        SIDE_ANGLE_X_BIN_WIDTH_M)
    representatives = []
    for x_low, x_high in zip(edges[:-1], edges[1:]):
        bin_mask = (
            (candidate_points[:, 0] >= x_low) &
            (candidate_points[:, 0] < x_high))
        bin_points = candidate_points[bin_mask]
        if len(bin_points) < SIDE_ANGLE_MIN_BIN_POINTS:
            continue
        representatives.append([
            float(np.median(bin_points[:, 0])),
            float(np.median(bin_points[:, 1]))])
    representatives = np.asarray(representatives, dtype=float)
    require_candidate_values(
        representatives, f"{side_name}扩大角度X分箱",
        min_values=SIDE_ANGLE_MIN_BINS)

    keep_mask = np.ones(len(representatives), dtype=bool)
    for _ in range(5):
        slope, intercept = np.polyfit(
            representatives[keep_mask, 0],
            representatives[keep_mask, 1], 1)
        residuals = (
            representatives[:, 1] -
            (slope * representatives[:, 0] + intercept))
        residual_median = float(np.median(residuals[keep_mask]))
        residual_mad = float(np.median(np.abs(
            residuals[keep_mask] - residual_median)))
        residual_limit = max(0.008, 3.0 * 1.4826 * residual_mad)
        next_keep_mask = (
            np.abs(residuals - residual_median) <= residual_limit)
        if next_keep_mask.sum() < SIDE_ANGLE_MIN_BINS:
            break
        if np.array_equal(next_keep_mask, keep_mask):
            keep_mask = next_keep_mask
            break
        keep_mask = next_keep_mask
    slope, _ = np.polyfit(
        representatives[keep_mask, 0],
        representatives[keep_mask, 1], 1)
    signed_angle = math.degrees(math.atan(float(slope)))
    return {
        "angle": signed_angle,
        "candidate_points": len(candidate_points),
        "used_bins": int(keep_mask.sum()),
        "total_bins": len(representatives),
        "min_x": SIDE_ANGLE_MIN_X_M,
        "max_x": max_x,
    }


def _refine_plane_model(points):
    """使用RANSAC内点最小二乘细化平面，并统一朝向+X。"""
    plane_points = np.asarray(points, dtype=float)
    require_candidate_values(
        plane_points, "异形车头平面内点",
        min_values=SPECIAL_FRONT_MIN_PLANE_POINTS)
    centroid = plane_points.mean(axis=0)
    _, _, vectors = np.linalg.svd(
        plane_points - centroid, full_matrices=False)
    normal = vectors[-1]
    normal /= np.linalg.norm(normal)
    if normal[0] < 0:
        normal = -normal
    offset = -float(np.dot(normal, centroid))
    return [float(normal[0]), float(normal[1]),
            float(normal[2]), offset]


def _fit_special_front_side(candidate_points, center_normal, side_name,
                            random_seed):
    """从一侧候选点中选出连续主斜面并拟合。"""
    candidate_pcd = o3d.geometry.PointCloud()
    candidate_pcd.points = o3d.utility.Vector3dVector(candidate_points)
    require_candidate_points(
        candidate_pcd, f"异形车头{side_name}候选",
        min_points=SPECIAL_FRONT_MIN_CLUSTER_POINTS)
    candidate_pcd = candidate_pcd.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2)[0]
    require_candidate_points(
        candidate_pcd, f"异形车头{side_name}离群过滤",
        min_points=SPECIAL_FRONT_MIN_CLUSTER_POINTS)

    labels = np.asarray(candidate_pcd.cluster_dbscan(
        eps=0.035, min_points=8, print_progress=False))
    valid_results = []
    if len(labels):
        for label in range(int(labels.max()) + 1):
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) < SPECIAL_FRONT_MIN_CLUSTER_POINTS:
                continue
            cluster_pcd = candidate_pcd.select_by_index(cluster_indices)
            extent = cluster_pcd.get_axis_aligned_bounding_box().get_extent()
            if (extent[2] < SPECIAL_FRONT_MIN_HEIGHT_M or
                    extent[1] < SPECIAL_FRONT_MIN_WIDTH_M):
                continue

            o3d.utility.random.seed(random_seed + label * 97)
            model, inliers = cluster_pcd.segment_plane(
                distance_threshold=0.006,
                ransac_n=3,
                num_iterations=8000)
            if len(inliers) < SPECIAL_FRONT_MIN_PLANE_POINTS:
                continue
            inlier_ratio = len(inliers) / len(cluster_indices)
            if inlier_ratio < SPECIAL_FRONT_MIN_INLIER_RATIO:
                continue

            inlier_pcd = cluster_pcd.select_by_index(inliers)
            refined_model = _refine_plane_model(inlier_pcd.points)
            normal = np.asarray(refined_model[:3], dtype=float)
            normal_angle = math.degrees(math.acos(np.clip(
                abs(float(np.dot(normal, center_normal))), -1.0, 1.0)))
            if not (SPECIAL_FRONT_MIN_ANGLE_DEG <= normal_angle <=
                    SPECIAL_FRONT_MAX_ANGLE_DEG):
                continue
            vertical_tilt = abs(math.degrees(math.asin(np.clip(
                float(normal[2]), -1.0, 1.0))))
            if vertical_tilt > SPECIAL_FRONT_MAX_VERTICAL_TILT_DEG:
                continue

            inlier_points = np.asarray(inlier_pcd.points)
            residuals = np.abs(
                inlier_points @ normal + refined_model[3])
            valid_results.append({
                "model": refined_model,
                "cloud": inlier_pcd,
                "inliers": len(inliers),
                "cluster_points": len(cluster_indices),
                "angle": normal_angle,
                "vertical_tilt": vertical_tilt,
                "extent": np.asarray(extent),
                "rms": float(np.sqrt(np.mean(residuals ** 2))),
            })

    if not valid_results:
        raise CornerDetectionCandidateError(
            f"异形车头{side_name}未找到满足连续性和尺寸要求的斜面")
    result = max(valid_results, key=lambda item: item["inliers"])
    model = result["model"]
    print(
        f'异形车头{side_name}: 内点={result["inliers"]}/'
        f'{result["cluster_points"]}, '
        f'相对正面夹角={result["angle"]:.2f}°, '
        f'垂直倾角={result["vertical_tilt"]:.2f}°, '
        f'范围Y={result["extent"][1]:.3f}m, '
        f'范围Z={result["extent"][2]:.3f}m, '
        f'RMS={result["rms"] * 1000:.2f}mm, '
        f'平面=[{model[0]:.5f}, {model[1]:.5f}, '
        f'{model[2]:.5f}, {model[3]:.5f}]'
    )
    return result["model"], result["cloud"]


def detect_special_front_planes(points, normals, center_model,
                                center_inlier_points):
    """以中间正面为基准，分别提取+Y左斜面和-Y右斜面。"""
    cloud_points = np.asarray(points, dtype=float)
    cloud_normals = np.asarray(normals, dtype=float)
    center_points = np.asarray(center_inlier_points, dtype=float)
    require_candidate_values(
        center_points, "异形车头中间正面",
        min_values=MIN_PLANE_CANDIDATE_POINTS)

    center_model = np.asarray(center_model[:4], dtype=float)
    center_norm = np.linalg.norm(center_model[:3])
    if center_norm == 0:
        raise CornerDetectionCandidateError("异形车头中间正面法向为零")
    center_normal = center_model[:3] / center_norm
    center_offset = center_model[3] / center_norm
    if center_normal[0] < 0:
        center_normal = -center_normal
        center_offset = -center_offset

    center_y = float(np.median(center_points[:, 1]))
    y_low, y_high = np.percentile(center_points[:, 1], [5, 95])
    center_half_span = max(center_y - float(y_low),
                           float(y_high) - center_y)
    outer_y_offset = max(0.40, center_half_span * 0.55)

    normal_cosines = np.abs(cloud_normals @ center_normal)
    normal_angles = np.degrees(np.arccos(np.clip(
        normal_cosines, -1.0, 1.0)))
    vertical_normal_limit = math.sin(math.radians(
        SPECIAL_FRONT_MAX_VERTICAL_TILT_DEG))
    center_distances = cloud_points @ center_normal + center_offset
    base_mask = (
        (normal_angles >= SPECIAL_FRONT_MIN_ANGLE_DEG) &
        (normal_angles <= SPECIAL_FRONT_MAX_ANGLE_DEG) &
        (np.abs(cloud_normals[:, 2]) <= vertical_normal_limit) &
        (center_distances >= -0.50) &
        (center_distances <= 0.08) &
        (np.abs(cloud_points[:, 1] - center_y) >= outer_y_offset)
    )
    print(
        f'异形车头斜面筛选: 中心Y={center_y:.3f}m, '
        f'外侧起点={outer_y_offset:.3f}m, '
        f'候选点={int(base_mask.sum())}'
    )

    left_mask = base_mask & (cloud_points[:, 1] > center_y)
    right_mask = base_mask & (cloud_points[:, 1] < center_y)
    left_model, left_cloud = _fit_special_front_side(
        cloud_points[left_mask], center_normal, "左斜面", 5101)
    right_model, right_cloud = _fit_special_front_side(
        cloud_points[right_mask], center_normal, "右斜面", 6101)
    return left_model, right_model, left_cloud, right_cloud


def _process_point_cloud_impl(pcd, method):
    # #保存pcd点云到路径
    # target_file_path = "/home/fanuc/Test_ws/corn_poits.pcd"
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


    # target_file_path_ply = "/home/fanuc/Test_ws/corn.ply"
    # #保存ply点云到目标路径
    # success = o3d.io.write_point_cloud(target_file_path_ply, pcd)
    # if success:
    #     print(f"传入点云已成功保存到 {target_file_path_ply}")
    # else:
    #     print("点云保存失败")


    def segment_plane(pcd, distance_threshold=0.005, ransac_n=3,
                      num_iterations=10000, candidate_name="平面"):
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
        require_candidate_points(
            pcd, candidate_name, max(ransac_n, MIN_PLANE_CANDIDATE_POINTS))
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
        normal = normal / np.linalg.norm(normal)  # 单位化目标法向
        # 计算平面法向量的旋转
        initial_normal = np.array([0, 0, 1])  # 初始法向量是 Z 轴方向
        axis = np.cross(initial_normal, normal)  # 旋转轴是初始法向量与目标法向量的叉积
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            # normal 与 initial_normal 共线（同向或反向），用任意垂直轴
            axis = np.array([1.0, 0.0, 0.0])
            angle = 0.0 if np.dot(initial_normal, normal) > 0 else np.pi
        else:
            axis = axis / axis_norm
            cos_angle = np.clip(np.dot(initial_normal, normal), -1.0, 1.0)
            angle = np.arccos(cos_angle)
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
        require_candidate_points(pcd, "L垛面聚类")
        labels = np.array(pcd.cluster_dbscan(eps=0.03, min_points=10))
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_count_dict = dict(zip(unique_labels, counts))
        min_cluster_size = 1000
        valid_labels = [label for label in unique_labels
                        if label != -1 and label_count_dict[label] >= min_cluster_size]
        if not valid_labels:
            raise CornerDetectionCandidateError(
                "L垛面未找到点数达到 1000 的有效聚类")
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
            plane_model, plane_inliers, _ = segment_plane(
                cluster_pcd, candidate_name=f"L垛面聚类 {label}")
            cluster_planes.append(plane_inliers)
            d_list.append(abs(plane_model[3]))
        d_array = np.array(d_list)
        require_candidate_values(d_array, "L垛面聚类平面")
        d_threshold = d_array.min() + (d_array.max() - d_array.min()) * (1 / 4)
        selected_clusters = [
            cluster_planes[i] for i, d in enumerate(d_list) if d <= d_threshold
        ]
        other_clusters = [
            cluster_planes[i] for i, d in enumerate(d_list) if d >= d_threshold
        ]
        if not selected_clusters:
            raise CornerDetectionCandidateError(
                "L垛面未找到满足 d 阈值条件的聚类")
        if not other_clusters:
            raise CornerDetectionCandidateError("L垛面缺少另一组平面聚类")
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
        final_plane_model, _, _ = segment_plane(
            merged_points, candidate_name="L垛面第一组")
        aabb = merged_points.get_axis_aligned_bounding_box()
        bounding_box_final = aabb.get_extent()
        other_plane_model, _, _ = segment_plane(
            other_points, candidate_name="L垛面第二组")
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
    view =  True
    debug = view
    view_normal = False
    corner_list = []
    # 1: 车头波纹板；2: I垛面；3: L垛面；
    # 4: I垛面角点，并返回相对车厢内基准的 Ry；
    # 5: 异形车头，返回正面与左右斜面交线在地面上的两个点。
    print(f'method: {method}')
    _FRONT_RIB_MIN = 0.05            # 车头加强筋兜底补偿(m)：筋检测不足时至少前移此距离
    _FRONT_RIB_MAX = 0.10            # 车头加强筋补偿上限(m)：检测过深时最多前移此距离
    _RIB_PCT       = 1               # 筋深分位(%)：取最凸向货舱的此分位均值作筋深(越小越激进)
    _SIDE_PCT      = 10              # 左右壁面分位(%)：取最凸入车厢的此分位均值作壁面位置
    _GROUND_PCT    = 10              # 地板分位(%)：取最浅（最高Z）的点均值作地板位置
    _SIDE_LAYER_GAP = 0.05           # 平行表面的 y 向分层间隔(m)，用于剔除垛面侧面
    down_pcd = pcd
    if view:
        down_pcd = fiterCloud(down_pcd)
        # 仅在Down窗口排除会破坏相机包围盒的异常超大坐标，
        # 后续角点算法仍使用原始 down_pcd。
        display_points = np.asarray(down_pcd.points)
        display_mask = (
            np.isfinite(display_points).all(axis=1) &
            (np.abs(display_points) < 100.0).all(axis=1))
        display_pcd = o3d.geometry.PointCloud()
        display_pcd.points = o3d.utility.Vector3dVector(
            display_points[display_mask])
        require_candidate_points(display_pcd, "Down可视化", min_points=1)
        removed_display_points = len(display_points) - len(display_pcd.points)
        if removed_display_points:
            print(f'Down可视化已排除异常超远点: {removed_display_points}')
        # display_pcd.paint_uniform_color([0.2, 0.2, 0.2])
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Down", width=800, height=600, left=500, top=200)
        vis.add_geometry(display_pcd, reset_bounding_box=True)
        vis.run()
        vis.destroy_window()

    side_angle_source_points = np.asarray(down_pcd.points)

    # 垛面点云直通滤波
    points = np.asarray(down_pcd.points)
    x_filtered = (points[:, 0] <= 2.0) & (points[:, 0] >= 1.0)
    y_filtered = (points[:, 1] <= 2.0) & (points[:, 1] >= -2.0)
    z_filtered = (points[:, 2] <= 1.0) & (points[:, 2] >= -1.0)
    filtered_points = points[x_filtered & y_filtered & z_filtered]
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd = fiterCloud(filtered_pcd)
    require_candidate_points(filtered_pcd, "ROI滤波", min_points=50)
    filtered_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))
    normals = np.asarray(filtered_pcd.normals)
    points = np.asarray(filtered_pcd.points)

    if method in (1, 2, 4, 5):
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
        [a, b, c, d], inlier_cloud, outlier_cloud = segment_plane(
            front_pcd, candidate_name="垛面")
        if a < 0:
            a, b, c, d = -a, -b, -c, -d
        rib_points = np.empty((0, 3))
        if method == 1:
            # 车头加强筋：用法向滤波后(未经统计滤波删除)的全部前面点，
            # 取最凸向货舱的 _RIB_PCT% 分位均值作筋深，再夹在 [_FRONT_RIB_MIN, _FRONT_RIB_MAX]
            n_main = np.array([a, b, c]) / np.linalg.norm([a, b, c])
            all_dist = front_points @ n_main + d
            thr = np.percentile(all_dist, _RIB_PCT)
            rib_points = front_points[all_dist <= thr]   # 被判为筋的最凸 _RIB_PCT% 点
            delta = float(min(all_dist[all_dist <= thr].mean(), 0))
            delta_eff = max(-_FRONT_RIB_MAX, min(delta, -_FRONT_RIB_MIN))
            i_model = [a, b, c, d - delta_eff]
            print(f'车头筋补偿：检测 delta={delta*1000:.1f}mm, 实际采用={-delta_eff*1000:.1f}mm, '
                  f'筋点数={len(rib_points)}')
        else:
            i_model = [a, b, c, d]
        if method == 5:
            (special_left_model,
             special_right_model,
             special_left_cloud,
             special_right_cloud) = detect_special_front_planes(
                points, normals, i_model, inlier_cloud.points)
        if view:
            front_plane2 = show_plane(i_model, [0, 1, 0])
            if method == 5:
                special_left_plane = show_plane(
                    special_left_model, [1, 0.4, 0])
                special_right_plane = show_plane(
                    special_right_model, [0, 0.4, 1])
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Front result", width=800, height=600, left=500, top=200)
            vis.add_geometry(filtered_pcd)
            vis.add_geometry(front_plane2)
            if method == 5:
                left_show = copy.deepcopy(special_left_cloud)
                right_show = copy.deepcopy(special_right_cloud)
                left_show.paint_uniform_color([1, 0, 0])
                right_show.paint_uniform_color([0, 0, 1])
                vis.add_geometry(left_show)
                vis.add_geometry(right_show)
                vis.add_geometry(special_left_plane)
                vis.add_geometry(special_right_plane)
            vis.run()
            vis.destroy_window()
            # 单独弹窗显示被判为筋的点（品红）叠加在原始点云（灰）上
            if len(rib_points):
                rib_pcd = o3d.geometry.PointCloud()
                rib_pcd.points = o3d.utility.Vector3dVector(rib_points)
                rib_pcd.paint_uniform_color([1, 0, 1])
                bg = copy.deepcopy(filtered_pcd)
                bg.paint_uniform_color([0.7, 0.7, 0.7])
                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name=f"Rib points (品红, {len(rib_points)}点)",
                                  width=800, height=600, left=500, top=200)
                vis.add_geometry(bg)
                vis.add_geometry(rib_pcd)
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
        # 滤波结果可视化：过滤后的前面板点云(红) + 原始点云(灰)背景
        if view:
            bg = copy.deepcopy(filtered_pcd)
            bg.paint_uniform_color([0.7, 0.7, 0.7])
            fg = copy.deepcopy(front_pcd)
            fg.paint_uniform_color([1, 0, 0])
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Front filtered", width=800, height=600)
            vis.add_geometry(bg)
            vis.add_geometry(fg)
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
        raise Exception(
            f'Wrong method value: {method}; expected 1, 2, 3, 4, or 5')

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
    angle_threshold = np.cos(np.radians(10))

    left_normal_mask = cos_theta > angle_threshold
    x_threshold = -i_model[3] - 0.03
    z_threshold = -0.5
    mask = left_normal_mask & ((-i_model[3]-0.6) < points[:, 0]) & (points[:, 0] < x_threshold) & (points[:, 2] > z_threshold)
    left_points = np.asarray(filtered_pcd.points)[mask]
    left_pcd = o3d.geometry.PointCloud()
    left_pcd.points = o3d.utility.Vector3dVector(left_points)
    # 显示法向筛选 + mask直通滤波后、统计离群过滤前的左壁候选点。
    if view:
        left_mask_pcd = copy.deepcopy(left_pcd)
        left_mask_pcd.paint_uniform_color([1, 0, 0])
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Left wall after mask BEFORE outlier ({len(left_points)} points)",
            width=800, height=600, left=500, top=200)
        vis.add_geometry(left_mask_pcd)
        vis.run()
        vis.destroy_window()
    left_pcd = fiterCloud(left_pcd)

    # 法向相同的左壁和垛面侧面按 y 连续性分层。从有足够点数的层中
    # 选择最外侧（y 最大）的层作为真正左壁。
    left_before_layer_points = np.asarray(left_pcd.points).copy()
    left_after_layer_points = left_before_layer_points
    if len(left_before_layer_points):
        y_values = left_before_layer_points[:, 1]
        sorted_indices = np.argsort(y_values)
        split_positions = np.where(
            np.diff(y_values[sorted_indices]) > _SIDE_LAYER_GAP
        )[0] + 1
        y_layers = np.split(sorted_indices, split_positions)
        largest_layer_size = max(len(layer) for layer in y_layers)
        min_layer_size = max(20, int(np.ceil(largest_layer_size * 0.1)))
        valid_layers = [
            layer for layer in y_layers if len(layer) >= min_layer_size
        ]
        if not valid_layers:
            valid_layers = [max(y_layers, key=len)]
        left_wall_indices = max(
            valid_layers,
            key=lambda layer: float(np.median(y_values[layer]))
        )
        left_after_layer_points = left_before_layer_points[left_wall_indices]
        left_pcd.points = o3d.utility.Vector3dVector(left_after_layer_points)
        left_points = left_after_layer_points
        print(
            f'左壁分层处理：处理前 {len(left_before_layer_points)} 点，'
            f'处理后 {len(left_after_layer_points)} 点，'
            f'剔除 {len(left_before_layer_points) - len(left_after_layer_points)} 点'
        )

    # 分层处理前：显示全部左壁候选点（红）
    if view:
        before_pcd = o3d.geometry.PointCloud()
        before_pcd.points = o3d.utility.Vector3dVector(left_before_layer_points)
        before_pcd.paint_uniform_color([1, 0, 0])
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Left wall BEFORE layer cleaning ({len(left_before_layer_points)} points)",
            width=800, height=600, left=500, top=200
        )
        vis.add_geometry(before_pcd)
        vis.run()
        vis.destroy_window()

        # 分层处理后：只显示最终参与侧壁分位计算的左壁点（绿）
        after_pcd = o3d.geometry.PointCloud()
        after_pcd.points = o3d.utility.Vector3dVector(left_after_layer_points)
        after_pcd.paint_uniform_color([0, 1, 0])
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Left wall AFTER layer cleaning ({len(left_after_layer_points)} points)",
            width=800, height=600, left=500, top=200
        )
        vis.add_geometry(after_pcd)
        vis.run()
        vis.destroy_window()
    [a, b, c, d], inlier_cloud, outlier_cloud = segment_plane(
        left_pcd, candidate_name="左侧壁")
    if b < 0:
        a, b, c, d = -a, -b, -c, -d
    temp_points = np.asarray(outlier_cloud.points)
    left_wall_points = np.asarray(left_pcd.points)
    left_percentile_mask = np.zeros(len(left_wall_points), dtype=bool)
    if len(temp_points) != 0:
        n_main = np.array([0, 1, 0])
        all_distances = left_wall_points @ n_main
        positive_mask = all_distances > 0
        positive_distances = all_distances[positive_mask]
        require_candidate_values(positive_distances, "左侧壁正向距离")
        thr_l = np.percentile(positive_distances, _SIDE_PCT)
        left_percentile_mask = positive_mask & (all_distances <= thr_l)
        delta = float(all_distances[left_percentile_mask].mean())
        side_left_model = [0, 1, 0, -delta]
    else:
        side_left_model = [a, b, c, d]
    if view:
        left_colored_pcd = o3d.geometry.PointCloud()
        left_colored_pcd.points = o3d.utility.Vector3dVector(left_wall_points)
        left_colors = np.tile([0.0, 1.0, 0.0], (len(left_wall_points), 1))
        left_colors[left_percentile_mask] = [1.0, 0.0, 0.0]
        left_colored_pcd.colors = o3d.utility.Vector3dVector(left_colors)
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=(
                f"Left wall 10pct: red={left_percentile_mask.sum()}, "
                f"green={len(left_wall_points) - left_percentile_mask.sum()}"),
            width=800, height=600, left=500, top=200)
        vis.add_geometry(left_colored_pcd)
        vis.run()
        vis.destroy_window()

    # 右侧面点云滤波
    if view_normal:
        o3d.visualization.draw_geometries([filtered_pcd], point_show_normal=True)
    target_normal = np.array([0, -1, 0])
    cos_theta = np.dot(normals, target_normal)
    angle_threshold = np.cos(np.radians(10))
    # right_indices = np.where(cos_theta > angle_threshold)[0]
    x_threshold = -i_model[3]-0.03 if method != 3 else -l_model[3]-0.03
    z_threshold = -0.5
    mask = (cos_theta > angle_threshold) & ((-i_model[3]-0.6) < points[:, 0]) & (points[:, 0] < x_threshold) & (points[:, 2] > z_threshold)
    right_points = np.asarray(filtered_pcd.points)[mask]
    right_pcd = o3d.geometry.PointCloud()
    right_pcd.points = o3d.utility.Vector3dVector(right_points)
    # 显示法向筛选 + mask直通滤波后、统计离群过滤前的右壁候选点。
    if view:
        right_mask_pcd = copy.deepcopy(right_pcd)
        right_mask_pcd.paint_uniform_color([1, 0, 0])
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Right wall after mask BEFORE outlier ({len(right_points)} points)",
            width=800, height=600, left=500, top=200)
        vis.add_geometry(right_mask_pcd)
        vis.run()
        vis.destroy_window()
    right_pcd = fiterCloud(right_pcd)

    # 法向相同的右壁和垛面侧面按 y 连续性分层。从有足够点数的层中
    # 选择最外侧（y 最小）的层作为真正右壁。
    right_before_layer_points = np.asarray(right_pcd.points).copy()
    right_after_layer_points = right_before_layer_points
    if len(right_before_layer_points):
        y_values = right_before_layer_points[:, 1]
        sorted_indices = np.argsort(y_values)
        split_positions = np.where(
            np.diff(y_values[sorted_indices]) > _SIDE_LAYER_GAP
        )[0] + 1
        y_layers = np.split(sorted_indices, split_positions)
        largest_layer_size = max(len(layer) for layer in y_layers)
        min_layer_size = max(20, int(np.ceil(largest_layer_size * 0.1)))
        valid_layers = [
            layer for layer in y_layers if len(layer) >= min_layer_size
        ]
        if not valid_layers:
            valid_layers = [max(y_layers, key=len)]
        right_wall_indices = min(
            valid_layers,
            key=lambda layer: float(np.median(y_values[layer]))
        )
        right_after_layer_points = right_before_layer_points[right_wall_indices]
        right_pcd.points = o3d.utility.Vector3dVector(right_after_layer_points)
        right_points = right_after_layer_points
        print(
            f'右壁分层处理：处理前 {len(right_before_layer_points)} 点，'
            f'处理后 {len(right_after_layer_points)} 点，'
            f'剔除 {len(right_before_layer_points) - len(right_after_layer_points)} 点'
        )

    # 分层处理前：显示全部右壁候选点（红）
    if view:
        before_pcd = o3d.geometry.PointCloud()
        before_pcd.points = o3d.utility.Vector3dVector(right_before_layer_points)
        before_pcd.paint_uniform_color([1, 0, 0])
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Right wall BEFORE layer cleaning ({len(right_before_layer_points)} points)",
            width=800, height=600, left=500, top=200
        )
        vis.add_geometry(before_pcd)
        vis.run()
        vis.destroy_window()

        # 分层处理后：只显示最终参与侧壁分位计算的右壁点（绿）
        after_pcd = o3d.geometry.PointCloud()
        after_pcd.points = o3d.utility.Vector3dVector(right_after_layer_points)
        after_pcd.paint_uniform_color([0, 1, 0])
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Right wall AFTER layer cleaning ({len(right_after_layer_points)} points)",
            width=800, height=600, left=500, top=200
        )
        vis.add_geometry(after_pcd)
        vis.run()
        vis.destroy_window()
    [a, b, c, d], inlier_cloud, outlier_cloud = segment_plane(
        right_pcd, candidate_name="右侧壁")
    if b > 0:
        a, b, c, d = -a, -b, -c, -d
    temp_points = np.asarray(outlier_cloud.points)
    right_wall_points = np.asarray(right_pcd.points)
    right_percentile_mask = np.zeros(len(right_wall_points), dtype=bool)
    if len(temp_points) != 0:
        n_main = np.array([0, -1, 0])
        distances = right_wall_points @ n_main
        require_candidate_values(distances, "右侧壁距离")
        thr_r = np.percentile(distances, _SIDE_PCT)
        right_percentile_mask = distances <= thr_r
        delta = float(distances[right_percentile_mask].mean())
        side_right_model = [0, -1, 0, -delta]
    else:
        side_right_model = [a, b, c, d]
    if view:
        right_colored_pcd = o3d.geometry.PointCloud()
        right_colored_pcd.points = o3d.utility.Vector3dVector(
            right_wall_points)
        right_colors = np.tile(
            [0.0, 1.0, 0.0], (len(right_wall_points), 1))
        right_colors[right_percentile_mask] = [1.0, 0.0, 0.0]
        right_colored_pcd.colors = o3d.utility.Vector3dVector(right_colors)
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=(
                f"Right wall 10pct: red={right_percentile_mask.sum()}, "
                f"green={len(right_wall_points) - right_percentile_mask.sum()}"),
            width=800, height=600, left=500, top=200)
        vis.add_geometry(right_colored_pcd)
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
    angle_threshold = np.cos(np.radians(15))

    ground_mask = ((cos_theta > angle_threshold) &
                   (points[:, 2] < 0.0) & (points[:, 0] > 1.0) &
                   (points[:, 0] < x_threshold))
    ground_points = points[ground_mask]
    ground_pcd = o3d.geometry.PointCloud()
    ground_pcd.points = o3d.utility.Vector3dVector(ground_points)
    # 法向筛选后再做统计离群点过滤，避免稀疏噪点干扰
    # 地面主高度层和 Ry 地面法向。
    ground_pcd = fiterCloud(ground_pcd)
    require_candidate_points(ground_pcd, "地面离群过滤")
    ground_before_slice_points = np.asarray(ground_pcd.points).copy()
    ground_points, ground_peak_z = slice_ground_height_layer(
        ground_before_slice_points)
    ground_pcd.points = o3d.utility.Vector3dVector(ground_points)
    print(
        f'地面高度切片: 主峰={ground_peak_z:.4f}m, '
        f'范围=[{ground_peak_z - GROUND_SLICE_HALF_BAND_M:.4f}, '
        f'{ground_peak_z + GROUND_SLICE_HALF_BAND_M:.4f}]m, '
        f'处理前={len(ground_before_slice_points)}, '
        f'处理后={len(ground_points)}'
    )
    # 可视化高度切片后的地面点（红）叠加在原始ROI点云（灰）上。
    if view:
        bg = copy.deepcopy(filtered_pcd)
        bg.paint_uniform_color([0.7, 0.7, 0.7])
        fg = copy.deepcopy(ground_pcd)
        fg.paint_uniform_color([1, 0, 0])
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=(
                f"Ground height slice ({len(ground_before_slice_points)} -> "
                f"{len(ground_points)} points)"),
            width=800, height=600, left=500, top=200)
        vis.add_geometry(bg)
        vis.add_geometry(fg)
        vis.run()
        vis.destroy_window()
    # 按固定向下法向投影，取距离最小的10%（即Z最高的10%）求平均，
    # 用该平均值定位角点Z。地面拟合法向只用于method=4计算Ry。
    ground_distances = ground_points @ np.array([0.0, 0.0, -1.0])
    ground_distances = ground_distances[ground_distances > 0.0]
    require_candidate_values(ground_distances, "地面正向距离")
    ground_threshold = np.percentile(ground_distances, _GROUND_PCT)
    ground_top_distances = ground_distances[
        ground_distances <= ground_threshold]
    require_candidate_values(ground_top_distances, "地面10%分位点")
    ground_delta = float(ground_top_distances.mean())
    raw_ground_z = -ground_delta
    ground_z = raw_ground_z + GROUND_HEIGHT_OFFSET_M
    ground_model = [0.0, 0.0, -1.0, ground_z]
    print(f'地面10%分位均值: raw_z={raw_ground_z:.4f}m, '
          f'补偿后z={ground_z:.4f}m, '
          f'点数={len(ground_top_distances)}/{len(ground_distances)}')

    # Ry 对高度接近的水平面做多次拟合，以法向中位数
    # 抑制单次Open3D RANSAC在多个平面之间的随机跳变。
    measured_ground_model = estimate_ground_normal_consensus(ground_points)
    ground_normal = np.asarray(measured_ground_model[:3], dtype=float)

    # 地面向下法向 n=[a,b,c]：RY = atan2(-nx, -nz)。
    # 前垛面朝前法向 n=[a,b,c]：RY = atan2(-nz, nx)。
    front_normal = np.asarray(i_model[:3], dtype=float)
    front_normal /= np.linalg.norm(front_normal)
    ry_from_ground = np.degrees(np.arctan2(-ground_normal[0], -ground_normal[2]))
    ry_from_front = np.degrees(np.arctan2(-front_normal[2], front_normal[0]))
    ry_difference = abs((ry_from_ground - ry_from_front + 180.0) % 360.0 - 180.0)
    print(
        f'俯仰角RY：地面法向={ry_from_ground:.3f}°, '
        f'垛面法向={ry_from_front:.3f}°, 差值={ry_difference:.3f}°'
    )

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
        ground_plane = show_plane(ground_model, [0, 1, 0])
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Ground result", width=800, height=600, left=500, top=200)
        vis.add_geometry(filtered_pcd)
        vis.add_geometry(ground_plane)
        vis.run()
        vis.destroy_window()

    lid2base_m = np.array([[-0.018, -0.999, -0.004, 266.14],
                           [0.999, -0.018, 0, 428.673],
                           [0.000, -0.004, 1, -738.086],
                           [0.000, 0.000, 0.000, 1.000]])

    # 角点 XYZ 由三平面求交：垛面使用实测拟合法向，
    # 左右侧面和底面仍使用固定法向。method=5的两个角点分别由
    # 中间正面、对应斜面和地面求交；实测地面法向仅供method=4计算Ry。
    ground_position_model = fixed_axis_plane(ground_model, axis=2, sign=-1)
    front_position_model = list(i_model[:4])
    side_left_position_model = fixed_axis_plane(side_left_model, axis=1, sign=1)
    side_right_position_model = fixed_axis_plane(side_right_model, axis=1, sign=-1)
    front_right_position_model = (
        list(l_model[:4]) if method == 3 else None)

    if method in (1, 2, 4, 5):
        if method == 5:
            corner_point1 = intersection_of_planes(
                front_position_model,
                list(special_left_model[:4]),
                ground_position_model)
            corner_point2 = intersection_of_planes(
                front_position_model,
                list(special_right_model[:4]),
                ground_position_model)
        else:
            corner_point1 = intersection_of_planes(
                front_position_model,
                ground_position_model, side_left_position_model)
            corner_point2 = intersection_of_planes(
                front_position_model,
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
        # 车头加强筋补偿已并入 i_model（前平面前移），此处不再额外偏移角点
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
        point1[3] = corner_point1[1]
        point2[3] = corner_point2[1]
        point2[4] = 0.0
        if method == 4:
            raw_ry = ry_from_ground_normal(
                measured_ground_model, corner_point1, corner_point2)
            point1[4] = relative_ry_from_0630_baseline(raw_ry)
        else:
            point1[4] = 0.0
        corner_list.append(point1)
        corner_list.append(point2)
        print(f'corner1 in lidar: x: {corner_point1[0] * 1000:.3f}, '
              f'y: {corner_point1[1] * 1000:.3f}, '
              f'z: {corner_point1[2] * 1000:.3f}')
        print(f'corner1 in robot base: x: {point1[0]:.3f}, y: {point1[1]:.3f}, z: {point1[2]:.3f},'
              f'rx: {point1[3] * 1000:.3f}, ry: {point1[4]:.3f}, rz: {point1[5]:.3f}')
        print(f'corner2 in lidar: x: {corner_point2[0] * 1000:.3f},'
              f'y: {corner_point2[1] * 1000:.3f},'
              f'z: {corner_point2[2] * 1000:.3f}')
        print(f'corner2 in robot base: x: {point2[0]:.3f}, y: {point2[1]:.3f}, z: {point2[2]:.3f},'
              f'rx: {point2[3] * 1000:.3f}, ry: {point2[4]:.3f}, rz: {point2[5]:.3f}')
        if view:
            final_side_left_plane = show_plane(
                side_left_model, [0, 0, 1])
            final_side_right_plane = show_plane(
                side_right_model, [0, 0, 1])
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Two corner result", width=800, height=600, left=500, top=200)
            vis.add_geometry(filtered_pcd)
            vis.add_geometry(front_plane2)
            if method == 5:
                vis.add_geometry(special_left_plane)
                vis.add_geometry(special_right_plane)
            vis.add_geometry(final_side_left_plane)
            vis.add_geometry(final_side_right_plane)
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
        point2[4] = 0.0
        point3[4] = 0.0
        point4[4] = 0.0
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
    # 扩大范围的侧壁方向仅用于调试打印，放在全部角点计算完成后执行，
    # 不参与也不改变前面用于角点求交的侧壁模型。
    if debug:
        try:
            left_reference_point = np.median(left_wall_points, axis=0)
            right_reference_point = np.median(right_wall_points, axis=0)
            left_reference_y = -(
                side_left_model[0] * left_reference_point[0] +
                side_left_model[2] * left_reference_point[2] +
                side_left_model[3]) / side_left_model[1]
            right_reference_y = -(
                side_right_model[0] * right_reference_point[0] +
                side_right_model[2] * right_reference_point[2] +
                side_right_model[3]) / side_right_model[1]
            left_front_x = -(
                i_model[1] * left_reference_y +
                i_model[2] * left_reference_point[2] +
                i_model[3]) / i_model[0]
            right_front_model = l_model if method == 3 else i_model
            right_front_x = -(
                right_front_model[1] * right_reference_y +
                right_front_model[2] * right_reference_point[2] +
                right_front_model[3]) / right_front_model[0]
            left_angle_debug = expanded_side_direction_angle_to_x(
                side_angle_source_points, left_reference_y,
                left_front_x, "左侧面")
            right_angle_debug = expanded_side_direction_angle_to_x(
                side_angle_source_points, right_reference_y,
                right_front_x, "右侧面")
            left_angle = left_angle_debug["angle"]
            right_angle = right_angle_debug["angle"]
            print(
                f'侧面方向与+X轴夹角(扩大范围): '
                f'左={abs(left_angle):.3f}° '
                f'(方向角{left_angle:+.3f}°, '
                f'X={left_angle_debug["min_x"]:.2f}~'
                f'{left_angle_debug["max_x"]:.2f}m, '
                f'点数={left_angle_debug["candidate_points"]}, '
                f'分箱={left_angle_debug["used_bins"]}/'
                f'{left_angle_debug["total_bins"]}), '
                f'右={abs(right_angle):.3f}° '
                f'(方向角{right_angle:+.3f}°, '
                f'X={right_angle_debug["min_x"]:.2f}~'
                f'{right_angle_debug["max_x"]:.2f}m, '
                f'点数={right_angle_debug["candidate_points"]}, '
                f'分箱={right_angle_debug["used_bins"]}/'
                f'{right_angle_debug["total_bins"]})'
            )
        except (CornerDetectionCandidateError, ValueError,
                np.linalg.LinAlgError) as exc:
            print(f'侧面扩大范围角度计算失败（不影响角点）: {exc}')
    end_time = time.time()
    print(f'cost time: {(end_time - start_time):.2f}')
    return corner_list


def process_point_cloud(pcd, method):
    """执行角点检测；候选点不足时安全返回空列表。"""
    try:
        return _process_point_cloud_impl(pcd, method)
    except CornerDetectionCandidateError as exc:
        print(f'角点检测失败: {exc}')
        print('corner_list return: []')
        return []


file_path = "/home/qinwentao/workcells/truck_loading_ws/log/robot_process/pcd_logs/0716/trun_cloud_20260716_101157_slow1.pcd"
pcd = o3d.io.read_point_cloud(file_path)
process_point_cloud(pcd, 2)
