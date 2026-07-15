import copy
import os
import open3d as o3d
import numpy as np
import math
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np

# 调试模式开关：True 才向终端打印运行日志（测量宽度、保存路径等）
# 离线测试入口 __main__ 内会自动置 True
DEBUG = False

def _dbg(msg):
    if DEBUG:
        print(msg)


# ══════════════════════════════════════════════════════════════════════════
# 测宽算法参数（集中配置，便于统一调整）
# ══════════════════════════════════════════════════════════════════════════
# ── 双雷达采集 ──
LIDAR_FRAMES = 3                 # 每个雷达累计采集帧数
LIDAR_TIMEOUT_SEC = 5.0          # 采集超时(秒)：topic 未发布/帧数不够时超时，按计算失败处理

# ── 直通滤波范围(米)：从合并点云裁出一个 y 切片用于测宽 ──
PASS_X = (-2.0, 2.0)             # 宽度方向(x)保留范围
PASS_Y = (-2.5, -1.0)            # 深度方向(y)保留范围 = 测宽切片位置
PASS_Z = (-1.5, 2.0)             # 高度方向(z)保留范围(宽松；顶部薄片噪声由 MIN_CLUSTER_ZSPAN_M 过滤)
PASS_MIN_PTS = 10                # 直通后最少点数，不足则报错

# ── 法向量滤波：保留法向接近 ±x 的侧面点 ──
NORMAL_KNN = 20                  # 法向估计的近邻点数 (从30降到20以提高速度)
NORMAL_ANGLE_DEG = 20            # 法向与 x 轴夹角阈值(度)：20° 能抓到第一层箱面残余侧面点(10°漏掉)

# ── DBSCAN 聚类 ──
DBSCAN_EPS = 0.15                # 邻域半径(米)：从0.2降到0.15，提高聚类精度同时保持速度
DBSCAN_MIN_POINTS = 8            # 成簇最少点数：从10降到8，保持敏感度

# ── 簇筛选与左右配对 ──
MIN_CLUSTER_PTS = 15             # 簇最小点数：15 能保住更稀疏的第一层箱面，又能过滤孤立噪点
MIN_CLUSTER_ZSPAN_M = 0.02       # 簇最小 z 跨度(米)：0.10 过滤顶部薄片(<0.08)，保住稀疏第一层(0.1-0.4)
MAX_CLUSTER_CZ_M = 1.0           # 簇 z 重心上限(米)：> 此值视为车厢顶部凸起结构(管线/灯)，丢弃

# ── 当前面 Y 锁定（深度方向，只保留最靠雷达的当前面箱，滤掉后排箱）──
FRONT_Y_BIN = 0.05               # Y 直方图 bin 宽(米)
FRONT_Y_MIN_PTS = 100            # Y bin 视为"有箱"的最小点数
FRONT_Y_DEPTH = 0.45             # 当前面保留深度(米)：从前沿往后取此窗口=最小箱长，
                                 #   保证落在前排箱内、不碰后排箱（箱长 450~530mm）

# ── 当前行 Z 自适应锁定（避免相邻层干扰）──
LAYER_SEARCH_TOL_RATIO = 0.4     # 搜箱顶容差 = box_h × 此比例(±)：理论与实际的最大偏差(40%箱高)
LAYER_TOP_NORMAL_DEG = 15        # 水平面识别角度阈值(度)：|nz| > cos(此值) 视为箱顶水平面
LAYER_TOP_MIN_PTS = 10           # 当前层箱顶水平面最少点数（高层点云较稀疏）
LAYER_Z_HIST_BIN = 0.01          # z 直方图 bin 宽(米)：1cm，对应实际放置精度量级
LAYER_PEAK_MIN_PTS = 3           # 箱顶候选高度 bin 的最少水平面点数
LAYER_PEAK_MIN_RATIO = 0.10      # 候选 bin 点数至少达到当前最强峰的此比例
LAYER_Z_MARGIN_TOP = 0.03        # 锁定 z 范围上沿留余量(米)：略高于本层顶面，防漏点
LAYER_Z_MARGIN_BOTTOM = 0.03     # 锁定 z 范围下沿留余量(米)：略高于本层底面，避下层顶面
FLOOR_MIN_PTS = 50              # 地板检测取最低点数：取 z 最小的此数个点的中位数作地板，
                                #   地板被货物遮挡仅剩零星点时仍稳定，又抗个别雷达噪点
FLOOR_Z_DEFAULT = -1.13         # 首层地板不可见时的标定回退值（雷达坐标系，米）
FLOOR_Z_MAX_DEVIATION = 0.15    # 自动标定相对回退值的最大允许偏差，超出视为噪点/遮挡
MIN_VALID_WIDTH_MM = 500         # 测宽下限(mm)，小于此判为货物窄缝 → 向外重选
MAX_Z_DIFF_M = 0.15              # 两箱面 z 重心最大高度差(米)，超出判为跨层(仅两侧都是箱面时校验)
WALL_PTS = 1000                  # 簇点数 ≥ 此判为墙，跳过 z 重心校验(单侧壁合并后约 1200 点)
FACE_PCT = 10                    # 取簇内侧面 FACE_PCT% 的点求面位置(抗噪)

# 地板在同一码垛过程中不变。第一层还能看到地板时标定，高层遮挡后复用，
# 避免把货物或车体上的低点误当地板。
_floor_z_cached = None

# ── 偏航补偿（雷达绕机器人 J1 轴摆动）──
# 拍照位 J1 与正对 J1 不同 → 点云绕 J1 轴偏航，需转回正对系再测量。
# J1 轴在雷达坐标系的水平位置(ax,ay)和旋转方向由"双角度同场景"标定确定。
J1_AXIS_XY = (0.269, 0.506)      # J1 轴水平位置 (ax, ay) 米；由车厢场景 5 个 J1 角度(-40.5~-80.5° 跨度40°)联合标定，残差中位数 12mm
J1_DEROTATE_SIGN = 1             # 补偿旋转方向：雷达系绕 J1 轴的旋转角 = θ_photo - θ_face = +yaw_offset_deg

# ── 可视化 ──
VIEW = False                     # 可视化总开关：True 才弹出各阶段点云窗口(原始/法向/聚类/拟合)
VIEW_FRONT_Y = 4.0               # 原始点云仅显示雷达前方此距离(米)内的点


class DualLidarOneShot(Node):

    def __init__(
        self,
        topic1='/lidar_points1',
        topic2='/lidar_points2',
        max_frames=LIDAR_FRAMES
    ):
        super().__init__('dual_lidar_oneshot')

        self.max_frames = max_frames

        # 雷达1
        self.frames1 = []
        self.count1 = 0
        self.merged_cloud1 = None

        # 雷达2
        self.frames2 = []
        self.count2 = 0
        self.merged_cloud2 = None

        self.sub1 = self.create_subscription(
            PointCloud2, topic1, self.cb1, 10
        )
        self.sub2 = self.create_subscription(
            PointCloud2, topic2, self.cb2, 10
        )

        self.get_logger().info(
            f"Waiting for {max_frames} frames from each lidar..."
        )

    def cb1(self, msg: PointCloud2):
        if self.count1 >= self.max_frames:
            return

        cloud = self.msg_to_np(msg)
        if cloud.size == 0:
            return

        self.frames1.append(cloud)
        self.count1 += 1

        self.get_logger().info(
            f"Lidar1: {self.count1}/{self.max_frames}"
        )

        self.try_finish()

    def cb2(self, msg: PointCloud2):
        if self.count2 >= self.max_frames:
            return

        cloud = self.msg_to_np(msg)
        if cloud.size == 0:
            return

        self.frames2.append(cloud)
        self.count2 += 1

        self.get_logger().info(
            f"Lidar2: {self.count2}/{self.max_frames}"
        )

        self.try_finish()

    def try_finish(self):
        if (
            self.count1 >= self.max_frames
            and self.count2 >= self.max_frames
        ):
            self.merged_cloud1 = np.vstack(self.frames1)
            self.merged_cloud2 = np.vstack(self.frames2)

            self.get_logger().info(
                f"Finished. "
                f"Lidar1 pts: {self.merged_cloud1.shape[0]}, "
                f"Lidar2 pts: {self.merged_cloud2.shape[0]}"
            )

            self._done = True

    @staticmethod
    def msg_to_np(msg: PointCloud2):
        # 向量化解析：直接从 structured array 提取 xyz 三列，避免 Python 逐点迭代
        data = point_cloud2.read_points(msg, ('x', 'y', 'z'), skip_nans=True)
        if data.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        return np.stack([data['x'], data['y'], data['z']], axis=-1).astype(np.float64)

def collect_dual_lidar_once(
    topic1='/lidar_points1',
    topic2='/lidar_points2',
    frames=LIDAR_FRAMES,
    timeout_sec=LIDAR_TIMEOUT_SEC
):
    """双雷达各采 frames 帧后合并。timeout_sec 内未集齐（topic 未发布等）→ 返回 (None, None)。"""
    rclpy.init()
    node = DualLidarOneShot(topic1, topic2, frames)
    node._done = False
    # 用 spin_once 循环替代 spin，避免在回调内 shutdown 导致 spin 不能正常返回
    _start = time.monotonic()
    while rclpy.ok() and not node._done:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.monotonic() - _start > timeout_sec:
            break
    timed_out = (not node._done) or node.merged_cloud1 is None or node.merged_cloud2 is None
    if timed_out:
        _dbg(f"采集超时 {timeout_sec}s：lidar1={node.count1}/{frames} "
             f"lidar2={node.count2}/{frames}（topic 是否在发布？）")
        node.destroy_node()
        rclpy.shutdown()
        return None, None
    pc1 = o3d.geometry.PointCloud()
    pc1.points = o3d.utility.Vector3dVector(node.merged_cloud1)
    pc2 = o3d.geometry.PointCloud()
    pc2.points = o3d.utility.Vector3dVector(node.merged_cloud2)
    node.destroy_node()
    rclpy.shutdown()
    return pc1, pc2

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


view = VIEW  # 兼容旧函数(clustFrontBoard)的小写开关，统一跟随 VIEW

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


def _derotate_about_j1(pts, angle_deg, axis_xy):
    """绕过 axis_xy=(ax,ay) 的竖直轴(平行 z)，将点云水平旋转 angle_deg 度，z 不变。

    用于把拍照位(J1 偏移)采集的点云转回正对参考系。
    axis_xy（J1 轴水平位置）和 angle_deg 的符号由"双角度同场景"标定确定。
    pts: Nx3 ndarray；返回旋转后的 Nx3（不改原数组）。
    """
    if not angle_deg or axis_xy is None:
        return pts
    ax, ay = axis_xy
    th = np.radians(angle_deg)
    c, s = np.cos(th), np.sin(th)
    x = pts[:, 0] - ax
    y = pts[:, 1] - ay
    out = pts.copy()
    out[:, 0] = x * c - y * s + ax
    out[:, 1] = x * s + y * c + ay
    return out


def _detect_floor_z(pts):
    """从点云自标定地板 z（雷达坐标系，米）：取车厢内(PASS_XY)最低 FLOOR_MIN_PTS 个点的中位数。
    地板是最低的物理面，但常被货物大面积遮挡导致其点极稀疏（低分位数都抓不到）——
    直接取 z 最小的若干点取中位数，既贴近真地板又抗个别雷达噪点。
    找不到时退回 PASS_Z 下沿。"""
    xy = pts[(pts[:, 0] >= PASS_X[0]) & (pts[:, 0] <= PASS_X[1]) &
             (pts[:, 1] >= PASS_Y[0]) & (pts[:, 1] <= PASS_Y[1]) &
             (pts[:, 2] >= PASS_Z[0]) & (pts[:, 2] <= PASS_Z[1])]
    if len(xy) < FLOOR_MIN_PTS:
        return PASS_Z[0]
    lowest = np.partition(xy[:, 2], FLOOR_MIN_PTS - 1)[:FLOOR_MIN_PTS]
    return float(np.median(lowest))


def _resolve_floor_z(pts, rel_top_h, box_h):
    """首层自动标定地板并缓存，后续层复用。

    只在 rel_top_h 接近一个箱高时重新标定，因为第二层开始地板通常已被遮挡。
    标定值距离现场回退值过大时拒绝，防止极低噪点或车体结构污染缓存。
    """
    global _floor_z_cached
    is_first_layer = rel_top_h <= box_h * 1.5
    if is_first_layer:
        detected = _detect_floor_z(pts)
        if np.isfinite(detected) and abs(detected - FLOOR_Z_DEFAULT) <= FLOOR_Z_MAX_DEVIATION:
            _floor_z_cached = detected
            _dbg(f"地板Z首层自动标定：{detected:.3f}m")
        else:
            _dbg(f"地板Z自动标定值 {detected:.3f}m 不可信，使用回退值 {FLOOR_Z_DEFAULT:.3f}m")
    return _floor_z_cached if _floor_z_cached is not None else FLOOR_Z_DEFAULT


def _lock_layer_z_range(pts, rel_top_h, box_h, y_range=None):
    """根据当前抓"距地板的理论顶面高度"把 Z 范围锁定到当前行，隔离上下相邻层干扰。
    内部自标定地板 z，再用实测箱顶精修，理论值仅作粗定位（容差 box_h×比例）。

    pts: Nx3 ndarray（已偏航补偿）；rel_top_h: 当前层顶面距地板高度(米，相对值)；
    box_h: 当前抓箱子竖向高度(米)。
    返回 (z_min, z_max, actual_top_z)。
    """
    floor_z = _resolve_floor_z(pts, rel_top_h, box_h)
    theo_top_z = floor_z + rel_top_h        # 理论顶面在雷达系的 z
    tol = box_h * LAYER_SEARCH_TOL_RATIO
    y_min, y_max = y_range if y_range is not None else PASS_Y
    # 理论顶面附近搜水平面点（箱顶），用实测精修。
    # 先限制在车厢 XY 测量区内：原来只裁 Z，会把整个场景中的同高度点
    # 都送入 KNN 法向估计，百万点点云下既慢，又容易被车厢外的水平结构干扰。
    band_mask = (
        (pts[:, 0] >= PASS_X[0]) & (pts[:, 0] <= PASS_X[1]) &
        (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max) &
        (pts[:, 2] >= theo_top_z - tol) & (pts[:, 2] <= theo_top_z + tol)
    )
    band = pts[band_mask]
    actual_top_z = theo_top_z
    if len(band) >= 30:
        _bp = o3d.geometry.PointCloud()
        _bp.points = o3d.utility.Vector3dVector(band)
        _bp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=NORMAL_KNN))
        nz = np.abs(np.asarray(_bp.normals)[:, 2])
        top_pts = band[nz > np.cos(np.radians(LAYER_TOP_NORMAL_DEG))]
        if len(top_pts) >= LAYER_TOP_MIN_PTS:
            z_lo, z_hi = top_pts[:, 2].min(), top_pts[:, 2].max()
            nbin = max(1, int((z_hi - z_lo) / LAYER_Z_HIST_BIN))
            hist, edges = np.histogram(top_pts[:, 2], bins=nbin)
            centers = (edges[:-1] + edges[1:]) * 0.5
            # 窗口内可能同时存在车体横梁、下层箱顶等水平面。不取全局最强峰，
            # 而是先过滤掉稀疏噪点，再选最接近理论顶面的高度峰。
            min_support = max(LAYER_PEAK_MIN_PTS,
                              int(np.ceil(hist.max() * LAYER_PEAK_MIN_RATIO)))
            candidates = np.flatnonzero(hist >= min_support)
            if len(candidates):
                best = candidates[np.argmin(np.abs(centers[candidates] - theo_top_z))]
                actual_top_z = float(centers[best])
            _dbg(f"当前行Z锁定：地板={floor_z:.3f}m 理论顶={theo_top_z:.3f}m 实测顶={actual_top_z:.3f}m "
                 f"(偏差 {(actual_top_z - theo_top_z) * 1000:+.0f}mm, 箱顶点={len(top_pts)})")
        else:
            _dbg(f"当前行Z锁定：理论顶附近无足够箱顶水平面，退回理论值 {theo_top_z:.3f}m（地板={floor_z:.3f}m）")
    else:
        _dbg(f"当前行Z锁定：理论顶附近点数不足({len(band)})，退回理论值 {theo_top_z:.3f}m（地板={floor_z:.3f}m）")

    z_min = actual_top_z - box_h + LAYER_Z_MARGIN_BOTTOM
    z_max = actual_top_z + LAYER_Z_MARGIN_TOP
    return z_min, z_max, actual_top_z


def _lock_front_face_y(pts):
    """深度(y)方向只保留最靠雷达的"当前面"箱，滤掉后排箱。
    已知箱长 450~530mm：定位最靠雷达(y 最大)的第一个密集 bin = 当前面前沿，
    从前沿往后取 FRONT_Y_DEPTH（最小箱长）窗口即可，保证落在前排箱内、不碰后排。
    返回 (y_min, y_max)；点数不足时退回 PASS_Y。"""
    if len(pts) < FRONT_Y_MIN_PTS:
        return PASS_Y
    nbin = max(1, int((pts[:, 1].max() - pts[:, 1].min()) / FRONT_Y_BIN))
    hist, edges = np.histogram(pts[:, 1], bins=nbin)
    # 定位最靠雷达的第一个密集 bin = 当前面前沿
    i = len(hist) - 1
    while i >= 0 and hist[i] < FRONT_Y_MIN_PTS:
        i -= 1
    if i < 0:
        return PASS_Y
    y_front = float(edges[i + 1])
    y_back = max(y_front - FRONT_Y_DEPTH, PASS_Y[0])
    return (y_back, y_front)


def _compute_width(pc1, pc2, view=None, yaw_offset_deg=0.0,
                   rel_top_h=None, box_h=None):
    """
    合并双雷达点云，测量左右侧面总宽度（mm）。

    view=None 时在调用时读取顶部 VIEW 开关（避免默认参数在定义时被绑死）。
    yaw_offset_deg: 拍照位相对正对的偏航角(度)。雷达绕机器人 J1 轴摆动导致点云偏航，
                    需将点云转回正对系再测量。
    rel_top_h / box_h: 当前抓"顶面距地板的高度"和箱子竖向高度(米)。两者都给定时，
                    内部自标定地板 + 实测箱顶，把直通滤波 Z 锁定到"当前行"隔离相邻层；
                    否则用全局 PASS_Z。

    策略：
      0. 偏航补偿 + （可选）当前行 Z 锁定
      1. 直通滤波
      2. 法向量滤波保留 ±x 侧面点
      3. DBSCAN 聚类，取最大两簇按 x 重心区分左右
      4. 固定法向 [1,0,0]，d 取靠近原点方向 10% 百分位
      5. 宽度 = 右面 x - 左面 x
    """
    if view is None:
        view = VIEW
    
    # 记录开始时间
    total_start_time = time.time()
    
    # 一次性合并点云并转换为numpy数组，避免后续重复转换
    pcd_combined = valid_pcd(pc1) + valid_pcd(pc2)
    pts = np.asarray(pcd_combined.points)

    # ── 0. 偏航补偿：绕 J1 轴把点云转回正对系 ────────────────────────────────
    # 雷达绕 J1 轴摆 yaw_offset_deg → 点云在水平面内偏航。标定出 J1_AXIS_XY 后启用。
    if yaw_offset_deg and J1_AXIS_XY is not None:
        _ang = J1_DEROTATE_SIGN * yaw_offset_deg
        pts = _derotate_about_j1(pts, _ang, J1_AXIS_XY)
        _dbg(f"偏航补偿：绕 J1 轴 {J1_AXIS_XY} 旋转 {_ang:+.1f}°")
    elif yaw_offset_deg:
        _dbg(f"偏航补偿角={yaw_offset_deg:.1f}°，但 J1_AXIS_XY 未标定 → 跳过补偿")

    start_time = time.time()

    # ── 0b. 先用全高点云锁定当前面 Y，避免箱顶搜索被后排/车体干扰 ──
    pass_y = PASS_Y
    _global_roi = (
        (pts[:, 0] >= PASS_X[0]) & (pts[:, 0] <= PASS_X[1]) &
        (pts[:, 1] >= PASS_Y[0]) & (pts[:, 1] <= PASS_Y[1]) &
        (pts[:, 2] >= PASS_Z[0]) & (pts[:, 2] <= PASS_Z[1])
    )
    _ymin, _ymax = _lock_front_face_y(pts[_global_roi])
    if (_ymin, _ymax) != PASS_Y:
        pass_y = (_ymin, _ymax)

    # ── 0c. 当前行 Z 锁定（可选）：自标定地板 + 实测箱顶，把 Z 收窄到当前行 ──────
    pass_z = PASS_Z
    _z_locked = False
    if rel_top_h is not None and box_h is not None:
        _zmin, _zmax, _ = _lock_layer_z_range(
            pts, rel_top_h, box_h, y_range=pass_y)
        # 与全局 PASS_Z 取交集，防止锁定范围越出有效区
        pass_z = (max(PASS_Z[0], _zmin), min(PASS_Z[1], _zmax))
        _z_locked = True
        _dbg(f"当前行Z范围：[{pass_z[0]:.3f}, {pass_z[1]:.3f}] m")

    # Z 锁定后再用当前层点云精修 Y；全高锁定失败时这一步仍可恢复。
    _xz = ((pts[:, 0] >= PASS_X[0]) & (pts[:, 0] <= PASS_X[1]) &
           (pts[:, 1] >= PASS_Y[0]) & (pts[:, 1] <= PASS_Y[1]) &
           (pts[:, 2] >= pass_z[0]) & (pts[:, 2] <= pass_z[1]))
    _ymin, _ymax = _lock_front_face_y(pts[_xz])
    if (_ymin, _ymax) != PASS_Y:
        pass_y = (_ymin, _ymax)
        _dbg(f"当前面Y范围：[{pass_y[0]:.3f}, {pass_y[1]:.3f}] m（已滤掉后排箱）")

    # 记录预处理阶段耗时
    preprocessing_time = time.time() - start_time
    start_time = time.time()

    # ── 1. 直通滤波 ──────────────────────────────────────────────────────────
    mask = (
        (pts[:, 0] >= PASS_X[0]) & (pts[:, 0] <= PASS_X[1]) &
        (pts[:, 1] >= pass_y[0]) & (pts[:, 1] <= pass_y[1]) &
        (pts[:, 2] >= pass_z[0]) & (pts[:, 2] <= pass_z[1])
    )
    
    # 添加早期退出检查，如果直通滤波后点数太少直接返回
    filtered_pts = pts[mask]
    if len(filtered_pts) < PASS_MIN_PTS:
        _dbg("计算失败：直通滤波后点数不足，请检查坐标范围参数")
        return None

    if view:
        # 原始点云(灰) + 直通框选中部分(绿)，直观看裁剪框相对整体的位置
        # 仅显示用裁剪：车厢长轴(y)太长，只显示雷达前方 VIEW_FRONT_Y 米内的点
        # 灰色背景 z 不跟随当前行锁定，显示完整车厢高度，便于看绿色切片落在哪一层
        # x 留 1m 余量裁掉车外远点；y 限前方窗口（不影响计算，只清理可视化）
        view_margin = 1.0
        front = pts[(pts[:, 1] >= -VIEW_FRONT_Y) & (pts[:, 1] <= 0.5) &
                    (pts[:, 0] >= PASS_X[0] - view_margin) & (pts[:, 0] <= PASS_X[1] + view_margin)
                    & (pts[:, 2] >= PASS_Z[0]) & (pts[:, 2] <= PASS_Z[1])]
        raw = o3d.geometry.PointCloud()
        raw.points = o3d.utility.Vector3dVector(front)
        raw.paint_uniform_color([0.6, 0.6, 0.6])
        sel = o3d.geometry.PointCloud()
        sel.points = o3d.utility.Vector3dVector(filtered_pts)
        sel.paint_uniform_color([0.1, 0.9, 0.1])
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"雷达前方{VIEW_FRONT_Y}m原始点云(灰) + 直通框选(绿)", width=1000, height=700)
        vis.add_geometry(raw)
        vis.add_geometry(sel)
        vis.run()
        vis.destroy_window()
    
    pts = filtered_pts  # 使用已经过滤的数据

    # 记录直通滤波耗时
    passthrough_time = time.time() - start_time
    start_time = time.time()

    # ── 2. 法向量滤波 ────────────────────────────────────────────────────────
    # 优化：仅在必要时进行法向量估计
    _pcd = o3d.geometry.PointCloud()
    _pcd.points = o3d.utility.Vector3dVector(pts)
    _pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=NORMAL_KNN))
    nx = np.abs(np.asarray(_pcd.normals)[:, 0])
    side_pts = pts[nx > np.cos(np.radians(NORMAL_ANGLE_DEG))]
    if len(side_pts) < DBSCAN_MIN_POINTS:
        _dbg(f"计算失败：法向滤波后侧面点不足({len(side_pts)})，无法聚类")
        return None

    if view:
        bg0 = o3d.geometry.PointCloud()
        bg0.points = o3d.utility.Vector3dVector(pts)
        bg0.paint_uniform_color([0.5, 0.5, 0.5])
        sp = o3d.geometry.PointCloud()
        sp.points = o3d.utility.Vector3dVector(side_pts)
        sp.paint_uniform_color([0.2, 0.8, 0.8])
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="法向滤波：灰=原始  青=侧面点云", width=1000, height=700)
        vis.add_geometry(bg0)
        vis.add_geometry(sp)
        vis.run()
        vis.destroy_window()

    # 记录法向量滤波耗时
    normal_filter_time = time.time() - start_time
    start_time = time.time()

    # ── 3. DBSCAN 聚类 ───────────────────────────────────────────────────────
    sp_pcd = o3d.geometry.PointCloud()
    sp_pcd.points = o3d.utility.Vector3dVector(side_pts)
    
    # 使用更高效的DBSCAN参数
    labels = np.array(sp_pcd.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS))
    n_clusters = labels.max() + 1

    # 添加早期退出：如果聚类数量过多，可能是参数不合适
    if n_clusters > 50:  # 设置合理的聚类上限
        _dbg(f"警告：聚类数量过多({n_clusters})，可能需要调整参数")
    
    if view:
        rng = np.random.default_rng(0)
        cluster_colors = rng.random((max(n_clusters, 1), 3))
        colors = np.full((len(side_pts), 3), 0.3)
        for k in range(n_clusters):
            colors[labels == k] = cluster_colors[k]
        sp_pcd.colors = o3d.utility.Vector3dVector(colors)
        # 打印各簇颜色 + 统计，便于在窗口里对照颜色定位
        _dbg(f"聚类结果 {n_clusters} 个簇（噪点={int((labels == -1).sum())}）：")
        for k in range(n_clusters):
            kp = side_pts[labels == k]
            r, g, b = cluster_colors[k]
            hex_c = f'#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}'
            _dbg(f"  k={k:2d} 颜色={hex_c} RGB=({r:.2f},{g:.2f},{b:.2f}) "
                 f"点数={len(kp):5d} cx={kp[:,0].mean():+.3f} cz={kp[:,2].mean():+.3f} "
                 f"zspan={float(kp[:,2].max()-kp[:,2].min()):.2f}")
        bg1 = o3d.geometry.PointCloud()
        bg1.points = o3d.utility.Vector3dVector(pts)
        bg1.paint_uniform_color([0.5, 0.5, 0.5])
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"聚类结果（{n_clusters} 个簇，噪点灰色）", width=1000, height=700)
        vis.add_geometry(bg1)
        vis.add_geometry(sp_pcd)
        vis.run()
        vis.destroy_window()

    # 记录聚类耗时
    clustering_time = time.time() - start_time
    start_time = time.time()

    # ── 4. 按 x 重心选左右两簇（各侧取最靠近原点的簇）──────────────────────
    # 过滤规则：
    #   - 点数不足 MIN_CLUSTER_PTS → 噪点
    #   - z 跨度不足 MIN_CLUSTER_ZSPAN_M → 顶部薄片(灯/横梁/手柄)，过滤
    #   - z 重心 > MAX_CLUSTER_CZ_M → 车厢顶部凸起结构(管线/灯具)，过滤
    #     （Z 锁定模式下范围已收窄到当前行，关闭此绝对上限避免误删高处当前行）
    left_candidates, right_candidates = [], []
    for k in range(n_clusters):
        kpts = side_pts[labels == k]
        if len(kpts) < MIN_CLUSTER_PTS:
            continue
        zspan = float(kpts[:, 2].max() - kpts[:, 2].min())
        if zspan < MIN_CLUSTER_ZSPAN_M:
            _dbg(f"簇 k={k} 被过滤（zspan={zspan*100:.1f}cm < {MIN_CLUSTER_ZSPAN_M*100:.0f}cm，疑似顶部薄片）")
            continue
        cz = float(kpts[:, 2].mean())
        if not _z_locked and cz > MAX_CLUSTER_CZ_M:
            _dbg(f"簇 k={k} 被过滤（cz={cz:.2f}m > {MAX_CLUSTER_CZ_M:.1f}m，疑似车厢顶部凸起）")
            continue
        cx = float(kpts[:, 0].mean())
        (left_candidates if cx < 0 else right_candidates).append((cx, k))

    if not left_candidates or not right_candidates:
        _dbg(f"计算失败：原点两侧未找到点数≥{MIN_CLUSTER_PTS}的聚类，请检查 DBSCAN 参数或点云坐标")
        return None

    # 候选按"内侧→外侧"排序：左侧 cx 从大(内)到小(外)，右侧 cx 从小(内)到大(外)
    left_sorted  = [k for _, k in sorted(left_candidates,  key=lambda t: -t[0])]
    right_sorted = [k for _, k in sorted(right_candidates, key=lambda t:  t[0])]

    # 业务约束：前一抓永远位于左侧，所以左侧始终取最内侧簇，
    # 只允许右侧按“内→外”重选：左内-右内，或左内-右外。
    left_k = left_sorted[0]
    lc = side_pts[labels == left_k]
    lx = lc[:, 0]
    x_left_face = float(np.median(lx))
    selected = None
    for ri, right_k_ in enumerate(right_sorted):
        rc_ = side_pts[labels == right_k_]
        rx_ = rc_[:, 0]
        x_right_ = float(np.median(rx_))
        gap_ = int((x_right_ - x_left_face) * 1000)
        z_diff_ = abs(float(lc[:, 2].mean()) - float(rc_[:, 2].mean()))
        has_wall_ = (len(lx) >= WALL_PTS) or (len(rx_) >= WALL_PTS)
        width_ok_ = gap_ >= MIN_VALID_WIDTH_MM
        z_ok_ = _z_locked or has_wall_ or (z_diff_ <= MAX_Z_DIFF_M)
        _dbg(f"候选左内 k={left_k}({len(lx)}点) 右#{ri} k={right_k_}({len(rx_)}点)  "
             f"宽度={gap_}mm z重心差={z_diff_*1000:.0f}mm 含墙={has_wall_}")
        if width_ok_ and z_ok_:
            selected = (right_k_, rc_, x_right_, gap_, z_diff_)
            break

    if selected is None:
        _dbg("计算失败：左内侧簇与所有右侧候选均不满足宽度/高度条件")
        return None

    right_k, rc, x_right_face, gap_mm, z_diff = selected
    _dbg(f"最终选中左内簇 k={left_k}({len(lc)}点) 右簇 k={right_k}({len(rc)}点)  "
         f"宽度={gap_mm}mm z重心差={z_diff*1000:.0f}mm")

    clusters = {
        'left':  side_pts[labels == left_k],
        'right': side_pts[labels == right_k],
    }

    # 记录簇选择耗时
    selection_time = time.time() - start_time

    # ── 4. 宽度 ──────────────────────────────────────────────────────────────
    total_calc_time = time.time() - total_start_time
    
    # 只有在调试模式下才输出时间统计
    if DEBUG:
        _dbg(f"最终测量宽度：{gap_mm} mm")
        _dbg(f"总耗时：{total_calc_time:.3f}s")
        _dbg(f"  - 预处理：{preprocessing_time:.3f}s")
        _dbg(f"  - 直通滤波：{passthrough_time:.3f}s") 
        _dbg(f"  - 法向量滤波：{normal_filter_time:.3f}s")
        _dbg(f"  - DBSCAN聚类：{clustering_time:.3f}s")
        _dbg(f"  - 簇选择：{selection_time:.3f}s")
        _dbg(f"  - 其他处理：{total_calc_time - (preprocessing_time + passthrough_time + normal_filter_time + clustering_time + selection_time):.3f}s")

    if view:
        bg = o3d.geometry.PointCloud()
        bg.points = o3d.utility.Vector3dVector(pts)
        bg.paint_uniform_color([0.5, 0.5, 0.5])

        lf = o3d.geometry.PointCloud()
        lf.points = o3d.utility.Vector3dVector(clusters['left'])
        lf.paint_uniform_color([1, 0.5, 0])

        rf = o3d.geometry.PointCloud()
        rf.points = o3d.utility.Vector3dVector(clusters['right'])
        rf.paint_uniform_color([0, 1, 0.3])

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="左簇(橙) / 右簇(绿) + 原始(灰)", width=1000, height=700)
        for g in [bg, lf, rf]:
            vis.add_geometry(g)
        vis.run()
        vis.destroy_window()

        left_plane_mesh  = show_plane([1, 0, 0, -x_left_face],  [1, 0.5, 0])
        right_plane_mesh = show_plane([1, 0, 0, -x_right_face], [0, 1, 0.3])

        bg2 = o3d.geometry.PointCloud()
        bg2.points = o3d.utility.Vector3dVector(pts)
        bg2.paint_uniform_color([0.5, 0.5, 0.5])
        vis = o3d.visualization.Visualizer()
        win_name = (f"拟合平面  面A(橙)={x_left_face:.3f}m  "
                    f"面B(绿)={x_right_face:.3f}m  宽={gap_mm}mm")
        vis.create_window(window_name=win_name, width=1000, height=700)
        for g in [bg2, left_plane_mesh, right_plane_mesh]:
            vis.add_geometry(g)
        vis.run()
        vis.destroy_window()

    return gap_mm


def check_stacking(length, pc1, pc2, tolerance=50, yaw_offset_deg=0.0,
                   rel_top_h=None, box_h=None):
    """测量堆叠宽度，返回 measured_mm（计算成功）或 None（计算失败/报错）。
    length、tolerance 参数保留以兼容旧调用，现已不参与判定——
    对外状态只表示是否顺利算出宽度，不再判定通过/不通过。
    yaw_offset_deg: 拍照位相对正对的偏航角(度)，用于点云偏航补偿。
    rel_top_h / box_h: 当前抓顶面距地板高度和箱子竖向高度(米)，用于把测宽锁定到当前行。
    """
    return _compute_width(pc1, pc2, yaw_offset_deg=yaw_offset_deg,
                          rel_top_h=rel_top_h, box_h=box_h)


def _default_save_dir():
    # 优先用 COLCON_PREFIX_PATH（/ws/install）推断 ws 根
    prefix = os.environ.get('COLCON_PREFIX_PATH', '')
    if prefix:
        ws_root = os.path.dirname(prefix.split(':')[0])
        if os.path.isdir(ws_root):
            return os.path.join(ws_root, 'log', 'robot_process', 'pcd_logs')
    # 从源码路径找 src/ 目录，定位 ws 根（.../ws/src/pkg/pkg/stacking_detection/）
    parts = os.path.abspath(__file__).split(os.sep)
    if 'src' in parts:
        ws_root = os.sep.join(parts[:parts.index('src')])
        return os.path.join(ws_root, 'log', 'robot_process', 'pcd_logs')
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pcd_logs')

_DEFAULT_SAVE_DIR = _default_save_dir()

def save_point_clouds(pc1, pc2, save_dir=_DEFAULT_SAVE_DIR):
    """将双雷达点云以时间戳命名保存为 PCD 文件。"""
    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(save_dir, f'merged_{ts}.pcd')
    o3d.io.write_point_cloud(path, pc1 + pc2, write_ascii=False)
    _dbg(f'点云已保存至：{path}')


def process_point_cloud(length, topic1='/lidar_points1', topic2='/lidar_points2'):
    """向后兼容接口：内部采集点云后调用 _compute_width。"""
    pc1, pc2 = collect_dual_lidar_once(topic1, topic2, frames=3)
    return _compute_width(pc1, pc2)


# ─── 离线测试入口 ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    import glob

    DEBUG = True   # 离线测试：打开调试打印

    # ↓ 只填文件名即可，目录自动使用 _DEFAULT_SAVE_DIR；留空则自动选取最新文件
    _FILENAME = '0626/merged_20260626_134759.pcd'

    args = sys.argv[1:]
    if _FILENAME:
        pcd_path = os.path.join(_DEFAULT_SAVE_DIR, _FILENAME)
    elif args:
        pcd_path = args[0]
    else:
        # 不带参数：自动加载默认目录中最新 merged 文件
        files = sorted(glob.glob(os.path.join(_DEFAULT_SAVE_DIR, 'merged_*.pcd')))
        if not files:
            print(f'用法: python stacking_detection_node.py <merged.pcd> [理论宽度mm]')
            sys.exit(1)
        pcd_path = files[-1]
        print(f'自动选取最新文件：{pcd_path}')

    if not os.path.isfile(pcd_path):
        print(f'[ERROR] 文件不存在: {pcd_path}')
        sys.exit(1)

    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f'加载点云：{pcd_path}  点数={len(pcd.points)}')

    empty = o3d.geometry.PointCloud()
    # 当前行 Z 锁定调试：rel_top_h=当前行顶面距地板高度(米)，box_h=箱竖向高度(米)
    # 不需要锁定时把这两个参数删掉即可
    measured = _compute_width(pcd, empty, yaw_offset_deg=4,
                              rel_top_h=0.285*8, box_h=0.285)   # view 跟随顶部 VIEW 开关
    print(f'\n测量宽度: {measured} mm')

    if len(args) >= 2:
        length = int(args[1])
        label = '通过' if abs(measured - length) <= 50 else '不通过'
        print(f'理论宽度: {length} mm  差值: {measured - length:+.1f} mm  结果: {label}')
