import copy
import os
import open3d as o3d
import numpy as np
import cv2
import math
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

# 调试模式开关：True 才向终端打印运行日志（测量宽度、保存路径等）
# 离线测试入口 __main__ 内会自动置 True
DEBUG = False
_VISUALIZATION_BACKEND_READY = False

def _dbg(msg):
    if DEBUG:
        print(msg)


def _prepare_visualization_backend():
    """仅在实际显示窗口时，为当前 Python 进程选择可用的 XWayland 后端。"""
    global _VISUALIZATION_BACKEND_READY
    if _VISUALIZATION_BACKEND_READY:
        return
    if (os.environ.get('DISPLAY') and
            os.environ.get('XDG_SESSION_TYPE', '').lower() == 'wayland'):
        # Open3D 0.19 Legacy Visualizer 在本机 Wayland/EGL 下无法初始化 GLEW。
        # 这里只修改当前 Python 进程；不会修改系统、桌面会话或其他程序。
        os.environ.pop('WAYLAND_DISPLAY', None)
        os.environ['XDG_SESSION_TYPE'] = 'x11'
        os.environ['GDK_BACKEND'] = 'x11'
        _dbg('Open3D可视化：当前进程使用 XWayland 后端')
    _VISUALIZATION_BACKEND_READY = True


def _show_geometries(window_name, geometries, width=800, height=600,
                     left=500, top=200):
    """按 corner_detection 的窗口方式显示；创建失败时安全跳过。"""
    _prepare_visualization_backend()
    vis = o3d.visualization.Visualizer()
    created = False
    try:
        kwargs = {
            'window_name': window_name,
            'width': width,
            'height': height,
        }
        if left is not None:
            kwargs['left'] = left
        if top is not None:
            kwargs['top'] = top
        created = bool(vis.create_window(**kwargs))
        if not created:
            _dbg(f"可视化窗口创建失败，已安全跳过：{window_name}")
            return False
        for index, geometry in enumerate(geometries):
            vis.add_geometry(geometry, reset_bounding_box=(index == 0))
        vis.run()
        return True
    except Exception as exc:
        _dbg(f"可视化异常，已安全跳过 {window_name}：{type(exc).__name__}: {exc}")
        return False
    finally:
        if created:
            vis.destroy_window()


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
MIN_CLUSTER_PTS = 10             # 簇最小点数：15 能保住更稀疏的第一层箱面，又能过滤孤立噪点
MIN_CLUSTER_ZSPAN_M = 0.02       # 簇最小 z 跨度(米)：0.10 过滤顶部薄片(<0.08)，保住稀疏第一层(0.1-0.4)
MAX_CLUSTER_CZ_M = 1.0           # 簇 z 重心上限(米)：> 此值视为车厢顶部凸起结构(管线/灯)，丢弃

# ── 候选面之间的箱体占用检查 ──
PAIR_OCCUPANCY_FRONT_DEPTH_M = 0.20  # 只检查当前面靠雷达的200mm，避免后排箱/车壁填充空区
PAIR_OCCUPANCY_SIDE_MARGIN_M = 0.05  # 两候选面内缩50mm，不把边界侧面自身当作箱体证据
PAIR_OCCUPANCY_BIN_M = 0.10          # 沿宽度方向每100mm统计一个占用 bin
PAIR_OCCUPANCY_MIN_PTS_PER_BIN = 10 # bin 内至少此点数才认为有箱体表面
PAIR_OCCUPANCY_MIN_COVERAGE = 0.60  # 覆盖率达到60%视为区间内有箱；未达到则是下一抓候选空缺口
PAIR_OCCUPANCY_MIN_REL_DENSITY = 0.25 # 密度低于本帧最完整箱体区25%时，即使有后方漏点也仍视为空缺口

# ── 当前面 Y 锁定（深度方向，只保留最靠雷达的当前面箱，滤掉后排箱）──
FRONT_Y_BIN = 0.05               # Y 直方图 bin 宽(米)
FRONT_Y_MIN_PTS = 100            # Y bin 视为"有箱"的最小点数
FRONT_Y_DEPTH = 0.45             # 当前面保留深度(米)：从前沿往后取此窗口=最小箱长，
                                 #   保证落在前排箱内、不碰后排箱（箱长 450~530mm）

# ── 当前箱体倾斜检测（自适应垛面横向 U-Z 投影，提取成对斜边）──
TILT_PASS_X = (-1.20, 1.20)      # 排除两侧车壁，只检查实际码垛宽度范围
TILT_FRONT_DEPTHS = (0.18, 0.24)  # 分别检测前沿180/240mm；不可合并，否则后排点会填平斜边
TILT_FRAME_MIN_PTS = 100         # 估计局部 U 轴所需的当前层最少点数
TILT_YAW_GRID_M = 0.01           # XY 俯视投影分辨率，用于自动估计垛面横向
TILT_YAW_MAX_DEG = 25.0          # 垛面横向相对雷达 X 轴的最大合理偏航
TILT_YAW_MIN_LINE_M = 0.50       # XY 俯视图中参与偏航投票的最小线长
TILT_YAW_ANGLE_BIN_DEG = 2.0     # 横向线角度投票 bin，取线长加权的主峰
TILT_FRONT_BIN_M = 0.05          # 局部深度方向定位当前面的直方图 bin
TILT_FRONT_BIN_MIN_PTS = 30      # 局部深度 bin 认定为实际箱面的最少点数
TILT_FRONT_MAX_PEAKS = 8         # 最多检查的局部深度主峰数，防止计算量无限增长
TILT_Z_MARGIN_BOTTOM = 0.10      # 当前箱底以下额外保留范围(米)，防止倾斜箱下沉后被裁掉
TILT_Z_MARGIN_TOP = 0.12         # 当前箱顶以上额外保留范围(米)，降低实测箱顶毫米级波动对 Hough 的影响
TILT_GRID_M = 0.005              # X-Z 投影栅格分辨率：5mm
TILT_MIN_LINE_M = 0.20           # 有效斜边最小长度：200mm
TILT_LINE_ANGLE_MIN_DEG = 12.0   # 排除接近水平的正常箱边
TILT_LINE_ANGLE_MAX_DEG = 78.0   # 排除接近竖直的正常箱边
TILT_RESULT_MIN_DEG = 15.0       # 两边综合倾角下限；过滤正常箱体边缘的小波动
TILT_EACH_LINE_MIN_DEG = 15.0    # 水平边和竖直边都必须明显偏转，避免一条正常边搭配噪声线误报
TILT_PAIR_ANGLE_MIN_DEG = 75.0   # 两条箱边应近似正交
TILT_PAIR_ANGLE_MAX_DEG = 105.0
TILT_PAIR_ENDPOINT_MAX_M = 0.08  # 两条斜边连接端点最大距离：80mm
TILT_HOUGH_THRESHOLD = 20
TILT_HOUGH_MAX_GAP_M = 0.06      # 雷达扫描线存在空隙，允许直线跨越60mm断点
TILT_LINE_RADIUS_M = 0.008       # 可视化斜线圆柱半径
TILT_TEXT_SCALE = 0.0025         # Open3D 3D文字缩放（约32mm字高，点云窗口内可读）

# ── 当前行 Z 自适应锁定（避免相邻层干扰）──
LAYER_SEARCH_TOL_RATIO = 0.4     # 搜箱顶容差 = box_h × 此比例(±)：理论与实际的最大偏差(40%箱高)
LAYER_TOP_NORMAL_DEG = 15        # 水平面识别角度阈值(度)：|nz| > cos(此值) 视为箱顶水平面
LAYER_TOP_MIN_PTS = 10           # 当前层箱顶水平面最少点数（高层点云较稀疏）
LAYER_Z_HIST_BIN = 0.01          # z 直方图 bin 宽(米)：1cm，对应实际放置精度量级
LAYER_PEAK_MIN_PTS = 3           # 箱顶候选高度 bin 的最少水平面点数
LAYER_PEAK_MIN_RATIO = 0.10      # 候选 bin 点数至少达到当前最强峰的此比例
LAYER_Z_MARGIN_TOP = 0.03        # 锁定 z 范围上沿留余量(米)：略高于本层顶面，防漏点
LAYER_Z_MARGIN_BOTTOM = 0.03     # 锁定 z 范围下沿留余量(米)：略高于本层底面，避下层顶面
LAYER_GUARD_START = 1            # 从第1层起启用理论层底保护，所有层采用一致的截取规则
LAYER_TOP_MAX_ERROR_RATIO = 0.30 # 实测顶偏离理论值超过30%箱高时视为相邻层误峰
FLOOR_MIN_PTS = 50              # 地板检测取最低点数：取 z 最小的此数个点的中位数作地板，
                                #   地板被货物遮挡仅剩零星点时仍稳定，又抗个别雷达噪点
FLOOR_Z_DEFAULT = -1.13         # 首层地板不可见时的标定回退值（雷达坐标系，米）
FLOOR_Z_MAX_DEVIATION = 0.15    # 自动标定相对回退值的最大允许偏差，超出视为噪点/遮挡
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
        cluster_views = []
        for label in valid_labels:
            indices = np.where(labels == label)[0]
            cluster_points = np.asarray(pcd.points)[indices]
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
            cluster_pcd.paint_uniform_color(np.random.rand(3))
            cluster_views.append(cluster_pcd)
        _show_geometries(
            "Front clustered", cluster_views, width=800, height=600)
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
        _show_geometries(
            "front d filtered", [pcd_show], width=800, height=600,
            left=500, top=200)
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
    _show_geometries("Stitched point cloud", [combined_pcd])

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

    # 层数升高后，真正箱顶可见点会逐渐变少，而搜索窗口内下层箱顶/侧面会越来越多。
    # 若仍让一个误选的低峰同时下拉 z_min/z_max，串入下层的风险会随层数增加。
    # 所有层统一采用两道约束，避免低层与高层的裁剪规则在第5层突然切换：
    #   1. 实测顶面偏差过大时回退理论顶面；
    #   2. z 下沿不得低于本层理论底面+余量，彻底阻止下层顶部进入候选面。
    layer_index = max(1, int(round(float(rel_top_h) / float(box_h))))
    if layer_index >= LAYER_GUARD_START:
        max_top_error = box_h * LAYER_TOP_MAX_ERROR_RATIO
        top_error = actual_top_z - theo_top_z
        if abs(top_error) > max_top_error:
            _dbg(
                f"当前层Z保护：第{layer_index}层实测顶偏差 {top_error * 1000:+.0f}mm "
                f"超过阈值 ±{max_top_error * 1000:.0f}mm，判为相邻层峰，回退理论顶 "
                f"{theo_top_z:.3f}m")
            actual_top_z = theo_top_z

        measured_z_min = actual_top_z - box_h + LAYER_Z_MARGIN_BOTTOM
        theoretical_z_min = theo_top_z - box_h + LAYER_Z_MARGIN_BOTTOM
        z_min = max(measured_z_min, theoretical_z_min)
        # 实测顶偏低时也不下拉上沿，避免把本层上半部一起裁掉；偏高时仍允许向上扩展。
        z_max = max(actual_top_z, theo_top_z) + LAYER_Z_MARGIN_TOP
        _dbg(
            f"当前层Z保护：第{layer_index}层理论底硬边界={theoretical_z_min:.3f}m，"
            f"最终范围=[{z_min:.3f}, {z_max:.3f}]m")
    else:
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


def _normalize_xz_line_angle(angle_deg):
    """把无方向直线角度归一化到 [-90°, 90°)。"""
    return (float(angle_deg) + 90.0) % 180.0 - 90.0


def _line_endpoint_gap_px(line_a, line_b):
    """返回两条 Hough 线段四组端点之间的最小像素距离。"""
    endpoints_a = np.asarray(line_a, dtype=float).reshape(2, 2)
    endpoints_b = np.asarray(line_b, dtype=float).reshape(2, 2)
    return min(
        float(np.linalg.norm(pa - pb))
        for pa in endpoints_a
        for pb in endpoints_b
    )


def _estimate_tilt_u_axis(pts):
    """在全垛 XY 俯视图中用长直线角度投票估计左右横向 U。"""
    mask = (
        (pts[:, 0] >= TILT_PASS_X[0]) & (pts[:, 0] < TILT_PASS_X[1]) &
        (pts[:, 1] >= PASS_Y[0]) & (pts[:, 1] <= PASS_Y[1]) &
        (pts[:, 2] >= PASS_Z[0]) & (pts[:, 2] <= PASS_Z[1])
    )
    xy = pts[mask, :2]
    if len(xy) < TILT_FRAME_MIN_PTS:
        return None

    nx = int(math.ceil((TILT_PASS_X[1] - TILT_PASS_X[0]) / TILT_YAW_GRID_M))
    ny = int(math.ceil((PASS_Y[1] - PASS_Y[0]) / TILT_YAW_GRID_M))
    image = np.zeros((ny, nx), dtype=np.uint8)
    ix = np.clip(
        ((xy[:, 0] - TILT_PASS_X[0]) / TILT_YAW_GRID_M).astype(int),
        0, nx - 1)
    iy = np.clip(
        ((xy[:, 1] - PASS_Y[0]) / TILT_YAW_GRID_M).astype(int),
        0, ny - 1)
    image[iy, ix] = 255
    image = cv2.morphologyEx(
        image, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 360.0, threshold=30,
        minLineLength=max(1, int(math.ceil(
            TILT_YAW_MIN_LINE_M / TILT_YAW_GRID_M))),
        maxLineGap=max(1, int(round(0.10 / TILT_YAW_GRID_M))))
    if lines is None:
        return None

    candidates = []
    for raw in lines[:, 0]:
        dx = int(raw[2]) - int(raw[0])
        dy = int(raw[3]) - int(raw[1])
        length_m = math.hypot(dx, dy) * TILT_YAW_GRID_M
        angle_deg = _normalize_xz_line_angle(
            math.degrees(math.atan2(dy, dx)))
        if (length_m >= TILT_YAW_MIN_LINE_M and
                abs(angle_deg) <= TILT_YAW_MAX_DEG):
            candidates.append((angle_deg, length_m))
    if not candidates:
        return None

    angle_edges = np.arange(
        -TILT_YAW_MAX_DEG,
        TILT_YAW_MAX_DEG + TILT_YAW_ANGLE_BIN_DEG,
        TILT_YAW_ANGLE_BIN_DEG)
    weights, _ = np.histogram(
        [item[0] for item in candidates], bins=angle_edges,
        weights=[item[1] for item in candidates])
    peak_index = int(np.argmax(weights))
    peak_candidates = [
        item for item in candidates
        if angle_edges[peak_index] <= item[0] < angle_edges[peak_index + 1]
    ]
    weight_sum = sum(item[1] for item in peak_candidates)
    if weight_sum <= 1e-9:
        return None
    yaw_deg = sum(
        angle_deg * length_m for angle_deg, length_m in peak_candidates
    ) / weight_sum
    yaw_rad = math.radians(yaw_deg)
    return np.array([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=float)


def _fit_tilt_projection_frame(pts, z_min, z_max):
    """从当前帧自动估计 U-V-Z 局部坐标和当前层深度主峰。"""
    roi_mask = (
        (pts[:, 0] >= TILT_PASS_X[0]) & (pts[:, 0] < TILT_PASS_X[1]) &
        (pts[:, 1] >= PASS_Y[0]) & (pts[:, 1] <= PASS_Y[1]) &
        (pts[:, 2] >= z_min) & (pts[:, 2] < z_max)
    )
    roi = pts[roi_mask]
    if len(roi) < TILT_FRAME_MIN_PTS:
        return None

    u_axis = _estimate_tilt_u_axis(pts)
    if u_axis is None:
        return None
    origin_xy = np.zeros(2, dtype=float)
    v_axis = np.array([-u_axis[1], u_axis[0]], dtype=float)
    # 当前车厢点云位于雷达 -Y 方向，V 固定指向雷达一侧。
    if v_axis[1] < 0.0:
        v_axis *= -1.0

    relative_xy = roi[:, :2] - origin_xy
    u_values = relative_xy @ u_axis
    v_values = relative_xy @ v_axis
    all_mask = (
        (pts[:, 0] >= TILT_PASS_X[0]) & (pts[:, 0] < TILT_PASS_X[1]) &
        (pts[:, 1] >= PASS_Y[0]) & (pts[:, 1] <= PASS_Y[1]) &
        (pts[:, 2] >= PASS_Z[0]) & (pts[:, 2] <= PASS_Z[1])
    )
    # U 窗口以全垛中心为基准，避免某一层只有单侧点云时窗口偏移。
    u_center = float(np.median(
        (pts[all_mask, :2] - origin_xy) @ u_axis))
    u_min = u_center + TILT_PASS_X[0]
    u_max = u_center + TILT_PASS_X[1]
    within_u = (u_values >= u_min) & (u_values < u_max)
    if int(np.count_nonzero(within_u)) < TILT_FRAME_MIN_PTS:
        return None

    front_values = v_values[within_u]
    value_min = float(np.min(front_values))
    value_max = float(np.max(front_values))
    edges = np.arange(
        value_min, value_max + TILT_FRONT_BIN_M * 1.01,
        TILT_FRONT_BIN_M)
    if len(edges) < 2:
        edges = np.array([value_min, value_min + TILT_FRONT_BIN_M])
    hist, edges = np.histogram(front_values, bins=edges)
    peak_indices = []
    for index, count in enumerate(hist):
        left = hist[index - 1] if index > 0 else -1
        right = hist[index + 1] if index + 1 < len(hist) else -1
        if (count >= TILT_FRONT_BIN_MIN_PTS and
                count >= left and count >= right):
            peak_indices.append(index)
    if not peak_indices:
        return None
    # 先检查支持点多的真实箱面；同时保留多个深度，不假设倾倒箱最靠近雷达。
    peak_indices = sorted(
        peak_indices, key=lambda index: int(hist[index]), reverse=True
    )[:TILT_FRONT_MAX_PEAKS]
    # 在主峰靠雷达侧再留一个 bin，防止直方图分箱恰好切断箱边。
    v_fronts = [
        float(edges[index + 1] + TILT_FRONT_BIN_M)
        for index in peak_indices
    ]

    return {
        'origin_xy': origin_xy,
        'u_axis': u_axis,
        'v_axis': v_axis,
        'u_range': (float(u_min), float(u_max)),
        'v_fronts': v_fronts,
        'yaw_deg': _normalize_xz_line_angle(
            math.degrees(math.atan2(u_axis[1], u_axis[0]))),
        'roi': roi,
    }


def _supporting_line_v(front_pts, frame, p0_uz, p1_uz):
    """从线段附近点云估计绘制线所在的局部 V 坐标。"""
    relative_xy = front_pts[:, :2] - frame['origin_xy']
    uz = np.column_stack((relative_xy @ frame['u_axis'], front_pts[:, 2]))
    p0 = np.asarray(p0_uz, dtype=float)
    p1 = np.asarray(p1_uz, dtype=float)
    vec = p1 - p0
    len_sq = float(np.dot(vec, vec))
    if len_sq <= 1e-12:
        all_v = (front_pts[:, :2] - frame['origin_xy']) @ frame['v_axis']
        return float(np.percentile(all_v, 80))
    t = np.clip(((uz - p0) @ vec) / len_sq, 0.0, 1.0)
    nearest = p0 + t[:, None] * vec
    distance = np.linalg.norm(uz - nearest, axis=1)
    support = front_pts[distance <= 0.02]
    source = support if len(support) >= 10 else front_pts
    source_v = (source[:, :2] - frame['origin_xy']) @ frame['v_axis']
    return float(np.percentile(source_v, 80))


def _make_line_cylinder(p0, p1, color):
    """创建连接两个三维点的粗线圆柱，便于在点云窗口中看清斜边。"""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    direction = p1 - p0
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        return None
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=TILT_LINE_RADIUS_M, height=length, resolution=12)
    unit = direction / length
    z_axis = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z_axis, unit)
    cross_norm = float(np.linalg.norm(cross))
    dot = float(np.clip(np.dot(z_axis, unit), -1.0, 1.0))
    if cross_norm > 1e-9:
        rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(
            cross / cross_norm * math.acos(dot))
        mesh.rotate(rotation, center=(0.0, 0.0, 0.0))
    elif dot < 0.0:
        mesh.rotate(
            o3d.geometry.get_rotation_matrix_from_xyz((math.pi, 0.0, 0.0)),
            center=(0.0, 0.0, 0.0))
    mesh.translate((p0 + p1) * 0.5)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def _make_tilt_text(text, target, color):
    """创建朝向雷达的 ASCII 三维文字；旧版 Open3D 不支持时由窗口标题兜底。"""
    try:
        text_mesh = o3d.t.geometry.TriangleMesh.create_text(
            text, depth=1.0).to_legacy()
        text_mesh.scale(TILT_TEXT_SCALE, center=(0.0, 0.0, 0.0))
        # 原文字位于 X-Y 平面，旋转到点云的 X-Z 正视平面。
        text_mesh.rotate(
            o3d.geometry.get_rotation_matrix_from_xyz((math.pi / 2.0, 0.0, 0.0)),
            center=(0.0, 0.0, 0.0))
        # 文字正面法向为 -Y；补一份反向三角形，使雷达侧观察时也可见。
        triangles = np.asarray(text_mesh.triangles)
        if len(triangles):
            text_mesh.triangles = o3d.utility.Vector3iVector(
                np.vstack((triangles, triangles[:, ::-1])))
        center = text_mesh.get_axis_aligned_bounding_box().get_center()
        text_mesh.translate(np.asarray(target, dtype=float) - center)
        text_mesh.paint_uniform_color(color)
        text_mesh.compute_vertex_normals()
        return text_mesh
    except Exception as exc:
        _dbg(f"倾斜检测文字创建失败，窗口标题仍保留标注：{type(exc).__name__}: {exc}")
        return None


def _build_tilt_2d_image(binary_image, edge_image, result, z_min, z_max):
    """生成带坐标、两条斜边及测量文字的 U-Z 正视检测结果图。"""
    scale = 3
    plot = np.zeros((*binary_image.shape, 3), dtype=np.uint8)
    plot[binary_image > 0] = [75, 75, 75]
    plot[edge_image > 0] = [225, 225, 225]
    # 投影数组的行号随 z 增大；显示时翻转，使正 Z 朝上。
    plot = cv2.flip(plot, 0)
    plot = cv2.resize(
        plot, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    margin_left, margin_right = 90, 25
    margin_top, margin_bottom = 105, 65
    canvas = np.full(
        (plot.shape[0] + margin_top + margin_bottom,
         plot.shape[1] + margin_left + margin_right, 3),
        20, dtype=np.uint8)
    canvas[margin_top:margin_top + plot.shape[0],
           margin_left:margin_left + plot.shape[1]] = plot
    line_colors = ((30, 45, 255), (0, 220, 255))  # OpenCV BGR：红、黄

    def to_canvas(pixel_x, pixel_z):
        return (
            int(round(margin_left + float(pixel_x) * scale)),
            int(round(margin_top +
                      (binary_image.shape[0] - 1 - float(pixel_z)) * scale)),
        )

    descriptions = []
    for index, (line, color) in enumerate(zip(result['lines'], line_colors), start=1):
        raw = line['pixels']
        p0 = to_canvas(raw[0], raw[1])
        p1 = to_canvas(raw[2], raw[3])
        cv2.line(canvas, p0, p1, color, thickness=5, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p0, 7, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p1, 7, color, thickness=-1, lineType=cv2.LINE_AA)
        length_mm = line['length_m'] * 1000.0
        descriptions.append(
            f"L{index}: {line['angle_deg']:+.1f} deg, {length_mm:.0f} mm")
        midpoint = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
        label_y = midpoint[1] - 12 if index == 1 else midpoint[1] + 25
        label = f"L{index} {line['angle_deg']:+.1f}deg {length_mm:.0f}mm"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_x = max(margin_left, min(
            midpoint[0] - text_width // 2,
            canvas.shape[1] - margin_right - text_width))
        cv2.rectangle(
            canvas,
            (label_x - 4, label_y - text_height - 4),
            (label_x + text_width + 4, label_y + 5),
            (10, 10, 10), thickness=-1)
        cv2.putText(
            canvas, label, (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    title = (f"STATUS=2 | Tilted box: {result['tilt_deg']:.1f} deg | "
             f"auto yaw: {result['projection_yaw_deg']:+.1f} deg | "
             f"V depth: {result['front_depth_m'] * 1000:.0f} mm")
    cv2.putText(
        canvas, title, (margin_left, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
    for index, (description, color) in enumerate(zip(descriptions, line_colors)):
        cv2.putText(
            canvas, description, (margin_left, 58 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)

    plot_left = margin_left
    plot_right = margin_left + plot.shape[1] - 1
    plot_top = margin_top
    plot_bottom = margin_top + plot.shape[0] - 1
    cv2.rectangle(
        canvas, (plot_left, plot_top), (plot_right, plot_bottom),
        (150, 150, 150), thickness=1)

    # U轴刻度按照局部垛面米制坐标绘制。
    u_min, u_max = result['u_range']
    for u_value in np.linspace(u_min, u_max, 7):
        ratio = (u_value - u_min) / (u_max - u_min)
        px = int(round(plot_left + ratio * (plot.shape[1] - 1)))
        cv2.line(canvas, (px, plot_bottom), (px, plot_bottom + 6), (180, 180, 180), 1)
        text_value = f"{u_value:+.1f}"
        (width, _), _ = cv2.getTextSize(
            text_value, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(
            canvas, text_value, (px - width // 2, plot_bottom + 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (190, 190, 190), 1, cv2.LINE_AA)

    # Z轴刻度显示当前箱体截取范围。
    for z_value in np.linspace(z_min, z_max, 5):
        ratio = (z_value - z_min) / max(z_max - z_min, 1e-9)
        py = int(round(plot_bottom - ratio * (plot.shape[0] - 1)))
        cv2.line(canvas, (plot_left - 6, py), (plot_left, py), (180, 180, 180), 1)
        text_value = f"{z_value:+.2f}"
        (width, _), _ = cv2.getTextSize(
            text_value, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(
            canvas, text_value, (plot_left - width - 10, py + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (190, 190, 190), 1, cv2.LINE_AA)

    cv2.putText(
        canvas, "U along pallet (m)",
        ((plot_left + plot_right) // 2 - 25, canvas.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(
        canvas, "Z (m)", (12, plot_top - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def _show_tilt_2d_image(image):
    """显示2D倾斜结果；无桌面或窗口创建失败时安全跳过。"""
    if not os.environ.get('DISPLAY'):
        _dbg('2D倾斜可视化：未检测到 DISPLAY，已安全跳过窗口')
        return False
    _prepare_visualization_backend()
    window_name = '2D tilted-box result (U-Z)'
    created = False
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        created = True
        cv2.resizeWindow(
            window_name,
            min(1500, int(image.shape[1])),
            min(900, int(image.shape[0])))
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        return True
    except Exception as exc:
        _dbg(f"2D倾斜可视化异常，已安全跳过：{type(exc).__name__}: {exc}")
        return False
    finally:
        if created:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass


def _show_tilt_result(tilt_roi, front_pts, result):
    """显示当前箱点云，并把检测到的两条斜边及角度/长度画在窗口内。"""
    background = o3d.geometry.PointCloud()
    background.points = o3d.utility.Vector3dVector(tilt_roi)
    background.paint_uniform_color([0.45, 0.45, 0.45])
    foreground = o3d.geometry.PointCloud()
    foreground.points = o3d.utility.Vector3dVector(front_pts)
    foreground.paint_uniform_color([0.15, 0.75, 0.85])
    geometries = [background, foreground]
    line_colors = ([1.0, 0.15, 0.05], [1.0, 0.85, 0.05])
    descriptions = []
    max_line_y = max(
        max(line['p0_xyz'][1], line['p1_xyz'][1])
        for line in result['lines'])
    for index, (line, color) in enumerate(zip(result['lines'], line_colors), start=1):
        p0 = line['p0_xyz'].copy()
        p1 = line['p1_xyz'].copy()
        # 沿局部 V 轴略微靠近雷达，防止线被点云遮挡。
        p0[:2] += result['v_axis'] * 0.01
        p1[:2] += result['v_axis'] * 0.01
        cylinder = _make_line_cylinder(p0, p1, color)
        if cylinder is not None:
            geometries.append(cylinder)
        length_mm = line['length_m'] * 1000.0
        descriptions.append(
            f"L{index}={line['angle_deg']:+.1f}deg/{length_mm:.0f}mm")
        midpoint = (p0 + p1) * 0.5
        # 两个标签分居线段上下，避免重叠；文字本身也略靠雷达侧防止被点遮住。
        label_z_offset = 0.055 if index == 1 else -0.055
        label = _make_tilt_text(
            f"L{index} {line['angle_deg']:+.1f}deg {length_mm:.0f}mm",
            [midpoint[0], max_line_y + 0.025, midpoint[2] + label_z_offset],
            color)
        if label is not None:
            geometries.append(label)
    title = (
        f"STATUS=2 | 箱体倾斜 {result['tilt_deg']:.1f}deg | "
        f"自估yaw={result['projection_yaw_deg']:+.1f}deg | "
        + " | ".join(descriptions))
    _show_geometries(title, geometries, width=1100, height=760)


def _find_tilt_pair_at_depth(pts, frame, z_min, z_max,
                             front_depth, v_front):
    """在一个独立的局部V深度窗口内寻找最佳斜边对。"""
    relative_xy = pts[:, :2] - frame['origin_xy']
    u_values = relative_xy @ frame['u_axis']
    v_values = relative_xy @ frame['v_axis']
    u_min, u_max = frame['u_range']
    v_back = v_front - front_depth
    tilt_mask = (
        (u_values >= u_min) & (u_values < u_max) &
        (v_values >= v_back) & (v_values <= v_front) &
        (pts[:, 2] >= z_min) & (pts[:, 2] < z_max)
    )
    front_pts = pts[tilt_mask]
    if len(front_pts) < 100:
        return None

    nx = int(math.ceil((u_max - u_min) / TILT_GRID_M))
    nz = int(math.ceil((z_max - z_min) / TILT_GRID_M))
    image = np.zeros((nz, nx), dtype=np.uint8)
    front_u = u_values[tilt_mask]
    ix = np.clip(
        ((front_u - u_min) / TILT_GRID_M).astype(int),
        0, nx - 1)
    iz = np.clip(
        ((front_pts[:, 2] - z_min) / TILT_GRID_M).astype(int),
        0, nz - 1)
    image[iz, ix] = 255
    # 补齐雷达水平扫描线之间的小空隙，保留箱体尺寸级的真实轮廓。
    image = cv2.morphologyEx(
        image, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    image = cv2.dilate(image, np.ones((3, 3), dtype=np.uint8))
    edges = cv2.Canny(image, 50, 150)
    lines_raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360.0,
        threshold=TILT_HOUGH_THRESHOLD,
        minLineLength=max(1, int(math.ceil(TILT_MIN_LINE_M / TILT_GRID_M))),
        maxLineGap=max(1, int(round(TILT_HOUGH_MAX_GAP_M / TILT_GRID_M))),
    )
    if lines_raw is None:
        return None

    candidates = []
    for raw in lines_raw[:, 0]:
        dx = int(raw[2]) - int(raw[0])
        dz = int(raw[3]) - int(raw[1])
        length_m = math.hypot(dx, dz) * TILT_GRID_M
        angle_deg = _normalize_xz_line_angle(math.degrees(math.atan2(dz, dx)))
        if length_m < TILT_MIN_LINE_M:
            continue
        if not (TILT_LINE_ANGLE_MIN_DEG <= abs(angle_deg) <= TILT_LINE_ANGLE_MAX_DEG):
            continue
        candidates.append({
            'pixels': np.asarray(raw, dtype=int),
            'length_m': length_m,
            'angle_deg': angle_deg,
        })

    best_pair = None
    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1:]:
            first_deviation = min(
                abs(first['angle_deg']), 90.0 - abs(first['angle_deg']))
            second_deviation = min(
                abs(second['angle_deg']), 90.0 - abs(second['angle_deg']))
            if min(first_deviation, second_deviation) < TILT_EACH_LINE_MIN_DEG:
                continue
            angle_gap = abs(first['angle_deg'] - second['angle_deg'])
            angle_gap = min(angle_gap, 180.0 - angle_gap)
            if not (TILT_PAIR_ANGLE_MIN_DEG <= angle_gap <= TILT_PAIR_ANGLE_MAX_DEG):
                continue
            endpoint_gap_m = (
                _line_endpoint_gap_px(first['pixels'], second['pixels'])
                * TILT_GRID_M)
            if endpoint_gap_m > TILT_PAIR_ENDPOINT_MAX_M:
                continue
            score = first['length_m'] + second['length_m'] - endpoint_gap_m
            if best_pair is None or score > best_pair[0]:
                best_pair = (score, endpoint_gap_m, first, second)
    if best_pair is None:
        return None

    return {
        'score': best_pair[0],
        'endpoint_gap_m': best_pair[1],
        'selected': [best_pair[2], best_pair[3]],
        'front_pts': front_pts,
        'front_depth_m': float(front_depth),
        'v_front': float(v_front),
        'u_range': (float(u_min), float(u_max)),
    }


def _detect_tilted_box(pts, actual_top_z, box_h, view=False):
    """在当前箱体的自适应 U-Z 正视投影中寻找相连、近似正交的两条长斜边。

    返回 None 表示未发现倾斜；否则返回包含两条三维可视化线及倾角的字典。
    局部 U 轴由本帧点云自动拟合，倾斜判断不使用 yaw_offset_deg。
    """
    if actual_top_z is None or box_h is None or box_h <= 0:
        return None

    z_min = float(actual_top_z - box_h - TILT_Z_MARGIN_BOTTOM)
    z_max = float(actual_top_z + TILT_Z_MARGIN_TOP)
    frame = _fit_tilt_projection_frame(pts, z_min, z_max)
    if frame is None:
        return None
    depth_results = [
        _find_tilt_pair_at_depth(
            pts, frame, z_min, z_max, front_depth, v_front)
        for v_front in frame['v_fronts']
        for front_depth in TILT_FRONT_DEPTHS
    ]
    depth_results = [item for item in depth_results if item is not None]
    if not depth_results:
        return None
    # 若两档都命中，优先选择总线长更大且端点更紧密的一档用于日志和可视化。
    detected = max(depth_results, key=lambda item: item['score'])
    front_pts = detected['front_pts']

    lines = []
    for item in detected['selected']:
        raw = item['pixels']
        u_min = detected['u_range'][0]
        p0_uz = np.array([
            u_min + (float(raw[0]) + 0.5) * TILT_GRID_M,
            z_min + (float(raw[1]) + 0.5) * TILT_GRID_M,
        ])
        p1_uz = np.array([
            u_min + (float(raw[2]) + 0.5) * TILT_GRID_M,
            z_min + (float(raw[3]) + 0.5) * TILT_GRID_M,
        ])
        line_v = _supporting_line_v(front_pts, frame, p0_uz, p1_uz)

        def to_xyz(point_uz):
            xy = (frame['origin_xy'] + point_uz[0] * frame['u_axis'] +
                  line_v * frame['v_axis'])
            return np.array([xy[0], xy[1], point_uz[1]], dtype=float)

        lines.append({
            **item,
            'p0_uz': p0_uz,
            'p1_uz': p1_uz,
            'p0_xyz': to_xyz(p0_uz),
            'p1_xyz': to_xyz(p1_uz),
        })

    # 水平边偏转量为 |angle|，竖直边偏转量为 90-|angle|；两者取平均更抗栅格误差。
    axis_deviations = [min(abs(line['angle_deg']), 90.0 - abs(line['angle_deg']))
                       for line in lines]
    tilt_deg = float(np.mean(axis_deviations))
    if (tilt_deg < TILT_RESULT_MIN_DEG or
            min(axis_deviations) < TILT_EACH_LINE_MIN_DEG):
        return None
    result = {
        'lines': lines,
        'tilt_deg': tilt_deg,
        'endpoint_gap_m': detected['endpoint_gap_m'],
        'front_depth_m': detected['front_depth_m'],
        'projection_yaw_deg': frame['yaw_deg'],
        'u_range': frame['u_range'],
        'v_axis': frame['v_axis'],
    }
    if view:
        _show_tilt_result(frame['roi'], front_pts, result)
    return result


def _measure_pair_occupancy(pts, x_left, x_right, y_front):
    """检查两个候选侧面之间是否存在连续的箱体点云。

    返回 (coverage, point_density, point_count, bin_count)：coverage 为沿 X
    方向被箱体点覆盖的 bin 比例。只取当前面靠雷达的薄层，避免
    空隙后方的旧箱或车壁被误认为当前抓。
    """
    gap = float(x_right - x_left)
    margin = min(PAIR_OCCUPANCY_SIDE_MARGIN_M, gap * 0.1)
    inner_left = float(x_left + margin)
    inner_right = float(x_right - margin)
    inner_width = inner_right - inner_left
    if inner_width <= 1e-6:
        return 0.0, 0.0, 0, 0

    occupancy_pts = pts[
        (pts[:, 0] >= inner_left) & (pts[:, 0] <= inner_right) &
        (pts[:, 1] >= y_front - PAIR_OCCUPANCY_FRONT_DEPTH_M) &
        (pts[:, 1] <= y_front)
    ]
    bin_count = max(1, int(math.ceil(
        inner_width / PAIR_OCCUPANCY_BIN_M)))
    hist, _ = np.histogram(
        occupancy_pts[:, 0], bins=bin_count,
        range=(inner_left, inner_right))
    occupied_bins = int(np.count_nonzero(
        hist >= PAIR_OCCUPANCY_MIN_PTS_PER_BIN))
    coverage = occupied_bins / float(bin_count)
    density = len(occupancy_pts) / inner_width
    return float(coverage), float(density), int(len(occupancy_pts)), bin_count


def _compute_width(pc1, pc2, view=None, yaw_offset_deg=0.0,
                   rel_top_h=None, box_h=None, expected_width_mm=None,
                   box_width_mm=None, log_callback=None):
    """
    合并双雷达点云，测量左右侧面总宽度（mm）。

    view=None 时在调用时读取顶部 VIEW 开关（避免默认参数在定义时被绑死）。
    yaw_offset_deg: 拍照位相对正对的偏航角(度)。雷达绕机器人 J1 轴摆动导致点云偏航，
                    需将点云转回正对系再测量。
    rel_top_h / box_h: 当前抓"顶面距地板的高度"和箱子竖向高度(米)。两者都给定时，
                    内部自标定地板 + 实测箱顶，把直通滤波 Z 锁定到"当前行"隔离相邻层；
                    否则用全局 PASS_Z。
    expected_width_mm / box_width_mm: 当前抓理论总宽度和单箱宽度；候选面间距必须落在
                    [理论宽度-单箱宽度, 理论宽度+单箱宽度] 范围内。
    log_callback: 检测到箱体倾斜时的日志回调；在线模式传主节点 logger.warning。

    策略：
      0. 偏航补偿 + （可选）当前行 Z 锁定
      1. 原始点云自适应 U-Z 轮廓倾斜检测；命中时返回 None
      2. 直通滤波
      3. 法向量滤波保留 ±x 侧面点
      4. DBSCAN 聚类，所有有效候选面两两组合
      5. 保留间距位于理论宽度 ± 单箱宽度范围内的组合
      6. 检查两面之间的点云覆盖率/密度，排除已有箱体占用的区间
      7. 在剩余空缺口中取最接近下一抓理论宽度者，宽度 = 右面 x - 左面 x
    """
    if view is None:
        view = VIEW
    
    # 记录开始时间
    total_start_time = time.time()
    
    # 一次性合并点云并转换为numpy数组，避免后续重复转换
    pcd_combined = valid_pcd(pc1) + valid_pcd(pc2)
    pts = np.asarray(pcd_combined.points)
    # 倾斜检测使用未做手工 yaw 补偿的原始点云，局部 U 轴由本帧自动估计。
    tilt_pts = pts

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
    actual_top_z = None
    if rel_top_h is not None and box_h is not None:
        _zmin, _zmax, actual_top_z = _lock_layer_z_range(
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

    # ── 0d. 箱体倾斜检测：使用独立正视轮廓，不受后续 ±X 侧面法向过滤影响 ──────
    tilt_result = _detect_tilted_box(
        tilt_pts, actual_top_z, box_h, view=view)
    if tilt_result is not None:
        line_text = "；".join(
            f"斜边{index}角度={line['angle_deg']:+.1f}°、长度={line['length_m'] * 1000:.0f}mm"
            for index, line in enumerate(tilt_result['lines'], start=1)
        )
        message = (
            f"垛面异常：检测到箱体倾斜，估计倾斜角={tilt_result['tilt_deg']:.1f}°；"
            f"自估垛面yaw={tilt_result['projection_yaw_deg']:+.1f}°；"
            f"{line_text}；端点间距={tilt_result['endpoint_gap_m'] * 1000:.0f}mm；"
            f"局部V窗口={tilt_result['front_depth_m'] * 1000:.0f}mm，返回 status=2")
        _dbg(message)
        if log_callback is not None:
            try:
                log_callback(message)
            except Exception as exc:
                _dbg(f"倾斜异常日志回调失败：{type(exc).__name__}: {exc}")
        return None

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
        _show_geometries(
            f"雷达前方{VIEW_FRONT_Y}m原始点云(灰) + 直通框选(绿)",
            [raw, sel])
    
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
        _show_geometries("法向滤波：灰=原始  青=侧面点云", [bg0, sp])

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
        _show_geometries(
            f"聚类结果（{n_clusters} 个簇，噪点灰色）", [bg1, sp_pcd])

    # 记录聚类耗时
    clustering_time = time.time() - start_time
    start_time = time.time()

    # ── 4. 生成有效候选面；不再以 X=0 强制拆分左右，两侧或同侧面均可组合 ──────
    # 过滤规则：
    #   - 点数不足 MIN_CLUSTER_PTS → 噪点
    #   - z 跨度不足 MIN_CLUSTER_ZSPAN_M → 顶部薄片(灯/横梁/手柄)，过滤
    #   - z 重心 > MAX_CLUSTER_CZ_M → 车厢顶部凸起结构(管线/灯具)，过滤
    #     （Z 锁定模式下范围已收窄到当前行，关闭此绝对上限避免误删高处当前行）
    candidates = []
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
        x_face = float(np.median(kpts[:, 0]))
        candidates.append({
            'k': k,
            'pts': kpts,
            'x_face': x_face,
        })

    if len(candidates) < 2:
        _dbg(f"计算失败：有效候选面不足2个（当前{len(candidates)}个）")
        return None

    if (expected_width_mm is None or box_width_mm is None or
            not np.isfinite(expected_width_mm) or not np.isfinite(box_width_mm) or
            expected_width_mm <= 0 or box_width_mm <= 0):
        _dbg(f"计算失败：理论宽度或单箱宽度无效（理论={expected_width_mm}, 单箱={box_width_mm}）")
        return None

    min_gap_mm = max(0.0, float(expected_width_mm) - float(box_width_mm))
    max_gap_mm = float(expected_width_mm) + float(box_width_mm)
    candidates.sort(key=lambda item: item['x_face'])
    valid_pairs = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1:]:
            gap_float = (right['x_face'] - left['x_face']) * 1000.0
            within_range = min_gap_mm <= gap_float <= max_gap_mm
            z_diff_ = abs(float(left['pts'][:, 2].mean()) -
                          float(right['pts'][:, 2].mean()))
            coverage, occupancy_density, occupancy_count, occupancy_bins = (
                _measure_pair_occupancy(
                    pts, left['x_face'], right['x_face'], pass_y[1])
                if within_range else (0.0, 0.0, 0, 0)
            )
            coverage_has_box = coverage >= PAIR_OCCUPANCY_MIN_COVERAGE
            _dbg(
                f"候选组合 k={left['k']}({len(left['pts'])}点,x={left['x_face']:+.3f}) - "
                f"k={right['k']}({len(right['pts'])}点,x={right['x_face']:+.3f})  "
                f"间距={gap_float:.0f}mm 允许=[{min_gap_mm:.0f},{max_gap_mm:.0f}]mm "
                f"z重心差={z_diff_*1000:.0f}mm "
                f"内部占用={coverage*100:.0f}%({occupancy_count}点/"
                f"{occupancy_bins}bin,密度={occupancy_density:.0f}点/m) "
                f"{'覆盖较高' if coverage_has_box else '低覆盖'} "
                f"{'间距满足' if within_range else '间距不满足'}")
            if within_range:
                # 首要选择最接近理论宽度者；差值相同时优先点数更充足的组合。
                score = (
                    abs(gap_float - float(expected_width_mm)),
                    -min(len(left['pts']), len(right['pts'])),
                    -(len(left['pts']) + len(right['pts'])),
                )
                valid_pairs.append((
                    score, left, right, gap_float, z_diff_,
                    coverage, occupancy_density))

    if valid_pairs:
        max_occupancy_density = max(item[6] for item in valid_pairs)
        min_density = (
            max_occupancy_density * PAIR_OCCUPANCY_MIN_REL_DENSITY)
        empty_gap_pairs = []
        for item in valid_pairs:
            has_box = (
                item[5] >= PAIR_OCCUPANCY_MIN_COVERAGE and
                item[6] >= min_density)
            if not has_box:
                empty_gap_pairs.append(item)
                _dbg(
                    f"候选组合 k={item[1]['k']} - k={item[2]['k']} 确认为空缺口："
                    f"覆盖率={item[5]*100:.0f}%，密度={item[6]:.0f}点/m，"
                    f"本帧箱体参考密度={max_occupancy_density:.0f}点/m")
            else:
                _dbg(
                    f"候选组合 k={item[1]['k']} - k={item[2]['k']} 被排除："
                    f"两面之间已有箱体（覆盖率={item[5]*100:.0f}%，"
                    f"密度={item[6]:.0f}点/m）")
        valid_pairs = empty_gap_pairs

    if not valid_pairs:
        _dbg(
            f"计算失败：{len(candidates)}个候选面中无“间距合理且两面之间为空”的缺口，"
            f"允许间距=[{min_gap_mm:.0f}, {max_gap_mm:.0f}]mm，"
            f"空缺口覆盖率阈值=<{PAIR_OCCUPANCY_MIN_COVERAGE*100:.0f}%")
        return None

    (_, left, right, gap_float, z_diff,
     selected_coverage, selected_density) = min(
        valid_pairs, key=lambda item: item[0])
    left_k, right_k = left['k'], right['k']
    lc, rc = left['pts'], right['pts']
    x_left_face, x_right_face = left['x_face'], right['x_face']
    gap_mm = int(gap_float)
    _dbg(
        f"最终选中候选面 k={left_k}({len(lc)}点,x={x_left_face:+.3f}) - "
        f"k={right_k}({len(rc)}点,x={x_right_face:+.3f})  "
        f"宽度={gap_mm}mm 理论={float(expected_width_mm):.0f}mm "
        f"允许=[{min_gap_mm:.0f},{max_gap_mm:.0f}]mm z重心差={z_diff*1000:.0f}mm "
        f"内部占用={selected_coverage*100:.0f}% "
        f"密度={selected_density:.0f}点/m")

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

        _show_geometries("左簇(橙) / 右簇(绿) + 原始(灰)", [bg, lf, rf])

        left_plane_mesh  = show_plane([1, 0, 0, -x_left_face],  [1, 0.5, 0])
        right_plane_mesh = show_plane([1, 0, 0, -x_right_face], [0, 1, 0.3])

        bg2 = o3d.geometry.PointCloud()
        bg2.points = o3d.utility.Vector3dVector(pts)
        bg2.paint_uniform_color([0.5, 0.5, 0.5])
        win_name = (f"拟合平面  面A(橙)={x_left_face:.3f}m  "
                    f"面B(绿)={x_right_face:.3f}m  宽={gap_mm}mm")
        _show_geometries(
            win_name, [bg2, left_plane_mesh, right_plane_mesh])

    return gap_mm


def check_stacking(length, pc1, pc2, tolerance=50, yaw_offset_deg=0.0,
                   rel_top_h=None, box_h=None, box_width_mm=None,
                   log_callback=None, view=False):
    """测量堆叠宽度，返回 measured_mm（计算成功）或 None（计算失败/报错）。
    length 为当前抓理论总宽度；box_width_mm 为当前姿态下单箱宽度。
    两候选面间距只有落在 length ± box_width_mm 范围内才算有效；tolerance 保留兼容旧调用。
    yaw_offset_deg: 拍照位相对正对的偏航角(度)，用于点云偏航补偿。
    rel_top_h / box_h: 当前抓顶面距地板高度和箱子竖向高度(米)，用于把测宽锁定到当前行。
    log_callback: 倾斜检出日志回调；倾斜按计算失败返回 None，由主节点发送 status=2。
    view: 在线默认 False，避免2D/3D交互窗口阻塞机器人状态返回；离线可显式开启。
    """
    return _compute_width(pc1, pc2, view=view, yaw_offset_deg=yaw_offset_deg,
                          rel_top_h=rel_top_h, box_h=box_h,
                          expected_width_mm=length, box_width_mm=box_width_mm,
                          log_callback=log_callback)


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


def process_point_cloud(length, box_width_mm=None,
                        topic1='/lidar_points1', topic2='/lidar_points2'):
    """向后兼容接口：内部采集点云后调用 _compute_width。"""
    pc1, pc2 = collect_dual_lidar_once(topic1, topic2, frames=3)
    return _compute_width(
        pc1, pc2, expected_width_mm=length, box_width_mm=box_width_mm)


# ─── 离线测试入口 ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    import glob

    DEBUG = True   # 离线测试：打开调试打印

    # ↓ 只填文件名即可，目录自动使用 _DEFAULT_SAVE_DIR；留空则自动选取最新文件
    _FILENAME = 'merged_20260721/merged_20260721_114209.pcd'

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
    # 细支箱离线回放使用与 robot_process_node 相同的补偿角：-56.6 - (-60.5) = +3.9°。
    measured = _compute_width(pcd, empty, yaw_offset_deg=3.9,
                              rel_top_h=0.3*1, box_h=0.3,
                              expected_width_mm=1100, box_width_mm=550)  # view 跟随顶部 VIEW 开关
    if measured is None:
        print('\n检测状态: status=2（倾斜异常或宽度计算失败）')
    else:
        print(f'\n检测状态: status=1  测量宽度: {measured} mm')

    if len(args) >= 2:
        length = int(args[1])
        if measured is None:
            print(f'理论宽度: {length} mm  无有效测量宽度，不进行差值计算')
        else:
            label = '通过' if abs(measured - length) <= 50 else '不通过'
            print(f'理论宽度: {length} mm  差值: {measured - length:+.1f} mm  结果: {label}')
