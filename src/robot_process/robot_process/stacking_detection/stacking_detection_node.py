"""双雷达垛面宽度与箱体倾斜检测。

在线入口先采集两路 PointCloud2，再进行偏航补偿、当前面/当前层裁剪、候选侧面
配对和缺口占用复核。模块内部点云单位为米，对外测量结果单位为毫米；检测失败
统一返回 ``None``，由主节点转换为机器人协议状态。
"""

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
    """仅在 ``DEBUG`` 开启时输出离线诊断信息。"""
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


def _make_xz_rectangle(x_left, x_right, y_value, z_min, z_max, color):
    """创建固定Y位置的X-Z矩形线框，用于标记订单目标缺口。"""
    points = np.array([
        [x_left, y_value, z_min],
        [x_right, y_value, z_min],
        [x_right, y_value, z_max],
        [x_left, y_value, z_max],
    ], dtype=float)
    rectangle = o3d.geometry.LineSet()
    rectangle.points = o3d.utility.Vector3dVector(points)
    rectangle.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
    ], dtype=int))
    rectangle.colors = o3d.utility.Vector3dVector(np.tile(
        np.asarray(color, dtype=float), (4, 1)))
    return rectangle


def _make_vertical_marker(x_value, y_value, z_min, z_max, color):
    """创建固定X位置的竖直线，用于标记正面点云突变边界。"""
    marker = o3d.geometry.LineSet()
    marker.points = o3d.utility.Vector3dVector(np.array([
        [x_value, y_value, z_min],
        [x_value, y_value, z_max],
    ], dtype=float))
    marker.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=int))
    marker.colors = o3d.utility.Vector3dVector(np.array([color], dtype=float))
    return marker


def _show_front_gap_diagnostics(
        filtered_pts, front_gap_pts, target_range,
        front_candidates, pass_y, pass_z):
    """显示正面缺口边界诊断，失败帧也保留订单目标和正面点云信息。"""
    if target_range is None:
        return False

    geometries = []
    background = o3d.geometry.PointCloud()
    background.points = o3d.utility.Vector3dVector(filtered_pts)
    background.paint_uniform_color([0.55, 0.55, 0.55])
    geometries.append(background)

    front_cloud = o3d.geometry.PointCloud()
    front_cloud.points = o3d.utility.Vector3dVector(front_gap_pts)
    front_cloud.paint_uniform_color([0.10, 0.55, 1.00])
    geometries.append(front_cloud)

    y_value = float(pass_y[1]) + 0.015
    z_min, z_max = map(float, pass_z)
    geometries.append(_make_xz_rectangle(
        float(target_range['left']), float(target_range['right']),
        y_value, z_min, z_max, [1.00, 0.85, 0.10]))
    check_region = _resolve_pair_occupancy_region(
        target_range['left'], target_range['right'], z_min, z_max)
    if check_region is not None:
        geometries.append(_make_xz_rectangle(
            check_region['x_min'], check_region['x_max'], y_value + 0.01,
            check_region['z_min'], check_region['z_max'], [1.00, 0.40, 0.00]))

    for candidate in front_candidates:
        edge_x = float(candidate['x_face'])
        side_name = candidate['side_name']
        if side_name == 'left':
            gap_left, gap_right = (
                edge_x, edge_x + FRONT_GAP_EDGE_WINDOW_M)
        else:
            gap_left, gap_right = (
                edge_x - FRONT_GAP_EDGE_WINDOW_M, edge_x)
        gap_points = front_gap_pts[
            (front_gap_pts[:, 0] >= gap_left) &
            (front_gap_pts[:, 0] <= gap_right)]
        if check_region is not None:
            gap_points = gap_points[
                (gap_points[:, 2] >= check_region['z_min']) &
                (gap_points[:, 2] <= check_region['z_max'])]
        if len(gap_points):
            gap_cloud = o3d.geometry.PointCloud()
            gap_cloud.points = o3d.utility.Vector3dVector(gap_points)
            gap_cloud.paint_uniform_color([1.00, 0.15, 0.10])
            geometries.append(gap_cloud)

        box_cloud = o3d.geometry.PointCloud()
        box_cloud.points = o3d.utility.Vector3dVector(candidate['pts'])
        box_cloud.paint_uniform_color([0.10, 1.00, 0.20])
        geometries.append(box_cloud)
        geometries.append(_make_vertical_marker(
            edge_x, y_value + 0.005, z_min, z_max,
            [0.90, 0.10, 1.00]))

    title = (
        "正面缺口诊断：灰=当前层 蓝=正面薄层 黄=订单范围 "
        "橙=自适应检查区 紫=突变边界 绿=箱侧窗口 红=检查区残点")
    return _show_geometries(title, geometries, width=1100, height=750)


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
MIN_CLUSTER_PTS = 10             # 候选簇至少10点，兼顾稀疏侧面与孤立噪点过滤
MIN_CLUSTER_ZSPAN_M = 0.02       # 候选簇至少20mm高，排除近似水平的薄片噪声
MAX_CLUSTER_CZ_M = 1.0           # 簇 z 重心上限(米)：> 此值视为车厢顶部凸起结构(管线/灯)，丢弃

# ── 候选面之间的箱体占用检查 ──
PAIR_OCCUPANCY_FRONT_NORMAL_ANGLE_DEG = 25  # 箱子正面法向与±Y轴的最大夹角
PAIR_OCCUPANCY_MIN_FRONT_PTS = 50   # 整帧正面点不足时退回原始点云薄层检查
PAIR_OCCUPANCY_FRONT_DEPTH_M = 0.20  # 正面点不足时的兜底检查深度
PAIR_OCCUPANCY_SIDE_MARGIN_M = 0.05  # 两候选面内缩50mm，不把边界侧面自身当作箱体证据
PAIR_OCCUPANCY_BIN_M = 0.10          # 沿宽度方向每100mm统计一个占用 bin
PAIR_OCCUPANCY_MIN_PTS_PER_BIN = 10 # bin 内至少此点数才认为有箱体表面
PAIR_OCCUPANCY_MIN_COVERAGE = 0.60  # 连续箱体覆盖的主判定阈值
PAIR_OCCUPANCY_MIN_PARTIAL_COVERAGE = 0.30  # 有完整箱体参照时允许识别部分覆盖
PAIR_OCCUPANCY_MIN_REL_DENSITY = 0.25  # 占用区密度至少达到本帧最完整箱体区的25%
PAIR_OCCUPANCY_Z_BIN_M = 0.10       # 阶梯缺口按高度每100mm统计一行
# 阶梯检查区按缺口宽高比自适应：只留底部小容差，不再忽略整个下半层。
PAIR_OCCUPANCY_BOTTOM_HEIGHT_RATIO = 0.20  # 底部避让最多为当前检测层高度的20%
PAIR_OCCUPANCY_BOTTOM_WIDTH_RATIO = 0.50   # 宽度限制为50%W，仍受20%H和80mm上限约束
PAIR_OCCUPANCY_BOTTOM_MAX_M = 0.08         # 底部额外避让绝对上限80mm
PAIR_OCCUPANCY_WINDOW_HEIGHT_RATIO = 0.50 # 局部复核窗口不超过检测层高度的50%
PAIR_OCCUPANCY_WINDOW_WIDTH_RATIO = 0.25  # 窗口高度随缺口宽度调整
PAIR_OCCUPANCY_WINDOW_MIN_M = 0.08         # 通常至少检查连续80mm高度
PAIR_OCCUPANCY_WINDOW_ZSPAN_RATIO = 0.40  # 局部箱面须具有足够真实高度跨度
PAIR_OCCUPANCY_WINDOW_ZSPAN_MAX_M = 0.06  # 连续60mm高度已可作为局部箱体证据
PAIR_WIDTH_TOLERANCE_BOXES = 1.5     # 候选面允许间距 = 检测参考宽度 ± 1.5个单箱宽度

# ── 所有垛面共用的订单位置引导 ──
ORDER_WALL_SPAN_TOLERANCE_M = 0.40   # 检出左右车壁间距与订单车宽的最大允许偏差
ORDER_TARGET_EDGE_MATCH_M = 0.18     # 实测候选面距订单换算边界180mm内视为同一边界
ORDER_TARGET_PAIR_EDGE_M = 0.25      # 订单引导时，候选对每侧最多偏离目标边界250mm
ORDER_TARGET_EDGE_MIN_PTS = 20       # 实测边界至少20点，弱小簇不能覆盖订单虚拟边界
# 侧面已通过聚类基础过滤，不再追加“覆盖当前层40%高度”的门槛。

# ── 独立车壁标定 ──
# 缺口测宽只使用当前面、当前层的小窗口；车壁定位则必须利用全深度、全高度的大面，
# 否则首层靠近地板/加强筋时，整面车壁会被裁成几十个点的小簇。
WALL_VOXEL_M = 0.025                 # 全局车壁先25mm降采样，控制法向和聚类耗时
WALL_NORMAL_RADIUS_M = 0.10          # 全局大面法向搜索半径
WALL_NORMAL_MAX_NN = 40
WALL_NORMAL_ANGLE_DEG = 30           # 兼容波纹板、加强筋造成的局部法向起伏
WALL_DBSCAN_EPS_M = 0.18
WALL_DBSCAN_MIN_POINTS = 10
WALL_MIN_POINTS = 100                # 降采样后的最低点数
WALL_MIN_Y_SPAN_M = 0.65             # 强车壁沿车长方向至少连续65cm
WALL_REFERENCE_MIN_Y_SPAN_M = 0.35   # 上一面参考只可引导重拟合当前弱车壁
WALL_MIN_Z_SPAN_M = 0.80             # 车壁须跨越多层高度，排除当前层箱侧
WALL_ONE_SIDE_MIN_ABS_X_M = 0.80     # 单壁推算只接受明显位于车厢外侧的强大面
WALL_CACHE_WIDTH_TOLERANCE_MM = 50.0 # 同一面内按相近有效车宽匹配车壁缓存
WALL_REFERENCE_X_TOLERANCE_M = 0.18  # 上一面坐标附近18cm内寻找当前帧车壁点

_wall_pair_cache = {}
_previous_wall_pair_cache = {}
_active_wall_face_key = None

# ── 缺口的正面点云突变边界 ──
# 现场正面点云受高层稀疏、遮挡和箱面起伏影响较大。目标位置由订单和车壁限定后，
# 仍优先采用法向聚类得到的可靠实测侧面；目标某侧缺失或只有弱候选时，才从当前层
# 宽高比自适应检查区内正面薄层的“有点↔空白”突变补齐该侧边界。
FRONT_GAP_EDGE_SEARCH_M = 0.18       # 仅在订单边界±180mm内寻找，防止吸附到相邻箱缝
FRONT_GAP_EDGE_BIN_M = 0.02          # X方向20mm分箱，兼顾边界精度和高层稀疏点云
FRONT_GAP_EDGE_WINDOW_M = 0.10       # 边界内外各检查100mm连续区域
FRONT_GAP_EDGE_MIN_PTS_PER_BIN = 5  # 每20mm箱侧至少5点才算有效覆盖
FRONT_GAP_EDGE_MIN_BOX_COVERAGE = 0.60  # 箱体一侧至少60%的bin有点
FRONT_GAP_EDGE_MAX_GAP_COVERAGE = 0.20  # 缺口一侧最多20%的bin有点
FRONT_GAP_EDGE_MIN_DENSITY_RATIO = 4.0  # 箱侧点密度至少为缺口侧4倍
FRONT_GAP_EDGE_MIN_BOX_PTS = 20      # 箱侧窗口至少20点，拒绝零星噪声突变
FRONT_GAP_EDGE_MIN_ZSPAN_RATIO = 0.30  # 正面边界高度至少覆盖当前层30%
FRONT_GAP_EDGE_MIN_ZSPAN_M = 0.10      # 正面突变仍需至少100mm高度，避免薄片误补边
FRONT_GAP_EDGE_MAX_ZSPAN_M = 0.20      # 高箱正面边界达到200mm即可，侧面缺失时不过严
FRONT_GAP_EDGE_Z_BIN_M = 0.02          # X-Z支持统计的高度分箱20mm
FRONT_GAP_EDGE_MIN_Z_BINS = 3          # 每个箱面X分箱至少由3个高度分箱共同支持
FRONT_GAP_EDGE_MIN_RUN_BINS = 3        # 边界后须紧邻至少3个连续有支持的X分箱
# 侧面测宽已进入机器人减速/停止区间时，强制用正面突变边界交叉复核。
# 机器人约定：余量 <50mm 停止，50~70mm 减速。
SIDE_WIDTH_FRONT_RECHECK_MARGIN_MM = 70.0
SIDE_WIDTH_STOP_MARGIN_MM = 50.0

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
TILT_PAIR_ANGLE_MIN_DEG = 85.0   # 两条箱边必须接近正交；收紧范围避免不同箱边误配
TILT_PAIR_ANGLE_MAX_DEG = 95.0
TILT_PAIR_ENDPOINT_MAX_M = 0.04  # 相邻箱边应在同一角点相接，端点最大距离40mm
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
    """累计两路雷达的固定帧数，并分别合并为单次检测点云。"""

    def __init__(
        self,
        topic1='/lidar_points1',
        topic2='/lidar_points2',
        max_frames=LIDAR_FRAMES
    ):
        """创建两个点云订阅；当两侧均达到 ``max_frames`` 时置完成标志。"""
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
        """接收雷达1点云并尝试完成本次采集。"""
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
        """接收雷达2点云并尝试完成本次采集。"""
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
        """两路帧数均满足要求时合并帧并标记采集完成。"""
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
        """将 ROS PointCloud2 向量化转换为有限 XYZ 数组。"""
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
    """RANSAC 分割主平面，返回 ``(模型, 内点云, 外点云)``。"""
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return plane_model, inlier_cloud, outlier_cloud


def show_plane(model, color):
    """把 ``ax+by+cz+d=0`` 绘制为指定颜色的薄盒网格。"""
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
    """求三个平面方程的唯一交点。"""
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
    """将 4×4 位姿矩阵转换为 ``[x,y,z,roll,pitch,yaw]``（角度制）。"""
    assert r.shape == (4, 4)
    # 计算欧拉角 (ZYX 顺序)
    yaw = np.arctan2(r[1, 0], r[0, 0])  # z轴旋转
    pitch = np.arctan2(-r[2, 0], np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2))
    roll = np.arctan2(r[2, 1], r[2, 2])  # x轴旋转
    return [r[0, 3], r[1, 3], r[2, 3], roll * 180 / math.pi, pitch * 180 / math.pi, yaw * 180 / math.pi]


def point_to_plane_distance(point, a, b, c, d):
    """返回点代入平面方程后的未归一化有符号值（兼容旧工具）。"""
    return a * point[0] + b * point[1] + c * point[2] + d


def fiterCloud(pcd):
    """执行统计离群点过滤；函数名保留旧接口拼写。"""
    down_pcd = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)[0]
    return down_pcd


view = VIEW  # 兼容旧函数(clustFrontBoard)的小写开关，统一跟随 VIEW

def clustFrontBoard(pcd):
    """按 DBSCAN 把旧版 L 形前板分成两组平面并返回模型及 Y 跨度。"""
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
    return [[final_plane_model[0], final_plane_model[1], final_plane_model[2], final_plane_model[3],
             bounding_box_final[1]],
            [other_plane_model[0], other_plane_model[1], other_plane_model[2], other_plane_model[3],
             bounding_box_other[1]]]

def point_to_plane_distance(x, y, z, a, b, c, d):
    """计算点到平面 ``ax+by+cz+d=0`` 的绝对欧氏距离。"""
    numerator = a * x + b * y + c * z + d
    denominator = math.sqrt(a * a + b * b + c * c)

    if denominator == 0:
        return 0.0  # 避免除零

    return abs(numerator / denominator)

def stitching_pcd(pcd1, pcd2, angle):
    """旧版点云拼接演示：使用固定示例变换并显示结果。

    ``angle`` 当前未参与计算；在线双雷达检测不调用该函数。
    """
    transformation_matrix = np.array([[0.866, -0.5, 0, 1],
                                    [0.5, 0.866, 0, 2],
                                    [0, 0, 1, 3],
                                    [0, 0, 0, 1]])

    pcd2.transform(transformation_matrix)
    combined_pcd = pcd1 + pcd2
    _show_geometries("Stitched point cloud", [combined_pcd])

def rotation_pcd(matrix, pcd):
    """按 ``[x,y,z,roll,pitch,yaw]`` 变换点云并原位返回。

    平移单位跟随点云，三个角度输入为度；旋转顺序为 ``Rz @ Ry @ Rx``。
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
    """原位删除包含 NaN/Inf 的点并返回同一个点云对象。"""
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


def reset_wall_cache():
    """清空当前订单的车壁标定状态。

    主节点每次开始处理新订单时调用，避免相同车宽的下一辆车复用上一辆车的
    点云坐标。离线批量回放跨订单时也应在订单边界调用。
    """
    global _active_wall_face_key
    _wall_pair_cache.clear()
    _previous_wall_pair_cache.clear()
    _active_wall_face_key = None


def activate_wall_face(face_key):
    """切换车壁标定所属的码垛面。

    进入新面时，当前面缓存会转为仅用于寻找当前帧弱车壁的参考，不会
    被直接当作新面坐标。返回 True 表示确实发生了面切换。
    """
    global _active_wall_face_key
    if face_key is None:
        return False
    normalized_key = tuple(face_key) if isinstance(face_key, list) else face_key
    if normalized_key == _active_wall_face_key:
        return False
    _previous_wall_pair_cache.clear()
    _previous_wall_pair_cache.update(
        {width: dict(pair) for width, pair in _wall_pair_cache.items()})
    _wall_pair_cache.clear()
    _active_wall_face_key = normalized_key
    return True


def _select_global_wall_pair(candidates, car_width_mm):
    """从全局大面候选中选择跨度最接近订单车宽的一对实测车壁。"""
    if (car_width_mm is None or not np.isfinite(car_width_mm) or
            float(car_width_mm) <= 0):
        return None
    expected_span = float(car_width_mm) / 1000.0
    best = None
    ordered = sorted(
        (
            item for item in candidates
            if float(item['y_span']) >= WALL_MIN_Y_SPAN_M
        ),
        key=lambda item: item['x_face'])
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1:]:
            span = float(right['x_face']) - float(left['x_face'])
            error = abs(span - expected_span)
            score = (
                error,
                -min(float(left['y_span']), float(right['y_span'])),
                -min(float(left['z_span']), float(right['z_span'])),
                -(int(left['points']) + int(right['points'])),
            )
            if best is None or score < best['score']:
                best = {
                    'score': score,
                    'left_candidate': left,
                    'right_candidate': right,
                    'left': float(left['x_face']),
                    'right': float(right['x_face']),
                    'span': span,
                    'wall_error': error,
                    'source': 'global_measured',
                }
    if best is None or best['wall_error'] > ORDER_WALL_SPAN_TOLERANCE_M:
        return None
    return best


def _extract_global_wall_candidates(pts):
    """提取跨越多层的车壁大面及可供历史参考重拟合的弱大面。"""
    roi_mask = (
        (pts[:, 0] >= PASS_X[0]) & (pts[:, 0] <= PASS_X[1]) &
        (pts[:, 1] >= PASS_Y[0]) & (pts[:, 1] <= PASS_Y[1]) &
        (pts[:, 2] >= PASS_Z[0]) & (pts[:, 2] <= PASS_Z[1])
    )
    roi = pts[roi_mask]
    if len(roi) < WALL_MIN_POINTS:
        return []

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(roi)
    cloud = cloud.voxel_down_sample(WALL_VOXEL_M)
    if len(cloud.points) < WALL_MIN_POINTS:
        return []
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=WALL_NORMAL_RADIUS_M,
            max_nn=WALL_NORMAL_MAX_NN))
    sampled = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    side_points = sampled[
        np.abs(normals[:, 0]) >
        np.cos(np.radians(WALL_NORMAL_ANGLE_DEG))]
    if len(side_points) < WALL_DBSCAN_MIN_POINTS:
        return []

    side_cloud = o3d.geometry.PointCloud()
    side_cloud.points = o3d.utility.Vector3dVector(side_points)
    labels = np.asarray(side_cloud.cluster_dbscan(
        eps=WALL_DBSCAN_EPS_M,
        min_points=WALL_DBSCAN_MIN_POINTS,
        print_progress=False))
    candidates = []
    cluster_count = int(labels.max()) + 1 if len(labels) else 0
    for cluster_id in range(cluster_count):
        cluster = side_points[labels == cluster_id]
        if len(cluster) < WALL_MIN_POINTS:
            continue
        y_span = float(np.ptp(cluster[:, 1]))
        z_span = float(np.ptp(cluster[:, 2]))
        if (y_span < WALL_REFERENCE_MIN_Y_SPAN_M or
                z_span < WALL_MIN_Z_SPAN_M):
            continue
        candidates.append({
            'k': f'global_wall_{cluster_id}',
            'x_face': float(np.median(cluster[:, 0])),
            'pts': cluster,
            'points': int(len(cluster)),
            'y_span': y_span,
            'z_span': z_span,
        })
    return candidates


def _find_wall_pair_cache(cache, car_width_mm, source):
    """按相近有效车宽查找指定的车壁坐标缓存。"""
    if car_width_mm is None or not np.isfinite(car_width_mm):
        return None
    for cached_width, cached in cache.items():
        if abs(float(cached_width) - float(car_width_mm)) <= \
                WALL_CACHE_WIDTH_TOLERANCE_MM:
            result = dict(cached)
            result['source'] = source
            return result
    return None


def _cached_wall_pair(car_width_mm):
    """读取当前码垛面内的车壁坐标缓存。"""
    return _find_wall_pair_cache(
        _wall_pair_cache, car_width_mm, 'current_face_cache')


def _previous_wall_pair(car_width_mm):
    """读取上一码垛面参考；返回值不能直接作为当前面坐标。"""
    return _find_wall_pair_cache(
        _previous_wall_pair_cache, car_width_mm, 'previous_face_reference')


def _remember_current_wall_pair(car_width_mm, pair):
    """只保存当前面车壁坐标，不长期持有候选点云。"""
    _wall_pair_cache[float(car_width_mm)] = {
        'left': float(pair['left']),
        'right': float(pair['right']),
        'span': float(pair['right']) - float(pair['left']),
        'wall_error': float(pair.get('wall_error', 0.0)),
        'candidate_count': int(pair.get('candidate_count', 0)),
    }


def _infer_from_single_outer_wall(candidates, car_width_mm, source):
    """仅当当前帧只有一侧外墙大面时，按车宽推算另一侧。"""
    left_sides = [
        item for item in candidates
        if float(item['x_face']) <= -WALL_ONE_SIDE_MIN_ABS_X_M]
    right_sides = [
        item for item in candidates
        if float(item['x_face']) >= WALL_ONE_SIDE_MIN_ABS_X_M]
    if bool(left_sides) == bool(right_sides):
        return None

    width_m = float(car_width_mm) / 1000.0
    if left_sides:
        side_candidates = left_sides
        measured_side = 'left'
    else:
        side_candidates = right_sides
        measured_side = 'right'
    wall = max(
        side_candidates,
        key=lambda item: (
            float(item['y_span']) * float(item['z_span']),
            int(item['points'])))
    if measured_side == 'left':
        left = float(wall['x_face'])
        right = left + width_m
    else:
        right = float(wall['x_face'])
        left = right - width_m
    return {
        'left': left,
        'right': right,
        'span': width_m,
        'wall_error': 0.0,
        'source': source,
        'measured_side': measured_side,
        'candidate_count': len(candidates),
    }


def _refit_walls_near_reference(candidates, reference, car_width_mm):
    """以上一面为搜索中心，但车壁X坐标必须由当前帧候选重新拟合。"""
    if reference is None:
        return None

    def nearest(expected_x):
        matches = [
            item for item in candidates
            if abs(float(item['x_face']) - float(expected_x)) <=
            WALL_REFERENCE_X_TOLERANCE_M
        ]
        if not matches:
            return None
        return min(
            matches,
            key=lambda item: (
                abs(float(item['x_face']) - float(expected_x)),
                -int(item['points'])))

    left = nearest(reference['left'])
    right = nearest(reference['right'])
    if left is not None and right is not None and left is not right:
        result = {
            'left': float(left['x_face']),
            'right': float(right['x_face']),
            'source': 'previous_face_refit',
            'candidate_count': len(candidates),
        }
        result['span'] = result['right'] - result['left']
        result['wall_error'] = abs(
            result['span'] - float(car_width_mm) / 1000.0)
        if (result['span'] > 0 and
                result['wall_error'] <= ORDER_WALL_SPAN_TOLERANCE_M):
            return result
        return None

    current = [candidate for candidate in (left, right) if candidate is not None]
    inferred = _infer_from_single_outer_wall(
        current, car_width_mm, 'previous_face_single_refit')
    if inferred is not None:
        inferred['candidate_count'] = len(candidates)
    return inferred


def _resolve_global_wall_pair(pts, car_width_mm):
    """优先当前帧重测车壁，上一面坐标只用于引导弱点云重拟合。"""
    if (car_width_mm is None or not np.isfinite(car_width_mm) or
            float(car_width_mm) <= 0):
        return None
    candidates = _extract_global_wall_candidates(pts)
    measured = _select_global_wall_pair(candidates, car_width_mm)
    if measured is not None:
        measured['candidate_count'] = len(candidates)
        _remember_current_wall_pair(car_width_mm, measured)
        return measured

    strong_candidates = [
        item for item in candidates
        if float(item['y_span']) >= WALL_MIN_Y_SPAN_M]
    inferred = _infer_from_single_outer_wall(
        strong_candidates, car_width_mm, 'single_wall_inferred')
    if inferred is not None:
        _remember_current_wall_pair(car_width_mm, inferred)
        return inferred

    # 当前面已有缓存时，优先用它引导当前弱点云重拟合；当前帧完全无点
    # 时才直接复用同一面缓存。
    cached = _cached_wall_pair(car_width_mm)
    if cached is not None:
        refitted = _refit_walls_near_reference(
            candidates, cached, car_width_mm)
        if refitted is not None:
            refitted['source'] = 'current_face_refit'
            _remember_current_wall_pair(car_width_mm, refitted)
            return refitted
        cached['candidate_count'] = len(candidates)
        return cached

    # 进入新面后不能直接复用上一面坐标；只有当前帧在附近确实找到车壁
    # 大面时，才接受重新拟合后的当前坐标。
    refitted = _refit_walls_near_reference(
        candidates, _previous_wall_pair(car_width_mm), car_width_mm)
    if refitted is not None:
        _remember_current_wall_pair(car_width_mm, refitted)
    return refitted


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

    # U/V 投影只与本帧局部坐标系有关。后面每个“深度峰×窗口”都会复用，
    # 因此在这里对整帧只计算一次，避免 _find_tilt_pair_at_depth 重复投影。
    all_relative_xy = pts[:, :2] - origin_xy
    all_u_values = all_relative_xy @ u_axis
    all_v_values = all_relative_xy @ v_axis
    u_values = all_u_values[roi_mask]
    v_values = all_v_values[roi_mask]
    all_mask = (
        (pts[:, 0] >= TILT_PASS_X[0]) & (pts[:, 0] < TILT_PASS_X[1]) &
        (pts[:, 1] >= PASS_Y[0]) & (pts[:, 1] <= PASS_Y[1]) &
        (pts[:, 2] >= PASS_Z[0]) & (pts[:, 2] <= PASS_Z[1])
    )
    # U 窗口以全垛中心为基准，避免某一层只有单侧点云时窗口偏移。
    u_center = float(np.median(all_u_values[all_mask]))
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

    # 所有深度窗口共有同一个 U/Z 范围。提前裁成基础点集，后续每个窗口
    # 只需做一次 V 范围筛选；边界条件与原逻辑保持完全一致。
    tilt_base_mask = (
        (all_u_values >= u_min) & (all_u_values < u_max) &
        (pts[:, 2] >= z_min) & (pts[:, 2] < z_max)
    )

    return {
        'origin_xy': origin_xy,
        'u_axis': u_axis,
        'v_axis': v_axis,
        'u_range': (float(u_min), float(u_max)),
        'v_fronts': v_fronts,
        'yaw_deg': _normalize_xz_line_angle(
            math.degrees(math.atan2(u_axis[1], u_axis[0]))),
        'roi': roi,
        'tilt_base_pts': pts[tilt_base_mask],
        'tilt_base_u': all_u_values[tilt_base_mask],
        'tilt_base_v': all_v_values[tilt_base_mask],
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
        """把 U-Z 投影像素换算到带边距、Z向上显示的画布坐标。"""
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

    title = (f"TILT CANDIDATE | Tilted box: {result['tilt_deg']:.1f} deg | "
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
        f"TILT CANDIDATE | 箱体倾斜 {result['tilt_deg']:.1f}deg | "
        f"自估yaw={result['projection_yaw_deg']:+.1f}deg | "
        + " | ".join(descriptions))
    _show_geometries(title, geometries, width=1100, height=760)


def _find_tilt_pair_at_depth(pts, frame, z_min, z_max,
                             front_depth, v_front):
    """在一个独立的局部V深度窗口内寻找最佳斜边对。"""
    u_min, u_max = frame['u_range']
    v_back = v_front - front_depth

    # _fit_tilt_projection_frame 已缓存相同 U/Z 范围内的点及其 U/V 坐标。
    # 保留兼容分支，便于旧测试或外部调试代码传入未带缓存的 frame。
    base_pts = frame.get('tilt_base_pts')
    base_u = frame.get('tilt_base_u')
    base_v = frame.get('tilt_base_v')
    if base_pts is None or base_u is None or base_v is None:
        relative_xy = pts[:, :2] - frame['origin_xy']
        u_values = relative_xy @ frame['u_axis']
        v_values = relative_xy @ frame['v_axis']
        base_mask = (
            (u_values >= u_min) & (u_values < u_max) &
            (pts[:, 2] >= z_min) & (pts[:, 2] < z_max)
        )
        base_pts = pts[base_mask]
        base_u = u_values[base_mask]
        base_v = v_values[base_mask]

    depth_mask = (base_v >= v_back) & (base_v <= v_front)
    front_pts = base_pts[depth_mask]
    if len(front_pts) < 100:
        return None

    nx = int(math.ceil((u_max - u_min) / TILT_GRID_M))
    nz = int(math.ceil((z_max - z_min) / TILT_GRID_M))
    image = np.zeros((nz, nx), dtype=np.uint8)
    front_u = base_u[depth_mask]
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
            """把局部 U-Z 线端点恢复到雷达 XYZ 坐标。"""
            xy = (frame['origin_xy'] + point_uz[0] * frame['u_axis'] +
                  line_v * frame['v_axis'])
            return np.array([xy[0], xy[1], point_uz[1]], dtype=float)

        lines.append({
            **item,
            'p0_uz': p0_uz,
            'p1_uz': p1_uz,
            'line_v': float(line_v),
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
        'origin_xy': frame['origin_xy'],
        'u_axis': frame['u_axis'],
        'v_axis': frame['v_axis'],
    }
    if view:
        _show_tilt_result(frame['roi'], front_pts, result)
    return result


def _build_tilt_outer_candidate(tilt_result, yaw_offset_deg=0.0):
    """把倾斜箱侧边沿局部 U 方向平移到箱体最外侧，生成竖直虚拟候选面。

    倾斜箱的侧边不再满足后续 ``±X`` 法向过滤，容易从候选面中消失。这里选取
    两条检出边中更接近竖直方向的一条作为侧边，再根据倾斜轮廓位于垛面中心的
    左侧或右侧，把它压到整个轮廓的最外侧 U。最后执行与测宽点云相同的 J1
    偏航补偿，使虚拟面的 x 坐标能直接参与现有候选面配对。
    """
    lines = tilt_result.get('lines') or []
    if len(lines) < 2:
        return None

    origin_xy = np.asarray(tilt_result.get('origin_xy'), dtype=float)
    u_axis = np.asarray(tilt_result.get('u_axis'), dtype=float)
    v_axis = np.asarray(tilt_result.get('v_axis'), dtype=float)
    if (origin_xy.shape != (2,) or u_axis.shape != (2,) or
            v_axis.shape != (2,) or not np.isfinite(
                np.concatenate((origin_xy, u_axis, v_axis))).all()):
        return None

    all_u = np.asarray([
        float(point[0])
        for line in lines
        for point in (line['p0_uz'], line['p1_uz'])
    ])
    u_center = float(np.mean(tilt_result['u_range']))
    outline_center_u = float(np.mean(all_u))
    if outline_center_u < u_center:
        side_name = '左外侧'
        outer_u = float(np.min(all_u))
    else:
        side_name = '右外侧'
        outer_u = float(np.max(all_u))

    # 与竖直方向夹角最小的线作为倾斜侧边；平移后保留它原有的高度跨度。
    side_line = max(lines, key=lambda line: abs(float(line['angle_deg'])))
    line_v = float(side_line['line_v'])
    outer_xy = origin_xy + outer_u * u_axis + line_v * v_axis
    z0 = float(side_line['p0_uz'][1])
    z1 = float(side_line['p1_uz'][1])
    sample_count = max(
        MIN_CLUSTER_PTS,
        int(math.ceil(abs(z1 - z0) / max(TILT_GRID_M, 1e-6))) + 1)
    virtual_pts = np.column_stack((
        np.full(sample_count, outer_xy[0]),
        np.full(sample_count, outer_xy[1]),
        np.linspace(z0, z1, sample_count),
    ))

    if yaw_offset_deg and J1_AXIS_XY is not None:
        virtual_pts = _derotate_about_j1(
            virtual_pts, J1_DEROTATE_SIGN * yaw_offset_deg, J1_AXIS_XY)
    x_face = float(np.median(virtual_pts[:, 0]))
    if not np.isfinite(x_face):
        return None
    return {
        'k': 'tilt_outer',
        'pts': virtual_pts,
        'x_face': x_face,
        'source': 'tilt_outer',
        'side_name': side_name,
        'outer_u': outer_u,
    }


def _measure_pair_occupancy(evidence_pts, x_left, x_right):
    """检查两个候选侧面之间是否存在连续的箱体点云。

    返回 (coverage, point_density, point_count, bin_count)：coverage 为沿 X
    方向被箱体点覆盖的 bin 比例。evidence_pts 由调用方优先传入法向
    接近 ±Y 的箱子正面点；正面点不足时才传入靠雷达的原始点云薄层。
    """
    gap = float(x_right - x_left)
    margin = min(PAIR_OCCUPANCY_SIDE_MARGIN_M, gap * 0.1)
    inner_left = float(x_left + margin)
    inner_right = float(x_right - margin)
    inner_width = inner_right - inner_left
    if inner_width <= 1e-6:
        return 0.0, 0.0, 0, 0

    occupancy_pts = evidence_pts[
        (evidence_pts[:, 0] >= inner_left) &
        (evidence_pts[:, 0] <= inner_right)
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


def _pair_contains_box(
        coverage, density, reference_coverage, reference_density):
    """判断候选面之间是否已有箱体，兼顾稀疏点云和局部箱体覆盖。"""
    density_reliable = (
        density >= reference_density * PAIR_OCCUPANCY_MIN_REL_DENSITY)
    full_coverage = coverage >= PAIR_OCCUPANCY_MIN_COVERAGE
    partial_coverage = (
        reference_coverage >= PAIR_OCCUPANCY_MIN_COVERAGE and
        coverage >= PAIR_OCCUPANCY_MIN_PARTIAL_COVERAGE
    )
    return density_reliable and (full_coverage or partial_coverage)


def _resolve_pair_occupancy_region(x_left, x_right, z_min, z_max):
    """按检测层高度H/候选宽度W计算异常物体检查区，坐标及尺寸均为米。

    底部避让=min(0.20H, 0.50W, 80mm)，防止固定忽略半层漏掉横倒箱体。
    X方向沿用现有侧面自身避让；此区域不改变侧面测宽坐标。
    只按几何设置容差，不能凭本帧点云把大块底部占用自动解释为合法支撑。
    """
    bounds = np.asarray((x_left, x_right, z_min, z_max), dtype=float)
    if not np.isfinite(bounds).all():
        return None
    x_left, x_right, z_min, z_max = map(float, bounds)
    width = x_right - x_left
    height = z_max - z_min
    if width <= 1e-6 or height <= 1e-6:
        return None
    bottom_margin = min(
        height * PAIR_OCCUPANCY_BOTTOM_HEIGHT_RATIO,
        width * PAIR_OCCUPANCY_BOTTOM_WIDTH_RATIO,
        PAIR_OCCUPANCY_BOTTOM_MAX_M)
    side_margin = min(PAIR_OCCUPANCY_SIDE_MARGIN_M, width * 0.1)
    check_height = height - bottom_margin
    window_height = min(
        check_height,
        max(PAIR_OCCUPANCY_WINDOW_MIN_M,
            min(height * PAIR_OCCUPANCY_WINDOW_HEIGHT_RATIO,
                width * PAIR_OCCUPANCY_WINDOW_WIDTH_RATIO)))
    return {
        'x_min': x_left + side_margin,
        'x_max': x_right - side_margin,
        'z_min': z_min + bottom_margin,
        'z_max': z_max,
        'width_m': width,
        'height_m': height,
        'height_width_ratio': height / width,
        'bottom_margin_m': bottom_margin,
        'window_height_m': window_height,
    }


def _measure_pair_occupancy_by_height(
        evidence_pts, x_left, x_right, z_min, z_max):
    """统计全高、自适应检查区及重叠高度窗口的占用，保留分层诊断。

    检查区覆盖底部小容差以上的整个高度；局部窗口防止低矮障碍的点数
    被整层平均密度稀释。不同窗口的密度按宽度和高度归一化后比较。
    """
    region = _resolve_pair_occupancy_region(x_left, x_right, z_min, z_max)
    if region is not None:
        evidence_pts = evidence_pts[
            (evidence_pts[:, 2] >= z_min) & (evidence_pts[:, 2] <= z_max)]
    overall = _measure_pair_occupancy(evidence_pts, x_left, x_right)
    result = {
        'coverage': overall[0], 'density': overall[1],
        'count': overall[2], 'bins': overall[3],
        'check_coverage': overall[0], 'check_density': overall[1],
        'check_count': overall[2], 'check_bins': overall[3],
        'row_coverages': (), 'region': region, 'height_windows': (),
    }
    if region is None:
        return result

    z_row_count = max(
        2, int(math.ceil((z_max - z_min) / PAIR_OCCUPANCY_Z_BIN_M)))
    z_edges = np.linspace(z_min, z_max, z_row_count + 1)
    row_coverages = []
    for row_index in range(z_row_count):
        row_mask = (
            (evidence_pts[:, 2] >= z_edges[row_index]) &
            (evidence_pts[:, 2] <= z_edges[row_index + 1])
        )
        row_coverage, _, _, _ = _measure_pair_occupancy(
            evidence_pts[row_mask], x_left, x_right)
        row_coverages.append(float(row_coverage))

    check_pts = evidence_pts[evidence_pts[:, 2] >= region['z_min']]
    checked = _measure_pair_occupancy(check_pts, x_left, x_right)
    # 相邻窗口重叠至少50%，避免障碍刚好跨窗口边界而被拆散。
    window_height = region['window_height_m']
    travel = max(0.0, region['z_max'] - region['z_min'] - window_height)
    window_count = max(1, int(math.ceil(travel / (window_height * 0.5))) + 1)
    height_windows = []
    for start_z in np.linspace(region['z_min'], region['z_min'] + travel, window_count):
        end_z = float(start_z + window_height)
        window_pts = check_pts[
            (check_pts[:, 2] >= start_z) & (check_pts[:, 2] <= end_z)]
        measured = _measure_pair_occupancy(window_pts, x_left, x_right)
        inner_pts = window_pts[
            (window_pts[:, 0] >= region['x_min']) &
            (window_pts[:, 0] <= region['x_max'])]
        z_span = float(np.ptp(inner_pts[:, 2])) if len(inner_pts) else 0.0
        required_z_span = min(
            PAIR_OCCUPANCY_WINDOW_ZSPAN_MAX_M,
            window_height * PAIR_OCCUPANCY_WINDOW_ZSPAN_RATIO)
        height_windows.append({
            'z_min': float(start_z), 'z_max': end_z,
            'coverage': measured[0], 'area_density': measured[1] / window_height,
            'count': measured[2], 'bins': measured[3],
            'z_span_m': z_span, 'height_reliable': z_span >= required_z_span,
        })
    result.update(
        check_coverage=checked[0], check_density=checked[1],
        check_count=checked[2], check_bins=checked[3],
        row_coverages=tuple(row_coverages), height_windows=tuple(height_windows))
    return result


def _find_occupied_height_window(occupancy, reference_coverage, reference_density):
    """返回自适应区内检出连续箱面的局部窗口；薄片和稀疏点不单独否决。"""
    matches = [
        window for window in occupancy.get('height_windows', ())
        if window['height_reliable'] and _pair_contains_box(
            window['coverage'], window['area_density'],
            reference_coverage, reference_density)
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: (item['coverage'], item['area_density']))


def _resolve_order_target_x(
        candidates, target_y_mm, car_width_mm, expected_width_mm,
        wall_pair=None):
    """用实测左右车壁把订单 Y 区间线性映射到点云 X 区间。

    订单 Y=0 位于点云右壁，Y 增大方向与点云 X 增大方向相反，因此映射时
    必须从右壁向左计算；返回值仍按点云 X 从小到大排列为 left/right。
    优先使用外部传入的全局车壁；未传入时保留当前层常规法向聚类的
    兼容逻辑。两种情况都要求车壁间距接近订单车宽；条件不满足时返回
    None，禁止仅凭理论位置创建候选面。
    """
    values = (target_y_mm, car_width_mm, expected_width_mm)
    if any(value is None or not np.isfinite(value) for value in values):
        return None
    target_y_mm = float(target_y_mm)
    car_width_mm = float(car_width_mm)
    expected_width_mm = float(expected_width_mm)
    if (car_width_mm <= 0 or expected_width_mm <= 0 or
            target_y_mm < 0 or
            target_y_mm + expected_width_mm > car_width_mm + 1e-6):
        return None

    wall_source = 'current_layer'
    if wall_pair is None:
        real_candidates = sorted(
            (
                candidate for candidate in candidates
                if candidate.get('source') == 'normal_cluster'
            ),
            key=lambda item: item['x_face'])
        selected_pair = None
        wall_error = float('inf')
        expected_wall_span_m = car_width_mm / 1000.0
        for left_index, left in enumerate(real_candidates):
            for right in real_candidates[left_index + 1:]:
                wall_span = float(right['x_face'] - left['x_face'])
                error = abs(wall_span - expected_wall_span_m)
                if wall_span > 0 and error < wall_error:
                    selected_pair = (left, right)
                    wall_error = error
        if (selected_pair is None or
                wall_error > ORDER_WALL_SPAN_TOLERANCE_M):
            return None
        wall_left_x, wall_right_x = sorted(
            (float(selected_pair[0]['x_face']),
             float(selected_pair[1]['x_face'])))
    else:
        wall_left_x, wall_right_x = sorted(
            (float(wall_pair['left']), float(wall_pair['right'])))
        wall_error = abs(
            (wall_right_x - wall_left_x) - car_width_mm / 1000.0)
        if wall_error > ORDER_WALL_SPAN_TOLERANCE_M:
            return None
        wall_source = wall_pair.get('source', 'external')

    wall_span = wall_right_x - wall_left_x
    target_left = (
        wall_right_x -
        (target_y_mm + expected_width_mm) /
        car_width_mm * wall_span)
    target_right = (
        wall_right_x -
        target_y_mm / car_width_mm * wall_span)
    return {
        'left': target_left,
        'right': target_right,
        'wall_left': wall_left_x,
        'wall_right': wall_right_x,
        'wall_error': wall_error,
        'wall_source': wall_source,
    }


def _is_reliable_order_edge(candidate, pass_z, log_callback=None):
    """补边/虚拟兜底共用质量门槛；侧面不再追加层高比例过滤。

    侧面仅保留原基础薄片过滤及点数要求。正面突变的高度质量检查保持不变，
    防止取消侧面二次高度门槛时，把水平薄片也放宽成正面边界。
    """
    if candidate.get('source') not in (
            'normal_cluster', 'tilt_outer', 'front_gap_edge'):
        return False
    layer_z_span = max(0.0, float(pass_z[1]) - float(pass_z[0]))
    required_z_span = MIN_CLUSTER_ZSPAN_M

    points = np.asarray(candidate.get('pts', ()))
    point_count = len(points)
    z_span = (
        float(np.ptp(points[:, 2]))
        if points.ndim == 2 and points.shape[1] >= 3 and point_count > 0
        else 0.0
    )
    if candidate.get('source') == 'front_gap_edge':
        required_z_span = min(
            FRONT_GAP_EDGE_MAX_ZSPAN_M,
            max(FRONT_GAP_EDGE_MIN_ZSPAN_M,
                layer_z_span * FRONT_GAP_EDGE_MIN_ZSPAN_RATIO))
    reliable = (
        point_count >= ORDER_TARGET_EDGE_MIN_PTS and
        z_span >= required_z_span)
    if not reliable:
        report = log_callback if log_callback is not None else _dbg
        report(
            f"订单边界候选 k={candidate.get('k')} 质量不足："
            f"点数={point_count}（要求≥{ORDER_TARGET_EDGE_MIN_PTS}），"
            f"zspan={z_span * 1000:.0f}mm"
            f"（要求≥{required_z_span * 1000:.0f}mm）；"
            "不能阻止正面补边或订单虚拟兜底")
    return reliable


def _add_missing_order_edge_candidate(
        candidates, target_range, pass_y, pass_z):
    """正面补边后仍仅缺一侧可靠边界时，补可被占用检查否决的虚拟边界。"""
    if target_range is None:
        return None
    reliable_candidates = [
        candidate for candidate in candidates
        if _is_reliable_order_edge(candidate, pass_z)
    ]

    def _nearest(edge_x):
        """返回距订单边界最近的可靠候选及其绝对误差。"""
        if not reliable_candidates:
            return None, float('inf')
        candidate = min(
            reliable_candidates,
            key=lambda item: abs(float(item['x_face']) - edge_x))
        return candidate, abs(float(candidate['x_face']) - edge_x)

    left_match, left_error = _nearest(target_range['left'])
    right_match, right_error = _nearest(target_range['right'])
    left_found = left_error <= ORDER_TARGET_EDGE_MATCH_M
    right_found = right_error <= ORDER_TARGET_EDGE_MATCH_M
    # 两侧都实测到时不补；两侧都没有时也不能只靠订单凭空构造缺口。
    if left_found == right_found:
        return None

    side_name = 'right' if left_found else 'left'
    x_face = float(target_range[side_name])
    z_values = np.linspace(
        float(pass_z[0]), float(pass_z[1]), max(MIN_CLUSTER_PTS, 12))
    virtual_pts = np.column_stack((
        np.full(len(z_values), x_face),
        np.full(len(z_values), float(sum(pass_y) * 0.5)),
        z_values,
    ))
    candidate = {
        'k': f'order_virtual_{side_name}',
        'pts': virtual_pts,
        'x_face': x_face,
        'source': 'order_virtual',
        'side_name': side_name,
        'matched_k': (
            left_match['k'] if left_found else right_match['k']),
    }
    candidates.append(candidate)
    return candidate


def _detect_front_gap_edge_candidate(
        front_gap_pts, target_range, side_name, pass_y, pass_z):
    """用当前层正面薄层点云寻找订单边界附近的有点/空白突变。

    ``target_range`` 内侧是待放置缺口：left边界应表现为左侧有箱、右侧为空；
    right边界则应表现为左侧为空、右侧有箱。这里只在订单边界附近搜索，并且
    使用与占用复核相同的宽高比自适应Z范围，只排除底部小容差。
    边界必须紧邻具有多高度支持的连续箱面，不能仅凭远处箱面的点数差定位。
    最后用边缘分箱内逐高度观测的边缘坐标中位数消除分箱起点误差。
    """
    if target_range is None or side_name not in ('left', 'right'):
        return None
    points = np.asarray(front_gap_pts)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
        return None

    z_min, z_max = map(float, pass_z)
    if (not np.isfinite(z_min) or not np.isfinite(z_max) or
            z_max <= z_min + 1e-6):
        return None
    region = _resolve_pair_occupancy_region(
        target_range['left'], target_range['right'], z_min, z_max)
    if region is None:
        return None
    check_points = points[
        (points[:, 2] >= region['z_min']) & (points[:, 2] <= region['z_max'])]
    if len(check_points) < FRONT_GAP_EDGE_MIN_BOX_PTS:
        return None

    target_x = float(target_range[side_name])
    window_bins = max(
        2, int(math.ceil(
            FRONT_GAP_EDGE_WINDOW_M / FRONT_GAP_EDGE_BIN_M)))
    # 搜索区外再留一个完整判定窗口，确保每个候选边界两侧都有足量数据。
    x_min = (
        target_x - FRONT_GAP_EDGE_SEARCH_M -
        window_bins * FRONT_GAP_EDGE_BIN_M)
    x_max = (
        target_x + FRONT_GAP_EDGE_SEARCH_M +
        window_bins * FRONT_GAP_EDGE_BIN_M)
    bin_count = max(
        window_bins * 2 + 1,
        int(math.ceil((x_max - x_min) / FRONT_GAP_EDGE_BIN_M)))
    edges = x_min + np.arange(bin_count + 1) * FRONT_GAP_EDGE_BIN_M
    hist, _ = np.histogram(check_points[:, 0], bins=edges)
    z_edges = region['z_min'] + np.arange(
        int(math.ceil((region['z_max'] - region['z_min']) /
                      FRONT_GAP_EDGE_Z_BIN_M)) + 1) * FRONT_GAP_EDGE_Z_BIN_M
    xz_hist, _, _ = np.histogram2d(
        check_points[:, 0], check_points[:, 2], bins=(edges, z_edges))
    height_support = np.count_nonzero(xz_hist, axis=1)
    supported = ((hist >= FRONT_GAP_EDGE_MIN_PTS_PER_BIN) &
                 (height_support >= FRONT_GAP_EDGE_MIN_Z_BINS))
    required_z_span = min(
        FRONT_GAP_EDGE_MAX_ZSPAN_M,
        max(FRONT_GAP_EDGE_MIN_ZSPAN_M,
            (z_max - z_min) * FRONT_GAP_EDGE_MIN_ZSPAN_RATIO))

    matches = []
    for edge_index in range(window_bins, len(hist) - window_bins + 1):
        edge_x = float(edges[edge_index])
        edge_error = abs(edge_x - target_x)
        if edge_error > FRONT_GAP_EDGE_SEARCH_M + 1e-9:
            continue
        left_hist = hist[edge_index - window_bins:edge_index]
        right_hist = hist[edge_index:edge_index + window_bins]
        if side_name == 'left':
            box_hist, gap_hist = left_hist, right_hist
            box_x_range = (edge_x - FRONT_GAP_EDGE_WINDOW_M, edge_x)
            box_support = supported[edge_index - window_bins:edge_index][::-1]
            gap_support = supported[edge_index:edge_index + window_bins]
            adjacent_gap_supported = supported[edge_index]
            boundary_bin = edge_index - 1
        else:
            gap_hist, box_hist = left_hist, right_hist
            box_x_range = (edge_x, edge_x + FRONT_GAP_EDGE_WINDOW_M)
            box_support = supported[edge_index:edge_index + window_bins]
            gap_support = supported[edge_index - window_bins:edge_index]
            adjacent_gap_supported = supported[edge_index - 1]
            boundary_bin = edge_index

        # 从空白直接进入连续箱面。旧60%窗口覆盖允许前方空40mm，甚至把
        # 局部杂点与远处箱面拼成边界；这里禁止中间隔着空白的“提前边界”。
        if (adjacent_gap_supported or
                not np.all(box_support[:FRONT_GAP_EDGE_MIN_RUN_BINS])):
            continue

        box_coverage = float(np.count_nonzero(box_support) / len(box_hist))
        # 边缘形状按多高度支持判断，避免两条薄片扫线遮住真实空白。
        # 密度比仍使用全部原始点，且后续独立的区间占用检查不删点、不放宽。
        raw_gap_coverage = float(np.count_nonzero(
            gap_hist >= FRONT_GAP_EDGE_MIN_PTS_PER_BIN) / len(gap_hist))
        gap_coverage = float(np.count_nonzero(gap_support) / len(gap_hist))
        box_count = int(box_hist.sum())
        gap_count = int(gap_hist.sum())
        density_ratio = box_count / float(max(gap_count, 1))
        if (box_coverage < FRONT_GAP_EDGE_MIN_BOX_COVERAGE or
                gap_coverage > FRONT_GAP_EDGE_MAX_GAP_COVERAGE or
                box_count < FRONT_GAP_EDGE_MIN_BOX_PTS or
                density_ratio < FRONT_GAP_EDGE_MIN_DENSITY_RATIO):
            continue

        boundary_points = check_points[
            (check_points[:, 0] >= edges[boundary_bin]) &
            (check_points[:, 0] < edges[boundary_bin + 1])]
        row_ids = np.minimum(
            ((boundary_points[:, 2] - region['z_min']) /
             FRONT_GAP_EDGE_Z_BIN_M).astype(int), len(z_edges) - 2)
        row_edges = [
            float(np.quantile(boundary_points[row_ids == row, 0],
                              .9 if side_name == 'left' else .1))
            for row in np.unique(row_ids)
        ]
        if len(row_edges) < FRONT_GAP_EDGE_MIN_Z_BINS:
            continue
        refined_x = float(np.median(row_edges))
        refined_error = abs(refined_x - target_x)
        if refined_error > FRONT_GAP_EDGE_SEARCH_M + 1e-9:
            continue
        box_left, box_right = box_x_range
        candidate_points = points[
            (points[:, 0] >= box_left) & (points[:, 0] <= box_right)]
        z_span = float(np.ptp(candidate_points[:, 2])) if len(candidate_points) else 0.
        if (len(candidate_points) < ORDER_TARGET_EDGE_MIN_PTS or
                z_span < required_z_span):
            _dbg(
                f"正面突变候选质量不足：目标{side_name} x={refined_x:+.3f}m，"
                f"点数={len(candidate_points)}，zspan={z_span * 1000:.0f}mm，"
                f"要求点数≥{ORDER_TARGET_EDGE_MIN_PTS}、"
                f"zspan≥{required_z_span * 1000:.0f}mm")
            continue

        # 几何连续性优先于点数，避免一条密集扫描线拉偏边界。
        contrast = box_count - gap_count
        matches.append({
            'score': (-box_coverage, gap_coverage, refined_error, -contrast),
            'x_face': refined_x,
            'grid_edge_x': edge_x,
            'height_support_bins': len(row_edges),
            'candidate_points': candidate_points,
            'edge_error': refined_error,
            'box_coverage': box_coverage,
            'gap_coverage': gap_coverage,
            'raw_gap_coverage': raw_gap_coverage,
            'box_count': box_count,
            'gap_count': gap_count,
            'density_ratio': density_ratio,
            'box_x_range': box_x_range,
        })

    if not matches:
        return None
    best = min(matches, key=lambda item: item['score'])
    _dbg(
        f"正面连续边缘：目标{side_name}，分箱边界={best['grid_edge_x']:+.3f}m，"
        f"逐高度边缘中位数={best['x_face']:+.3f}m，"
        f"高度支持={best['height_support_bins']}格，"
        f"连续箱面≥{FRONT_GAP_EDGE_MIN_RUN_BINS * FRONT_GAP_EDGE_BIN_M * 1000:.0f}mm")

    return {
        'k': f'front_gap_{side_name}',
        'pts': best['candidate_points'],
        'x_face': best['x_face'],
        'source': 'front_gap_edge',
        'side_name': side_name,
        'edge_error': best['edge_error'],
        'box_coverage': best['box_coverage'],
        'gap_coverage': best['gap_coverage'],
        'raw_gap_coverage': best['raw_gap_coverage'],
        'box_count': best['box_count'],
        'gap_count': best['gap_count'],
        'density_ratio': best['density_ratio'],
        'check_region': region,
        'edge_method': 'height_supported_contiguous_edge',
        'grid_edge_x': best['grid_edge_x'],
        'height_support_bins': best['height_support_bins'],
    }


def _add_front_gap_edge_candidates(
        candidates, front_gap_pts, target_range, pass_y, pass_z,
        log_callback=None):
    """目标侧面缺失或质量不足时用正面突变补齐，返回新增候选列表。

    车壁和可靠侧面/倾斜候选保持优先。弱侧面不能阻止正面提取；补边成功后
    弱候选只保留作占用参照，不再抢占最终结果。正面检不出时，阶梯模式才
    继续尝试订单虚拟边界兜底。
    """
    if target_range is None:
        return []
    report = log_callback if log_callback is not None else _dbg

    wall_positions = (
        float(target_range['wall_left']),
        float(target_range['wall_right']),
    )

    def _is_wall_candidate(candidate):
        """保留订单映射所用车壁，避免以墙体厚度突变替代车壁平面。"""
        if candidate.get('source') != 'normal_cluster':
            return False
        x_face = float(candidate['x_face'])
        return any(
            abs(x_face - wall_x) <= 1e-6
            for wall_x in wall_positions)

    added = []
    for side_name in ('left', 'right'):
        target_x = float(target_range[side_name])
        # 缺口贴车壁时，车壁平面就是该侧真实边界；墙体表面的点云突变受厚度和
        # 扫描角影响，不应替代已经标定出的车壁位置。
        if any(
                abs(target_x - wall_x) <= 1e-6
                for wall_x in wall_positions):
            continue

        side_candidates = [
            existing for existing in candidates
            if existing.get('source') in ('normal_cluster', 'tilt_outer')
            and not _is_wall_candidate(existing)
            and abs(float(existing['x_face']) - target_x) <=
            ORDER_TARGET_EDGE_MATCH_M
        ]
        reliable_sides = [
            existing for existing in side_candidates
            if _is_reliable_order_edge(existing, pass_z, report)
        ]
        if reliable_sides:
            selected_side = min(
                reliable_sides,
                key=lambda item: abs(float(item['x_face']) - target_x))
            _dbg(
                f"目标{side_name}优先使用侧面候选 "
                f"k={selected_side.get('k')} "
                f"x={float(selected_side['x_face']):+.3f}m，"
                "正面缺口突变仅作缺边兜底，本侧不参与")
            continue

        reason = ('side_candidate_weak' if side_candidates
                  else 'side_candidate_missing')
        report(
            f"目标{side_name}侧面{'质量不足' if side_candidates else '缺失'}，"
            "先尝试正面突变补边")
        candidate = _detect_front_gap_edge_candidate(
            front_gap_pts, target_range, side_name, pass_y, pass_z)
        if candidate is not None:
            candidate['fallback_reason'] = reason
            for existing in side_candidates:
                existing['superseded_by'] = candidate['k']
            candidates.append(candidate)
            added.append(candidate)
        else:
            report(
                f"目标{side_name}正面补边未找到可靠突变边界；"
                "若满足阶梯模式及单侧实测条件，再尝试订单虚拟边界兜底")
    return added


def _format_width_method(left, right, gap_mm, expected_width_mm):
    """按最终选中边界标明测宽来源，而非按生成过哪些候选来判断。"""
    names = {
        'normal_cluster': '侧面聚类',
        'tilt_outer': '倾斜外侧补偿',
        'front_gap_edge': '正面突变补边',
        'order_virtual': '订单虚拟边界',
    }
    sources = {left.get('source'), right.get('source')}
    if 'order_virtual' in sources:
        method, note = '虚拟边界兜底', '包含订单推算边界，非双侧实测'
    elif 'front_gap_edge' in sources:
        method, note = '正面补边测宽', '采用点云实测突变边界'
    elif 'tilt_outer' in sources:
        method, note = '倾斜外侧补偿测宽', '包含倾斜外侧补偿边界'
    else:
        method, note = '侧面测宽', '双侧侧面聚类间距'
    return (
        f"[STACK-WIDTH] 测宽方式={method}；"
        f"点云X左边界={names.get(left.get('source'), '未知')}"
        f"(k={left['k']},x={left['x_face']:+.3f}m)；"
        f"点云X右边界={names.get(right.get('source'), '未知')}"
        f"(k={right['k']},x={right['x_face']:+.3f}m)；"
        f"测量宽度={gap_mm}mm，理论抓宽={float(expected_width_mm):.0f}mm；"
        f"{note}，status=1")


def _compute_width(pc1, pc2, view=None, yaw_offset_deg=0.0,
                   rel_top_h=None, box_h=None, expected_width_mm=None,
                   box_width_mm=None, log_callback=None, box_type=None,
                   target_y_mm=None, car_width_mm=None,
                   stair_step_mode=False, detection_width_mm=None,
                   detection_target_y_mm=None):
    """
    合并双雷达点云，测量左右侧面总宽度（mm）。

    view=None 时在调用时读取顶部 VIEW 开关（避免默认参数在定义时被绑死）。
    yaw_offset_deg: 拍照位相对正对的偏航角(度)。雷达绕机器人 J1 轴摆动导致点云偏航，
                    需将点云转回正对系再测量。
    rel_top_h / box_h: 当前抓"顶面距地板的高度"和箱子竖向高度(米)。两者都给定时，
                    内部自标定地板 + 实测箱顶，把直通滤波 Z 锁定到"当前行"隔离相邻层；
                    否则用全局 PASS_Z。
    expected_width_mm / box_width_mm: 当前抓理论总宽度和单箱宽度。理论宽度保留给
                    机器人报文及细支首层兜底，不因检测参考宽度而改变。
    detection_width_mm: 仅供点云候选筛选的真实缺口宽度；未提供时兼容使用
                    expected_width_mm。候选面间距必须落在检测参考宽度
                    ±1.5×单箱宽度范围内。
    target_y_mm / car_width_mm: 当前抓在订单车宽方向的起点和车厢宽度(mm)。
                    所有模式都会在左右车壁可靠时换算目标X区间并约束候选缺口。
    detection_target_y_mm: 真实缺口在订单车宽方向的起点；未提供时使用target_y_mm。
                    所有模式都优先使用法向聚类得到的箱体侧面；目标某侧没有
                    侧面候选时，才用正面点云有点/空白突变补边。阶梯模式两种
                    实测边界都缺失且自适应检查区为空时，才允许订单虚拟边界参与。
    stair_step_mode: 混装阶梯垛模式；启用 X-Z 分层占用检查。
    box_type: 当前抓箱型；首位为2表示细支烟箱。细支烟箱第一层计算失败时，
                    可兜底返回理论宽度+一个单箱宽度；阶梯目标已明确占用时禁止此兜底。
    log_callback: 检测到箱体倾斜及二次复核结果的日志回调；在线模式传主节点 logger.warning。

    策略：
      0. 偏航补偿 + （可选）当前行 Z 锁定
      1. 原始点云自适应 U-Z 轮廓倾斜检测；命中后生成最外侧虚拟候选面继续复核
      2. 直通滤波
      3. 法向量滤波保留 ±x 侧面点
      4. DBSCAN 聚类，所有有效候选面两两组合
      5. 按订单目标位置筛选候选，并保留检测参考宽度 ± 1.5个单箱宽度的组合
      6. 优先采用订单边界附近的可靠实测侧面；目标某侧缺失或只有弱候选时，
         才用正面点云突变补齐（贴车壁的边界始终使用实测车壁）
      7. 检查两面之间的点云覆盖率/密度；阶梯模式按宽高比确定范围并分高度复核
      8. 在剩余空缺口中取最接近真实缺口宽度者，宽度 = 右面 x - 左面 x
    """
    if view is None:
        view = VIEW

    slim_first_layer_fallback = (
        str(box_type)[:1] == '2' and
        rel_top_h is not None and box_h is not None and
        np.isfinite(rel_top_h) and np.isfinite(box_h) and
        box_h > 0 and rel_top_h <= box_h * 1.5 and
        expected_width_mm is not None and box_width_mm is not None and
        np.isfinite(expected_width_mm) and np.isfinite(box_width_mm) and
        expected_width_mm > 0 and box_width_mm > 0
    )
    
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

    # 车壁标定保留偏航补偿后的全局点云引用。后续缺口测宽仍会按当前面Y和
    # 当前层Z裁剪，但车壁检测不再跟随这个局部窗口。
    global_wall_pts = pts

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
    tilt_message_base = None
    tilt_candidate = None
    if tilt_result is not None:
        line_text = "；".join(
            f"斜边{index}角度={line['angle_deg']:+.1f}°、长度={line['length_m'] * 1000:.0f}mm"
            for index, line in enumerate(tilt_result['lines'], start=1)
        )
        tilt_message_base = (
            f"垛面异常：检测到箱体倾斜，估计倾斜角={tilt_result['tilt_deg']:.1f}°；"
            f"自估垛面yaw={tilt_result['projection_yaw_deg']:+.1f}°；"
            f"{line_text}；端点间距={tilt_result['endpoint_gap_m'] * 1000:.0f}mm；"
            f"局部V窗口={tilt_result['front_depth_m'] * 1000:.0f}mm")
        tilt_candidate = _build_tilt_outer_candidate(
            tilt_result, yaw_offset_deg=yaw_offset_deg)
        if tilt_candidate is not None:
            _dbg(
                f"{tilt_message_base}；倾斜面平移到{tilt_candidate['side_name']}，"
                f"虚拟候选面x={tilt_candidate['x_face']:+.3f}m，继续距离复核")
        else:
            _dbg(f"{tilt_message_base}；倾斜外侧候选面生成失败，继续用常规候选面复核")

    def _report_tilt_result(success, detail):
        """统一输出倾斜候选复核后的最终状态，并安全调用外部日志回调。"""
        if tilt_message_base is None:
            return
        message = (
            f"{tilt_message_base}；倾斜候选面复核"
            f"{'通过' if success else '失败'}：{detail}，"
            f"返回 status={'1' if success else '2'}")
        _dbg(message)
        if log_callback is not None:
            try:
                log_callback(message)
            except Exception as exc:
                _dbg(f"倾斜异常日志回调失败：{type(exc).__name__}: {exc}")

    def _report_stair_step(detail):
        """把补边过程及最终测宽方式写入在线主日志，不依赖 DEBUG。"""
        _dbg(detail)
        if log_callback is not None:
            try:
                log_callback(detail)
            except Exception as exc:
                _dbg(f"阶梯缺口日志回调失败：{type(exc).__name__}: {exc}")

    def _failure_result(detail, *, allow_slim_fallback=True):
        """普通失败返回None；未被明确占用否决时允许细支首层估算兜底。"""
        if not slim_first_layer_fallback or not allow_slim_fallback:
            _report_tilt_result(False, detail)
            _report_stair_step(
                f"[STACK-WIDTH] 测宽方式=无有效结果；{detail}，status=2")
            return None
        fallback_mm = int(round(
            float(expected_width_mm) + float(box_width_mm)))
        _report_stair_step(
            f"[STACK-WIDTH] 测宽方式=细支首层估算兜底；"
            f"测量宽度={fallback_mm}mm，非点云实测；原因={detail}，status=1")
        if tilt_message_base is not None:
            message = (
                f"{tilt_message_base}；倾斜候选面复核失败：{detail}；"
                f"细支烟箱第一层启用测宽兜底，测量值={fallback_mm}mm，"
                f"返回 status=1")
        else:
            message = (
                f"细支烟箱第一层测宽失败兜底：{detail}；"
                f"理论宽度={float(expected_width_mm):.0f}mm，"
                f"单箱宽度={float(box_width_mm):.0f}mm，"
                f"测量值={fallback_mm}mm，返回 status=1")
        _dbg(message)
        if log_callback is not None:
            try:
                log_callback(message)
            except Exception as exc:
                _dbg(f"细支首层兜底日志回调失败：{type(exc).__name__}: {exc}")
        return fallback_mm

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
        return _failure_result(
            f"直通滤波后点数不足({len(filtered_pts)})")

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
    normals = np.asarray(_pcd.normals)
    nx = np.abs(normals[:, 0])
    side_pts = pts[nx > np.cos(np.radians(NORMAL_ANGLE_DEG))]
    if len(side_pts) < DBSCAN_MIN_POINTS:
        _dbg(f"计算失败：法向滤波后侧面点不足({len(side_pts)})，无法聚类")
        return _failure_result(
            f"法向滤波后侧面点不足({len(side_pts)})")

    # 两候选侧面之间是否已有箱子，优先使用法向接近 ±Y 的箱子正面点判断。
    # 这里复用侧面提取已经算好的法向，不增加一次法向估计。
    ny = np.abs(normals[:, 1])
    front_face_pts = pts[
        ny > np.cos(np.radians(PAIR_OCCUPANCY_FRONT_NORMAL_ANGLE_DEG))]
    if len(front_face_pts) >= PAIR_OCCUPANCY_MIN_FRONT_PTS:
        occupancy_evidence_pts = front_face_pts
        occupancy_mode = "正面法向"
        _dbg(
            f"区间占用检查：使用箱子正面点云({len(front_face_pts)}点，"
            f"法向与±Y夹角≤{PAIR_OCCUPANCY_FRONT_NORMAL_ANGLE_DEG}°)")
    else:
        # 极稀疏或反光严重时保留原方案兜底，避免整帧没有正面点便把所有区间判空。
        y_front = pass_y[1]
        occupancy_evidence_pts = pts[
            (pts[:, 1] >= y_front - PAIR_OCCUPANCY_FRONT_DEPTH_M) &
            (pts[:, 1] <= y_front)
        ]
        occupancy_mode = f"原始点云前沿{PAIR_OCCUPANCY_FRONT_DEPTH_M * 1000:.0f}mm兜底"
        _dbg(
            f"区间占用检查：箱子正面点不足"
            f"({len(front_face_pts)}<{PAIR_OCCUPANCY_MIN_FRONT_PTS})，"
            f"使用{occupancy_mode}({len(occupancy_evidence_pts)}点)")

    # 正面边界判断不依赖法向：侧面缺失时，扫描当前面前沿200mm薄层中
    # X方向的点云有/无突变。相比仅用±Y法向，能保留斜扫线和纸箱表面波纹点。
    front_gap_y_min = max(
        float(pass_y[0]),
        float(pass_y[1]) - PAIR_OCCUPANCY_FRONT_DEPTH_M)
    front_gap_pts = pts[
        (pts[:, 1] >= front_gap_y_min) &
        (pts[:, 1] <= float(pass_y[1]))]
    _dbg(
        f"正面缺口边界检查：前沿Y=[{front_gap_y_min:.3f},"
        f"{float(pass_y[1]):.3f}]m，共{len(front_gap_pts)}点")

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
            'source': 'normal_cluster',
        })

    # 倾斜侧面通常会被 ±X 法向过滤丢掉。把压到轮廓最外侧的竖直虚拟面加入
    # 同一个候选列表，后续仍沿用检测参考宽度和区间占用检查，不单独放宽阈值。
    if tilt_candidate is not None:
        candidates.append(tilt_candidate)
        _dbg(
            f"加入倾斜外侧候选面：k={tilt_candidate['k']} "
            f"x={tilt_candidate['x_face']:+.3f}m "
            f"点数={len(tilt_candidate['pts'])}")

    if (expected_width_mm is None or box_width_mm is None or
            not np.isfinite(expected_width_mm) or not np.isfinite(box_width_mm) or
            expected_width_mm <= 0 or box_width_mm <= 0):
        _dbg(f"计算失败：理论宽度或单箱宽度无效（理论={expected_width_mm}, 单箱={box_width_mm}）")
        return _failure_result(
            f"理论宽度或单箱宽度无效（理论={expected_width_mm}, 单箱={box_width_mm}）")

    if detection_width_mm is None:
        detection_width_mm = float(expected_width_mm)
    elif (not np.isfinite(detection_width_mm) or
          float(detection_width_mm) <= 0):
        _dbg(f"计算失败：检测参考宽度无效（{detection_width_mm}）")
        return _failure_result(
            f"检测参考宽度无效（{detection_width_mm}）")
    else:
        detection_width_mm = float(detection_width_mm)

    order_position_required = (
        detection_target_y_mm is not None or target_y_mm is not None)
    if detection_target_y_mm is None:
        detection_target_y_mm = target_y_mm
    elif not np.isfinite(detection_target_y_mm):
        _dbg(f"计算失败：检测目标Y无效（{detection_target_y_mm}）")
        return _failure_result(
            f"检测目标Y无效（{detection_target_y_mm}）")
    else:
        detection_target_y_mm = float(detection_target_y_mm)

    # 车壁使用独立的全深度、全高度大面检测；当前帧失败时可复用本订单缓存，
    # 再失败才允许由唯一单侧大车壁结合订单车宽推算另一侧。缺口侧面本身仍
    # 使用当前面、当前层局部点云，不改变现有测宽范围。
    global_wall_pair = _resolve_global_wall_pair(
        global_wall_pts, car_width_mm)
    target_range = _resolve_order_target_x(
        candidates, detection_target_y_mm, car_width_mm,
        detection_width_mm, wall_pair=global_wall_pair)
    if target_range is None:
        if order_position_required:
            message = (
                "订单目标位置无法确认：当前面双侧、单侧及局部车壁"
                "均未提取成功，禁止直接复用上一面坐标")
            _report_stair_step(message)
            _dbg(f"计算失败：{message}")
            return _failure_result(message)
        if stair_step_mode:
            message = (
                "阶梯缺口模式：订单位置无效或左右车壁标定失败，"
                "不创建虚拟边界，退回常规候选组合")
            _report_stair_step(message)
        else:
            _dbg(
                "订单位置约束：目标参数无效或左右车壁标定失败，"
                "退回宽度候选组合")
    else:
        mode_name = "阶梯缺口模式" if stair_step_mode else "订单位置约束"
        wall_source_names = {
            'global_measured': '当前帧全局大面实测',
            'current_face_refit': '当前面历史位置引导当前帧弱点重拟合',
            'current_face_cache': '同一面历史车壁缓存',
            'previous_face_refit': '上一面位置引导当前帧双侧弱点重拟合',
            'previous_face_single_refit': '上一面位置引导当前帧单侧弱点重拟合',
            'single_wall_inferred': '当前帧单侧大面＋订单车宽推算',
            'current_layer': '当前层局部候选',
        }
        wall_source = target_range.get('wall_source', 'current_layer')
        if wall_source == 'current_layer':
            _remember_current_wall_pair(
                car_width_mm, {
                    'left': target_range['wall_left'],
                    'right': target_range['wall_right'],
                    'wall_error': target_range['wall_error'],
                })
        wall_message = (
            f"[STACK-WALL] 来源={wall_source_names.get(wall_source, wall_source)}；"
            f"车壁=[{target_range['wall_left']:+.3f},"
            f"{target_range['wall_right']:+.3f}]m；"
            f"跨度={(target_range['wall_right'] - target_range['wall_left']) * 1000:.0f}mm；"
            f"订单车宽={float(car_width_mm):.0f}mm；"
            f"误差={target_range['wall_error'] * 1000:.0f}mm")
        if (global_wall_pair is not None and
                global_wall_pair.get('measured_side') is not None):
            wall_message += (
                f"；仅{global_wall_pair.get('measured_side')}侧为点云实测，"
                "另一侧非实测")
        _report_stair_step(wall_message)
        _dbg(
            f"{mode_name}：车壁=[{target_range['wall_left']:+.3f},"
            f"{target_range['wall_right']:+.3f}]m，"
            f"检测目标Y=[{float(detection_target_y_mm):.0f},"
            f"{float(detection_target_y_mm) + float(detection_width_mm):.0f}]mm "
            f"映射X=[{target_range['left']:+.3f},"
            f"{target_range['right']:+.3f}]m")
        front_gap_candidates = _add_front_gap_edge_candidates(
            candidates, front_gap_pts, target_range, pass_y, pass_z,
            log_callback=_report_stair_step)
        for front_candidate in front_gap_candidates:
            message = (
                f"{mode_name}：目标{front_candidate['side_name']}"
                f"侧面候选{'质量不足' if front_candidate['fallback_reason'] == 'side_candidate_weak' else '缺失'}，"
                f"使用正面点云突变兜底边界 "
                f"x={front_candidate['x_face']:+.3f}m，"
                f"订单边界偏差={front_candidate['edge_error'] * 1000:.0f}mm，"
                f"缺口侧覆盖={front_candidate['gap_coverage'] * 100:.0f}%，"
                f"原始缺口侧覆盖={front_candidate['raw_gap_coverage'] * 100:.0f}%，"
                f"箱体侧覆盖={front_candidate['box_coverage'] * 100:.0f}%，"
                f"点数={front_candidate['gap_count']}/"
                f"{front_candidate['box_count']}；"
                f"方法=多高度支持的连续正面边缘，"
                f"分箱边界={front_candidate['grid_edge_x']:+.3f}m，"
                f"高度支持={front_candidate['height_support_bins']}格；"
                "按逐高度边缘中位数参与测宽")
            _report_stair_step(message)
        if view:
            _show_front_gap_diagnostics(
                pts, front_gap_pts, target_range,
                front_gap_candidates, pass_y, pass_z)
        if stair_step_mode:
            virtual_candidate = _add_missing_order_edge_candidate(
                candidates, target_range, pass_y, pass_z)
            if virtual_candidate is not None:
                message = (
                    f"阶梯缺口模式：目标{virtual_candidate['side_name']}边界缺失，"
                    f"由实测候选 k={virtual_candidate['matched_k']} 配合订单位置，"
                    f"加入虚拟候选面 x={virtual_candidate['x_face']:+.3f}m；"
                    "后续仍须通过宽高比自适应区域及局部高度窗口占用检查")
                _report_stair_step(message)
    height_aware_occupancy = stair_step_mode and target_range is not None
    if height_aware_occupancy:
        target_check_region = _resolve_pair_occupancy_region(
            target_range['left'], target_range['right'], pass_z[0], pass_z[1])
        if target_check_region is not None:
            _report_stair_step(
                "阶梯自适应占用区域："
                f"H={target_check_region['height_m'] * 1000:.0f}mm，"
                f"W={target_check_region['width_m'] * 1000:.0f}mm，"
                f"H/W={target_check_region['height_width_ratio']:.3f}，"
                f"底部避让={target_check_region['bottom_margin_m'] * 1000:.0f}mm，"
                f"目标检查Z=[{target_check_region['z_min']:+.3f},"
                f"{target_check_region['z_max']:+.3f}]m，"
                f"局部窗口高度={target_check_region['window_height_m'] * 1000:.0f}mm；"
                "逐候选按实测间距复算区域")

    if len(candidates) < 2:
        _dbg(f"计算失败：有效候选面不足2个（当前{len(candidates)}个）")
        return _failure_result(
            f"有效候选面不足2个（当前{len(candidates)}个）")

    gap_tolerance_mm = (
        float(box_width_mm) * PAIR_WIDTH_TOLERANCE_BOXES)
    min_gap_mm = max(
        0.0, float(detection_width_mm) - gap_tolerance_mm)
    max_gap_mm = float(detection_width_mm) + gap_tolerance_mm
    candidates.sort(key=lambda item: item['x_face'])
    valid_pairs = []
    occupancy_reference_pairs = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1:]:
            gap_float = (right['x_face'] - left['x_face']) * 1000.0
            width_in_range = min_gap_mm <= gap_float <= max_gap_mm
            target_error = None
            target_in_range = True
            if target_range is not None:
                left_error = abs(
                    float(left['x_face']) - target_range['left'])
                right_error = abs(
                    float(right['x_face']) - target_range['right'])
                target_error = left_error + right_error
                target_in_range = (
                    left_error <= ORDER_TARGET_PAIR_EDGE_M and
                    right_error <= ORDER_TARGET_PAIR_EDGE_M)
            superseded = bool(left.get('superseded_by') or right.get('superseded_by'))
            selectable = width_in_range and target_in_range and not superseded
            z_diff_ = abs(float(left['pts'][:, 2].mean()) -
                          float(right['pts'][:, 2].mean()))
            # target_y 只限制最终选择；所有宽度合理的候选都参与占用统计，
            # 否则目标候选会拿自身当密度参照，稀疏空缺口容易被误判为有箱。
            if width_in_range and height_aware_occupancy:
                occupancy = _measure_pair_occupancy_by_height(
                    occupancy_evidence_pts,
                    left['x_face'], right['x_face'],
                    pass_z[0], pass_z[1])
                decision_coverage = occupancy['check_coverage']
                decision_density = occupancy['check_density']
            elif width_in_range:
                measured_occupancy = _measure_pair_occupancy(
                    occupancy_evidence_pts,
                    left['x_face'], right['x_face'])
                occupancy = {
                    'coverage': measured_occupancy[0],
                    'density': measured_occupancy[1],
                    'count': measured_occupancy[2],
                    'bins': measured_occupancy[3],
                    'check_coverage': measured_occupancy[0],
                    'check_density': measured_occupancy[1],
                    'check_count': measured_occupancy[2],
                    'check_bins': measured_occupancy[3],
                    'row_coverages': (),
                }
                decision_coverage = occupancy['coverage']
                decision_density = occupancy['density']
            else:
                occupancy = {
                    'coverage': 0.0, 'density': 0.0,
                    'count': 0, 'bins': 0,
                    'check_coverage': 0.0, 'check_density': 0.0,
                    'check_count': 0, 'check_bins': 0,
                    'row_coverages': (),
                }
                decision_coverage = 0.0
                decision_density = 0.0
            coverage_has_box = (
                decision_coverage >= PAIR_OCCUPANCY_MIN_COVERAGE)
            target_text = (
                f" 订单边界误差={target_error * 1000:.0f}mm "
                f"{'位置满足' if target_in_range else '位置不满足'}"
                if target_error is not None else "")
            height_text = ""
            if width_in_range and height_aware_occupancy:
                row_text = ",".join(
                    f"{value * 100:.0f}%"
                    for value in occupancy['row_coverages'])
                region = occupancy['region']
                height_text = (
                    f" 全高={occupancy['coverage']*100:.0f}% "
                    f"自适应区={occupancy['check_coverage']*100:.0f}% "
                    f"检查Z=[{region['z_min']:+.3f},{region['z_max']:+.3f}]m "
                    f"底部避让={region['bottom_margin_m']*1000:.0f}mm "
                    f"分层=[{row_text}]")
            _dbg(
                f"候选组合 k={left['k']}({len(left['pts'])}点,x={left['x_face']:+.3f}) - "
                f"k={right['k']}({len(right['pts'])}点,x={right['x_face']:+.3f})  "
                f"间距={gap_float:.0f}mm 允许=[{min_gap_mm:.0f},{max_gap_mm:.0f}]mm "
                f"z重心差={z_diff_*1000:.0f}mm{target_text} "
                f"内部占用[{occupancy_mode}]={decision_coverage*100:.0f}%"
                f"({occupancy['check_count'] if height_aware_occupancy else occupancy['count']}点/"
                f"{occupancy['check_bins'] if height_aware_occupancy else occupancy['bins']}bin,"
                f"密度={decision_density:.0f}点/m){height_text} "
                f"{'覆盖较高' if coverage_has_box else '低覆盖'} "
                f"{'候选有效' if selectable else '候选无效'}"
                f"{'（弱侧面已由正面补边替代，仅作占用参照）' if superseded else ''}")
            if width_in_range:
                # 有订单位置时先选最接近目标边界者；否则保持原来的宽度优先策略。
                if target_error is not None:
                    score = (
                        target_error,
                        abs(gap_float - float(detection_width_mm)),
                        -min(len(left['pts']), len(right['pts'])),
                        -(len(left['pts']) + len(right['pts'])),
                    )
                else:
                    score = (
                        abs(gap_float - float(detection_width_mm)),
                        -min(len(left['pts']), len(right['pts'])),
                        -(len(left['pts']) + len(right['pts'])),
                    )
                pair = {
                    'score': score,
                    'left': left,
                    'right': right,
                    'gap': gap_float,
                    'z_diff': z_diff_,
                    'coverage': decision_coverage,
                    'density': decision_density,
                    'occupancy': occupancy,
                }
                occupancy_reference_pairs.append(pair)
                if selectable:
                    valid_pairs.append(pair)

    occupied_target_pair_count = 0
    if valid_pairs:
        max_occupancy_coverage = max(
            item['coverage'] for item in occupancy_reference_pairs)
        max_occupancy_density = max(
            item['density'] for item in occupancy_reference_pairs)
        # 宽度合理但位置不符的箱体区仍参与参照，不能只用目标区自己作基准。
        reference_windows = [
            window for item in occupancy_reference_pairs
            for window in item['occupancy'].get('height_windows', ())
            if window['height_reliable']
        ]
        max_window_coverage = max(
            (window['coverage'] for window in reference_windows), default=0.0)
        max_window_density = max(
            (window['area_density'] for window in reference_windows), default=0.0)
        empty_gap_pairs = []
        for item in valid_pairs:
            has_box = _pair_contains_box(
                item['coverage'], item['density'],
                max_occupancy_coverage, max_occupancy_density)
            occupied_window = _find_occupied_height_window(
                item['occupancy'], max_window_coverage, max_window_density)
            has_box = has_box or occupied_window is not None
            if not has_box:
                empty_gap_pairs.append(item)
                _dbg(
                    f"候选组合 k={item['left']['k']} - "
                    f"k={item['right']['k']} 确认为空缺口："
                    f"{'自适应区' if height_aware_occupancy else ''}"
                    f"覆盖率={item['coverage']*100:.0f}%，"
                    f"密度={item['density']:.0f}点/m，"
                    f"本帧箱体参考密度={max_occupancy_density:.0f}点/m")
            else:
                occupied_target_pair_count += 1
                _dbg(
                    f"候选组合 k={item['left']['k']} - "
                    f"k={item['right']['k']} 被排除："
                    f"两面之间已有箱体（"
                    f"{'自适应区' if height_aware_occupancy else ''}"
                    f"覆盖率={item['coverage']*100:.0f}%，"
                    f"密度={item['density']:.0f}点/m，"
                    f"参考密度={max_occupancy_density:.0f}点/m）")
                if occupied_window is not None:
                    message = (
                        f"阶梯局部异常占用：候选 k={item['left']['k']} - "
                        f"k={item['right']['k']}，"
                        f"Z=[{occupied_window['z_min']:+.3f},"
                        f"{occupied_window['z_max']:+.3f}]m，"
                        f"覆盖率={occupied_window['coverage']*100:.0f}%，"
                        f"点数={occupied_window['count']}，"
                        f"实际高度跨度={occupied_window['z_span_m']*1000:.0f}mm，"
                        f"面密度={occupied_window['area_density']:.0f}点/m²，"
                        f"参照面密度={max_window_density:.0f}点/m²；排除此候选")
                    _report_stair_step(message)
        valid_pairs = empty_gap_pairs

    if not valid_pairs:
        if height_aware_occupancy and occupied_target_pair_count:
            detail = (
                f"阶梯缺口异常占用：{occupied_target_pair_count}个位置/宽度合理的候选"
                "均未通过自适应区域复核；禁止以细支首层估算值覆盖该失败，返回 status=2")
            _report_stair_step(detail)
            return _failure_result(detail, allow_slim_fallback=False)
        _dbg(
            f"计算失败：{len(candidates)}个候选面中无“间距合理且两面之间为空”的缺口，"
            f"允许间距=[{min_gap_mm:.0f}, {max_gap_mm:.0f}]mm，"
            f"有箱判据=相对密度≥{PAIR_OCCUPANCY_MIN_REL_DENSITY*100:.0f}%且"
            f"覆盖率≥{PAIR_OCCUPANCY_MIN_COVERAGE*100:.0f}%（或存在完整覆盖参照时"
            f"≥{PAIR_OCCUPANCY_MIN_PARTIAL_COVERAGE*100:.0f}%）")
        return _failure_result(
            f"{len(candidates)}个候选面中未找到允许范围"
            f"[{min_gap_mm:.0f},{max_gap_mm:.0f}]mm内的空缺口")

    selected = min(valid_pairs, key=lambda item: item['score'])

    # 侧面聚类可能在高层吸附到缺口深处、靠层底的局部箱侧面，
    # 导致前沿实际足够宽，但返回值进入机器人减速或停止档。只在最终候选
    # 为双侧面测宽且余量 <=70mm 时，强制重新检测非车壁一侧的正面突变。
    # 正面方案仍须通过订单位置、允许宽度和区间占用复核，并且按最终发送的
    # 整数毫米判断档位。正面值只有在更宽、且比侧面值更接近检测参考宽度时
    # 才替换；复核失败则保留原来的保守值。
    selected_sources = {
        selected['left'].get('source'), selected['right'].get('source')}
    side_gap_to_send_mm = int(float(selected['gap']))
    side_width_margin_mm = (
        side_gap_to_send_mm - float(expected_width_mm))
    if (selected_sources == {'normal_cluster'} and
            side_width_margin_mm <= SIDE_WIDTH_FRONT_RECHECK_MARGIN_MM and
            target_range is not None):
        alarm_level = (
            '停止档' if side_width_margin_mm < SIDE_WIDTH_STOP_MARGIN_MM
            else '减速档')
        _report_stair_step(
            f"[STACK-RECHECK] 侧面测宽进入{alarm_level}："
            f"侧面宽度={side_gap_to_send_mm}mm（发送值），"
            f"理论抓宽={float(expected_width_mm):.0f}mm，"
            f"余量={side_width_margin_mm:.0f}mm；"
            "强制使用正面突变边界复算")

        wall_positions = (
            float(target_range['wall_left']),
            float(target_range['wall_right']),
        )
        front_replacements = {}
        missing_front_sides = []
        for side_name in ('left', 'right'):
            target_x = float(target_range[side_name])
            target_is_wall = any(
                abs(target_x - wall_x) <= 1e-6
                for wall_x in wall_positions)
            if target_is_wall:
                continue
            front_candidate = _detect_front_gap_edge_candidate(
                front_gap_pts, target_range, side_name, pass_y, pass_z)
            if front_candidate is None:
                missing_front_sides.append(side_name)
            else:
                front_candidate['fallback_reason'] = 'narrow_side_recheck'
                front_replacements[side_name] = front_candidate

        if missing_front_sides:
            _report_stair_step(
                "[STACK-RECHECK] 正面突变复算未找到全部必需边界："
                f"缺失={','.join(missing_front_sides)}；"
                f"保留原侧面宽度={selected['gap']:.0f}mm")
        elif not front_replacements:
            _report_stair_step(
                "[STACK-RECHECK] 订单目标两侧均为车壁，"
                f"无可用正面突变替换的箱体边界；"
                f"保留原侧面宽度={selected['gap']:.0f}mm")
        else:
            front_left = front_replacements.get('left', selected['left'])
            front_right = front_replacements.get('right', selected['right'])
            front_gap_float = (
                float(front_right['x_face']) - float(front_left['x_face'])) * 1000.0
            front_gap_to_send_mm = int(front_gap_float)
            side_reference_error_mm = abs(
                side_gap_to_send_mm - float(detection_width_mm))
            front_reference_error_mm = abs(
                front_gap_to_send_mm - float(detection_width_mm))
            front_is_wider = front_gap_to_send_mm > side_gap_to_send_mm
            front_is_closer_to_reference = (
                front_reference_error_mm < side_reference_error_mm)
            front_left_error = abs(
                float(front_left['x_face']) - float(target_range['left']))
            front_right_error = abs(
                float(front_right['x_face']) - float(target_range['right']))
            front_geometry_valid = (
                min_gap_mm <= front_gap_float <= max_gap_mm and
                front_left_error <= ORDER_TARGET_PAIR_EDGE_M and
                front_right_error <= ORDER_TARGET_PAIR_EDGE_M)

            if height_aware_occupancy and front_geometry_valid:
                front_occupancy = _measure_pair_occupancy_by_height(
                    occupancy_evidence_pts,
                    front_left['x_face'], front_right['x_face'],
                    pass_z[0], pass_z[1])
                front_coverage = front_occupancy['check_coverage']
                front_density = front_occupancy['check_density']
            elif front_geometry_valid:
                front_measured_occupancy = _measure_pair_occupancy(
                    occupancy_evidence_pts,
                    front_left['x_face'], front_right['x_face'])
                front_occupancy = {
                    'coverage': front_measured_occupancy[0],
                    'density': front_measured_occupancy[1],
                    'count': front_measured_occupancy[2],
                    'bins': front_measured_occupancy[3],
                    'check_coverage': front_measured_occupancy[0],
                    'check_density': front_measured_occupancy[1],
                    'check_count': front_measured_occupancy[2],
                    'check_bins': front_measured_occupancy[3],
                    'row_coverages': (),
                }
                front_coverage = front_occupancy['coverage']
                front_density = front_occupancy['density']
            else:
                front_occupancy = None
                front_coverage = 0.0
                front_density = 0.0

            front_has_box = True
            occupied_front_window = None
            if front_geometry_valid:
                front_has_box = _pair_contains_box(
                    front_coverage, front_density,
                    max_occupancy_coverage, max_occupancy_density)
                if height_aware_occupancy:
                    occupied_front_window = _find_occupied_height_window(
                        front_occupancy,
                        max_window_coverage, max_window_density)
                    front_has_box = (
                        front_has_box or occupied_front_window is not None)

            if (front_geometry_valid and not front_has_box and
                    front_is_wider and front_is_closer_to_reference):
                selected = {
                    'score': selected['score'],
                    'left': front_left,
                    'right': front_right,
                    'gap': front_gap_float,
                    'z_diff': abs(
                        float(front_left['pts'][:, 2].mean()) -
                        float(front_right['pts'][:, 2].mean())),
                    'coverage': front_coverage,
                    'density': front_density,
                    'occupancy': front_occupancy,
                }
                _report_stair_step(
                    "[STACK-RECHECK] 正面突变复算通过："
                    f"边界=[{float(front_left['x_face']):+.3f},"
                    f"{float(front_right['x_face']):+.3f}]m，"
                    f"正面宽度={front_gap_to_send_mm}mm（发送值），"
                    f"比侧面宽度增加="
                    f"{front_gap_to_send_mm - side_gap_to_send_mm}mm，"
                    f"检测参考偏差={front_reference_error_mm:.0f}mm"
                    f"（侧面偏差={side_reference_error_mm:.0f}mm），"
                    f"内部占用={front_coverage * 100:.0f}%；"
                    "最终采用较宽的正面复算结果")
            else:
                reasons = []
                if not front_geometry_valid:
                    reasons.append(
                        f"位置/宽度不合法({front_gap_float:.0f}mm)")
                if front_geometry_valid and front_has_box:
                    reasons.append(
                        f"内部有占用({front_coverage * 100:.0f}%)")
                if (front_geometry_valid and not front_has_box and
                        not front_is_wider):
                    reasons.append(
                        f"发送值未宽于侧面结果({front_gap_to_send_mm}<="
                        f"{side_gap_to_send_mm}mm)")
                if (front_geometry_valid and not front_has_box and
                        front_is_wider and
                        not front_is_closer_to_reference):
                    reasons.append(
                        f"未更接近检测参考宽度"
                        f"(正面偏差={front_reference_error_mm:.0f}mm>="
                        f"侧面偏差={side_reference_error_mm:.0f}mm)")
                _report_stair_step(
                    "[STACK-RECHECK] 正面突变复算未采用："
                    f"{'；'.join(reasons)}；"
                    f"保留原侧面宽度={selected['gap']:.0f}mm")

    left = selected['left']
    right = selected['right']
    gap_float = selected['gap']
    z_diff = selected['z_diff']
    selected_coverage = selected['coverage']
    selected_density = selected['density']
    selected_occupancy = selected['occupancy']
    left_k, right_k = left['k'], right['k']
    lc, rc = left['pts'], right['pts']
    x_left_face, x_right_face = left['x_face'], right['x_face']
    gap_mm = int(gap_float)
    _report_stair_step(_format_width_method(
        left, right, gap_mm, expected_width_mm))
    _dbg(
        f"最终选中候选面 k={left_k}({len(lc)}点,x={x_left_face:+.3f}) - "
        f"k={right_k}({len(rc)}点,x={x_right_face:+.3f})  "
        f"宽度={gap_mm}mm 检测参考={float(detection_width_mm):.0f}mm "
        f"理论抓宽={float(expected_width_mm):.0f}mm "
        f"允许=[{min_gap_mm:.0f},{max_gap_mm:.0f}]mm z重心差={z_diff*1000:.0f}mm "
        f"内部占用[{occupancy_mode}]="
        f"{'自适应区' if height_aware_occupancy else ''}{selected_coverage*100:.0f}% "
        f"(全高={selected_occupancy['coverage']*100:.0f}%) "
        f"密度={selected_density:.0f}点/m")
    if height_aware_occupancy:
        region = selected_occupancy['region']
        _report_stair_step(
            f"阶梯缺口复核通过：检测目标Y={float(detection_target_y_mm):.0f}mm，"
            f"候选X=[{x_left_face:+.3f},{x_right_face:+.3f}]m，"
            f"全高占用={selected_occupancy['coverage']*100:.0f}%，"
            f"自适应区占用={selected_occupancy['check_coverage']*100:.0f}%，"
            f"检查Z=[{region['z_min']:+.3f},{region['z_max']:+.3f}]m，"
            f"底部避让={region['bottom_margin_m']*1000:.0f}mm，局部高度窗口复核通过，"
            f"测量宽度={gap_mm}mm，返回 status=1")

    clusters = {
        'left': lc,
        'right': rc,
    }

    if tilt_result is not None:
        tilt_used = (
            left.get('source') == 'tilt_outer' or
            right.get('source') == 'tilt_outer')
        _report_tilt_result(
            True,
            f"找到有效候选间距={gap_mm}mm"
            f"（允许[{min_gap_mm:.0f},{max_gap_mm:.0f}]mm，"
            f"倾斜外侧候选{'已参与' if tilt_used else '未参与最终组合'}）")

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
                   log_callback=None, view=False, box_type=None,
                   target_y_mm=None, car_width_mm=None,
                   stair_step_mode=False, detection_width_mm=None,
                   detection_target_y_mm=None):
    """测量堆叠宽度，返回 measured_mm（计算成功）或 None（计算失败/报错）。
    length 为当前抓理论总宽度；box_width_mm 为当前姿态下单箱宽度。
    两候选面间距只有落在检测参考宽度 ± 1.5×box_width_mm 范围内才算有效；
    tolerance 仅保留旧调用签名，当前候选范围由内部箱宽倍数规则决定，不参与计算。
    yaw_offset_deg: 拍照位相对正对的偏航角(度)，用于点云偏航补偿。
    rel_top_h / box_h: 当前抓顶面距地板高度和箱子竖向高度(米)，用于把测宽锁定到当前行。
    log_callback: 倾斜检出及二次复核日志回调。倾斜外侧候选能组成有效空缺口时
                  正常返回 measured_mm，否则返回 None，由主节点发送 status=2。
    box_type: 当前抓箱型。2xx细支烟箱第一层计算失败时返回
              length + box_width_mm，由主节点发送status=1。
    target_y_mm / car_width_mm: 当前抓订单Y起点和车宽(mm)；所有模式在车壁
                  可靠时均据此约束候选缺口位置。
    detection_width_mm / detection_target_y_mm: 仅供点云候选筛选使用的
                  真实缺口宽度和起点；机器人报文仍使用length和最终测量值。
    stair_step_mode: 混装阶梯缺口模式；在通用订单位置约束之外，额外支持缺失
                     边界补偿，并按宽高比自适应检查区及局部高度窗口复核占用。
    view: 在线默认 False，避免2D/3D交互窗口阻塞机器人状态返回；离线可显式开启。
    """
    return _compute_width(pc1, pc2, view=view, yaw_offset_deg=yaw_offset_deg,
                          rel_top_h=rel_top_h, box_h=box_h,
                          expected_width_mm=length, box_width_mm=box_width_mm,
                          log_callback=log_callback, box_type=box_type,
                          target_y_mm=target_y_mm,
                          car_width_mm=car_width_mm,
                          stair_step_mode=stair_step_mode,
                          detection_width_mm=detection_width_mm,
                          detection_target_y_mm=detection_target_y_mm)


def _default_save_dir():
    """定位当前 robot_process 所属工作空间的点云日志目录。

    不能取 COLCON_PREFIX_PATH 的第一项：叠加加载多个工作空间时，第一项可能是
    另一个 overlay，导致点云保存到错误的 ws。优先级为：
      1. ROBOT_PROCESS_PCD_DIR 显式配置；
      2. 当前 Python 文件实际指向的源码工作空间（兼容 --symlink-install）；
      3. ament 索引中 robot_process 自身的安装前缀；
      4. 当前模块旁的 pcd_logs 兜底目录。
    """
    configured = os.environ.get('ROBOT_PROCESS_PCD_DIR', '').strip()
    if configured:
        return os.path.abspath(os.path.expanduser(configured))

    module_path = os.path.realpath(__file__)
    parts = module_path.split(os.sep)
    # 源码布局：<ws>/src/robot_process/robot_process/stacking_detection/...
    for index in range(len(parts) - 1):
        if parts[index] == 'src' and parts[index + 1] == 'robot_process':
            ws_root = os.sep.join(parts[:index]) or os.sep
            return os.path.join(ws_root, 'log', 'robot_process', 'pcd_logs')

    # 普通 colcon 安装布局：<ws>/install/robot_process。通过包自身前缀定位，
    # 不受其他已 source 工作空间在环境变量中的排列顺序影响。
    try:
        from ament_index_python.packages import get_package_prefix
        package_prefix = os.path.realpath(get_package_prefix('robot_process'))
        prefix_parts = package_prefix.split(os.sep)
        install_indices = [
            index for index, name in enumerate(prefix_parts)
            if name == 'install'
        ]
        if install_indices:
            install_index = install_indices[-1]
            ws_root = os.sep.join(prefix_parts[:install_index]) or os.sep
            return os.path.join(ws_root, 'log', 'robot_process', 'pcd_logs')
    except Exception as exc:
        _dbg(f"无法从ament索引定位robot_process工作空间，使用模块目录兜底：{exc}")

    return os.path.join(os.path.dirname(module_path), 'pcd_logs')

_DEFAULT_SAVE_DIR = _default_save_dir()

def save_point_clouds(
        pc1, pc2, save_dir=_DEFAULT_SAVE_DIR, file_name=None):
    """保存双雷达合并点云并返回实际文件路径。

    file_name 未提供时沿用时间戳命名；在线任务可在采集前预分配文件名，
    使触发参数日志、检测结果和最终保存的点云能够按同一名称关联。
    """
    os.makedirs(save_dir, exist_ok=True)
    if file_name is None:
        file_name = f"merged_{time.strftime('%Y%m%d_%H%M%S')}.pcd"
    if os.path.basename(file_name) != file_name:
        raise ValueError(f"点云文件名不能包含目录: {file_name}")
    if not file_name.lower().endswith('.pcd'):
        raise ValueError(f"点云文件名必须以.pcd结尾: {file_name}")
    path = os.path.join(save_dir, file_name)
    saved = o3d.io.write_point_cloud(
        path, pc1 + pc2, write_ascii=False)
    if not saved:
        raise OSError(f"Open3D写入点云失败: {path}")
    _dbg(f'点云已保存至：{path}')
    return path


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
    _FILENAME = 'merged_20260904/merged_20260904_185735.pcd'

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
    measured = _compute_width(
        pcd, empty,
        yaw_offset_deg=3.9,
        view=True,
        rel_top_h=0.596,
        box_h=0.298,
        expected_width_mm=1132,
        box_width_mm=283,
        box_type=202,
        target_y_mm=592.6,
        car_width_mm=2970,
        stair_step_mode=False,
        detection_width_mm=1232.8,
        detection_target_y_mm=585.6,
    )
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
