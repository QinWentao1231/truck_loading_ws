"""车厢宽度变形检测节点（独立运行 + RViz 实时可视化）。

工作流（分站采集，机器人物理移动，节点不控制运动）：
  1. 机器人移动到某站 → 调用服务 /cabin/capture_station 抓一帧（站号自增）
  2. 重复直到车尾
  3. 调用 /cabin/analyze → 拼接 + 逐片测宽 + 聚合判定 + 发布点云/标注 + 告警
  4. /cabin/reset 清空重来

离线测试：设参数 offline_pcd_dir 指向若干 *.pcd（按文件名排序为各站），
启动即自动拼接分析并发布，无需硬件。

所有可调量均为参数（含站间位移 station_interval_m、变形阈值 max_spread_mm）。
本包不 import 任何其它功能包，零耦合。
"""
import os
import glob
import json
import struct

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_srvs.srv import Trigger
from visualization_msgs.msg import MarkerArray

from cabin_width_check.width_core import measure_profile, aggregate
from cabin_width_check.stitching import stitch_stations
from cabin_width_check.rviz_markers import build_markers

_AXIS = {'x': 0, 'y': 1, 'z': 2}


def _load_config():
    """读 config.json：优先 share 目录（ros2 run），回退源码目录（vscode）。"""
    path = None
    try:
        from ament_index_python.packages import get_package_share_directory  # type: ignore
        path = os.path.join(get_package_share_directory('cabin_width_check'), 'config.json')
    except Exception:
        path = None
    if not path or not os.path.isfile(path):
        path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


class WidthCheckNode(Node):

    def __init__(self):
        super().__init__('cabin_width_check')
        cfg = _load_config()

        # ── 参数（config.json 为默认值，可被 ros2 param / launch 覆盖）──
        from rcl_interfaces.msg import ParameterDescriptor

        def p(name, default):
            val = cfg.get(name, default)
            # None / 空列表无法推断类型，用 dynamic_typing 声明，避免 rclpy 警告
            if val is None or (isinstance(val, list) and len(val) == 0):
                self.declare_parameter(name, val, ParameterDescriptor(dynamic_typing=True))
            else:
                self.declare_parameter(name, val)
            return self.get_parameter(name).value

        self.station_interval_m = float(p('station_interval_m', 3.0))
        self.slice_step_m       = float(p('slice_step_m', 0.5))
        self.max_spread_mm      = float(p('max_spread_mm', 60.0))
        self.region_tol_mm      = float(p('region_tol_mm', 15.0))
        self.min_pts_per_slice  = int(p('min_pts_per_slice', 200))
        self.face_percentile    = float(p('face_percentile', 5.0))
        self.use_icp_refine     = bool(p('use_icp_refine', False))
        self.icp_max_corr       = float(p('icp_max_corr', 0.1))
        self.long_axis          = str(p('long_axis', 'y'))
        self.width_axis         = str(p('width_axis', 'x'))
        self.long_axis_sign     = float(p('long_axis_sign', 1.0))
        self.center_x_m         = p('center_x_m', None)
        self.slice_range_m      = list(p('slice_range_m', []) or [])
        self.z_band_m           = list(p('z_band_m', []) or [])
        self.cabin_frame        = str(p('cabin_frame', 'cabin'))
        self.topic1             = str(p('topic1', '/lidar_points1'))
        self.topic2             = str(p('topic2', '/lidar_points2'))
        self.cloud_topic        = str(p('cloud_topic', '/cabin/cloud'))
        self.marker_topic       = str(p('marker_topic', '/cabin/markers'))
        self.republish_period_s = float(p('republish_period_s', 1.0))
        self.offline_pcd_dir    = str(p('offline_pcd_dir', ''))

        self.long_idx  = _AXIS.get(self.long_axis, 1)
        self.width_idx = _AXIS.get(self.width_axis, 0)

        # ── 状态 ──
        self._latest = {self.topic1: None, self.topic2: None}
        self.stations = []          # 每站 Nx3 ndarray
        self._last_cloud_msg = None
        self._last_markers = None

        # ── 订阅 / 发布 / 服务 ──
        self.create_subscription(PointCloud2, self.topic1,
                                 lambda m: self._on_cloud(self.topic1, m), 10)
        self.create_subscription(PointCloud2, self.topic2,
                                 lambda m: self._on_cloud(self.topic2, m), 10)
        self.pub_cloud  = self.create_publisher(PointCloud2, self.cloud_topic, 1)
        self.pub_marker = self.create_publisher(MarkerArray, self.marker_topic, 1)
        self.create_service(Trigger, '/cabin/capture_station', self.srv_capture)
        self.create_service(Trigger, '/cabin/analyze', self.srv_analyze)
        self.create_service(Trigger, '/cabin/reset', self.srv_reset)
        self.create_timer(self.republish_period_s, self._republish)

        self.get_logger().info(
            f"车厢宽度检测节点已启动 | 站间={self.station_interval_m}m "
            f"切片={self.slice_step_m}m 阈值={self.max_spread_mm}mm "
            f"长轴={self.long_axis} 宽度轴={self.width_axis}")

        if self.offline_pcd_dir:
            self._run_offline()

    # ── 点云缓冲 ──
    @staticmethod
    def _msg_to_np(msg):
        return np.array([[x, y, z] for x, y, z in
                         point_cloud2.read_points(msg, ('x', 'y', 'z'), skip_nans=True)],
                        dtype=float)

    def _on_cloud(self, topic, msg):
        self._latest[topic] = self._msg_to_np(msg)

    # ── 服务 ──
    def srv_capture(self, req, resp):
        c1, c2 = self._latest[self.topic1], self._latest[self.topic2]
        if c1 is None and c2 is None:
            resp.success = False
            resp.message = '尚未收到任何雷达点云'
            return resp
        parts = [c for c in (c1, c2) if c is not None and c.size]
        station = np.vstack(parts)
        self.stations.append(station)
        msg = f'已采集站 {len(self.stations)}（{station.shape[0]} 点）'
        self.get_logger().info(msg)
        resp.success = True
        resp.message = msg
        return resp

    def srv_reset(self, req, resp):
        self.stations.clear()
        resp.success = True
        resp.message = '已清空所有站点'
        self.get_logger().info(resp.message)
        return resp

    def srv_analyze(self, req, resp):
        if not self.stations:
            resp.success = False
            resp.message = '没有已采集的站点，先调用 /cabin/capture_station'
            return resp
        try:
            ok, info = self._analyze(self.stations)
            resp.success = ok
            resp.message = info
        except Exception as e:
            self.get_logger().error(f'分析失败：{e}')
            resp.success = False
            resp.message = f'分析失败：{e}'
        return resp

    # ── 核心分析 + 发布 ──
    def _analyze(self, stations):
        merged = stitch_stations(
            stations, self.station_interval_m, long_idx=self.long_idx,
            sign=self.long_axis_sign, use_icp=self.use_icp_refine,
            icp_max_corr=self.icp_max_corr)
        if merged.size == 0:
            return False, '拼接后点云为空'

        y_range = self.slice_range_m if len(self.slice_range_m) == 2 else None
        z_band = self.z_band_m if len(self.z_band_m) == 2 else None
        profile = measure_profile(
            merged, self.slice_step_m, y_range=y_range,
            long_idx=self.long_idx, width_idx=self.width_idx,
            center_x=self.center_x_m, face_percentile=self.face_percentile,
            min_pts=self.min_pts_per_slice, z_band=z_band)
        agg = aggregate(profile, self.max_spread_mm, region_tol_mm=self.region_tol_mm)
        if agg is None:
            return False, '所有切片均无效（点数不足？检查坐标轴/阈值参数）'

        # 发布点云（按宽度偏差着色）与标注
        cloud_msg = self._build_cloud(merged, profile, agg)
        markers = build_markers(self.cabin_frame, profile, agg,
                                long_idx=self.long_idx, width_idx=self.width_idx)
        self._last_cloud_msg = cloud_msg
        self._last_markers = markers
        self.pub_cloud.publish(cloud_msg)
        self.pub_marker.publish(markers)

        line = (f"宽度检测 | 中位={agg['median_mm']:.0f}mm "
                f"max={agg['w_max_mm']:.0f}@{agg['y_max']:.1f}m "
                f"min={agg['w_min_mm']:.0f}@{agg['y_min']:.1f}m "
                f"spread={agg['spread_mm']:.0f}mm 阈值={agg['max_spread_mm']:.0f}mm "
                f"有效片={agg['n_valid']}/{agg['n_total']}")
        if agg['alert']:
            self.get_logger().warning(f"⚠ 变形超限！{line}")
        else:
            self.get_logger().info(f"✓ 正常 {line}")
        return True, line

    def _build_cloud(self, points, profile, agg):
        """构造带 RGB 的 PointCloud2，按所在切片宽度偏差着色（绿→红）。"""
        rgb = self._color_by_deviation(points, profile, agg)
        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        data = []
        for (x, y, z), (r, g, b) in zip(points, rgb):
            rgb_int = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
            rgb_f = struct.unpack('f', struct.pack('I', rgb_int))[0]
            data.append([float(x), float(y), float(z), rgb_f])
        header = self._make_header()
        return point_cloud2.create_cloud(header, fields, data)

    def _color_by_deviation(self, points, profile, agg):
        """按所在切片的区域着色：最宽区=红、最窄区=蓝、正常=灰。"""
        valid = [p for p in profile if p['valid']]
        ys_centers = np.array([p['y'] for p in valid])
        region = np.array([p['region'] for p in valid])
        yv = points[:, self.long_idx]
        idx = np.clip(np.searchsorted(ys_centers, yv), 0, len(valid) - 1)
        reg = region[idx]
        rgb = np.full((points.shape[0], 3), 0.55)   # 默认灰
        rgb[reg == 'wide']   = (1.0, 0.15, 0.15)    # 最宽区 红
        rgb[reg == 'narrow'] = (0.15, 0.35, 1.0)    # 最窄区 蓝
        return rgb

    def _make_header(self):
        from std_msgs.msg import Header
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        h.frame_id = self.cabin_frame
        return h

    def _republish(self):
        if self._last_cloud_msg is not None:
            self._last_cloud_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_cloud.publish(self._last_cloud_msg)
        if self._last_markers is not None:
            self.pub_marker.publish(self._last_markers)

    # ── 离线回放 ──
    def _run_offline(self):
        files = sorted(glob.glob(os.path.join(self.offline_pcd_dir, '*.pcd')))
        if not files:
            self.get_logger().error(f'离线目录无 pcd 文件：{self.offline_pcd_dir}')
            return
        self.get_logger().info(f'离线加载 {len(files)} 站：{[os.path.basename(f) for f in files]}')
        for f in files:
            self.stations.append(self._load_pcd(f))
        ok, info = self._analyze(self.stations)
        self.get_logger().info(f'离线分析结果：{info}')

    @staticmethod
    def _load_pcd(path):
        """优先用自带 numpy 读取器（无 open3d 依赖），失败再退回 open3d。"""
        from cabin_width_check.pcd_io import read_pcd_xyz
        try:
            return read_pcd_xyz(path)
        except Exception:
            import open3d as o3d
            return np.asarray(o3d.io.read_point_cloud(path).points, dtype=float)


def main(args=None):
    rclpy.init(args=args)
    node = WidthCheckNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
