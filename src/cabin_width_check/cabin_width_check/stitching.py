"""分站点云拼接（站间已知位移 + 可选 ICP 精化）。无 ROS 依赖。

机器人/雷达架从车头开始每隔 interval 米停一站采集，逐站按已知里程位移
平移到统一车厢系。相邻站若有重叠，可用 point-to-plane ICP 以位移为初值微调，
消除累计漂移；无 open3d 或不启用时退化为纯里程平移。
"""
import numpy as np


def stitch_stations(station_clouds, interval, long_idx=1, sign=1.0,
                    use_icp=False, icp_max_corr=0.1):
    """拼接多站点云为整车点云（Nx3）。

    station_clouds: 按采集顺序的点云列表，每个为 Nx3 ndarray。
    interval: 站间标称位移（米）。sign: 长轴推进方向(+1/-1)。
    """
    merged = []
    prev = None
    for k, cloud in enumerate(station_clouds):
        pts = np.asarray(cloud, dtype=float).copy()
        if pts.size == 0:
            continue
        off = np.zeros(3)
        off[long_idx] = sign * interval * k
        pts = pts + off
        if use_icp and prev is not None:
            pts = _icp_refine(prev, pts, icp_max_corr)
        merged.append(pts)
        prev = pts
    return np.vstack(merged) if merged else np.empty((0, 3))


def _icp_refine(target, source, max_corr):
    """用 ICP 把 source 对齐到 target（已按里程预对齐，初值取单位阵）。"""
    try:
        import open3d as o3d
    except ImportError:
        return source
    t = o3d.geometry.PointCloud()
    t.points = o3d.utility.Vector3dVector(target)
    s = o3d.geometry.PointCloud()
    s.points = o3d.utility.Vector3dVector(source)
    reg = o3d.pipelines.registration.registration_icp(
        s, t, max_corr, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    s.transform(reg.transformation)
    return np.asarray(s.points)
