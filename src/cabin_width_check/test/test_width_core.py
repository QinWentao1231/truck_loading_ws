"""width_core 纯算法离线单测：构造带变形的合成车厢点云验证测宽与告警。"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cabin_width_check.width_core import measure_profile, aggregate  # noqa: E402


def _synth_cabin(length=17.0, base_w=2.46, bulge_at=8.0, bulge=0.08, n_per_m=4000):
    """合成车厢侧壁点云：长轴 y∈[0,length]，宽度 x=±w/2，
    在 bulge_at 处单侧内凹 bulge（宽度变窄），用于触发变形告警。
    含地面点（x 铺满全宽）以验证内表面提取不被内部点污染。"""
    pts = []
    ny = int(length * n_per_m)
    ys = np.random.uniform(0, length, ny)
    for y in ys:
        w = base_w - (bulge if abs(y - bulge_at) < 0.5 else 0.0)
        z = np.random.uniform(0.2, 1.8)
        # 左右壁各一点
        pts.append([-w / 2, y, z])
        pts.append([+w / 2, y, z])
    cloud = np.array(pts)
    # 地面点：x 铺满 [-base_w/2, base_w/2]，z≈0（关键回归点）
    ng = ny
    floor = np.column_stack([
        np.random.uniform(-base_w / 2, base_w / 2, ng),
        np.random.uniform(0, length, ng),
        np.random.normal(0, 0.005, ng)])
    return np.vstack([cloud, floor])


def test_profile_and_alert():
    np.random.seed(0)
    pts = _synth_cabin(bulge=0.08)
    profile = measure_profile(pts, slice_step=0.5, long_idx=1, width_idx=0,
                              min_pts=50, face_percentile=5.0)
    agg = aggregate(profile, max_spread_mm=60.0)
    assert agg is not None
    # 中位宽应≈2460mm（含地面点也不被污染——回归点）
    assert abs(agg['median_mm'] - 2460) < 30, f"中位宽异常={agg['median_mm']}"
    # 基准宽 ~2460mm，凹陷处 ~2380mm，spread 应 ≈80mm > 60 → 告警
    assert agg['w_max_mm'] > agg['w_min_mm']
    assert agg['spread_mm'] > 60.0
    assert agg['alert'] is True
    # 最窄位置应落在 bulge_at≈8m 附近
    assert abs(agg['y_min'] - 8.0) < 1.0


def test_no_alert_when_uniform():
    np.random.seed(1)
    pts = _synth_cabin(bulge=0.0)
    profile = measure_profile(pts, slice_step=0.5, long_idx=1, width_idx=0,
                              min_pts=50, face_percentile=5.0)
    agg = aggregate(profile, max_spread_mm=60.0)
    assert agg is not None
    assert agg['alert'] is False


if __name__ == '__main__':
    test_profile_and_alert()
    test_no_alert_when_uniform()
    print('width_core 单测通过')
