"""车厢宽度测量纯算法（无 ROS 依赖，可离线单测）。

坐标约定（可配置）：长轴=Y(1)、宽度=X(0)、高度=Z(2)。
整车点云沿长轴等距切薄片，每片测内净宽（左右壁内表面距离），
聚合得到最大/最小宽度及其位置，差值超阈值判定为变形。
"""
import numpy as np


def measure_slice_width(slice_pts, width_idx=0, center_x=None,
                        face_percentile=5.0, min_pts=200,
                        z_idx=2, z_band=None):
    """测量单个薄片的内净宽。

    左右壁内表面是点云在宽度方向的最外侧极值：左壁取低百分位（最靠 -X，
    用 face_percentile 抗噪），右壁取高百分位（最靠 +X）。净宽=右壁-左壁。
    用百分位而非 min/max 是为了抵抗墙外杂点/噪声。返回 dict 或 None。
    """
    if slice_pts.shape[0] < min_pts:
        return None
    pts = slice_pts
    if z_band:
        m = (pts[:, z_idx] >= z_band[0]) & (pts[:, z_idx] <= z_band[1])
        pts = pts[m]
        if pts.shape[0] < min_pts:
            return None
    xs = pts[:, width_idx]
    c = float(np.median(xs)) if center_x is None else float(center_x)
    left = xs[xs < c]
    right = xs[xs >= c]
    if left.size < max(1, min_pts // 4) or right.size < max(1, min_pts // 4):
        return None
    x_left_inner = float(np.percentile(left, face_percentile))
    x_right_inner = float(np.percentile(right, 100.0 - face_percentile))
    width = x_right_inner - x_left_inner
    if width <= 0:
        return None
    return {'width': width, 'x_left': x_left_inner, 'x_right': x_right_inner,
            'n_pts': int(pts.shape[0])}


def measure_profile(points, slice_step, y_range=None, long_idx=1, width_idx=0,
                    center_x=None, face_percentile=5.0, min_pts=200,
                    z_idx=2, z_band=None):
    """沿长轴等距切片，逐片测宽，返回每片结果列表（含无效片）。"""
    ys = points[:, long_idx]
    y0 = float(ys.min()) if (not y_range or y_range[0] is None) else float(y_range[0])
    y1 = float(ys.max()) if (not y_range or y_range[1] is None) else float(y_range[1])
    n = max(1, int(np.ceil((y1 - y0) / slice_step)))
    profile = []
    for i in range(n):
        a = y0 + i * slice_step
        b = a + slice_step
        sl = points[(ys >= a) & (ys < b)]
        res = measure_slice_width(sl, width_idx, center_x, face_percentile,
                                  min_pts, z_idx, z_band)
        yc = (a + b) / 2.0
        if res is not None:
            res.update({'y': yc, 'valid': True})
            profile.append(res)
        else:
            profile.append({'y': yc, 'width': None, 'valid': False,
                            'x_left': None, 'x_right': None,
                            'n_pts': int(sl.shape[0])})
    return profile


def aggregate(profile, max_spread_mm, region_tol_mm=15.0):
    """聚合宽度曲线，返回最大/最小宽度、位置、差值与是否告警。

    并对每个有效切片打区域标签 p['region']：
      'wide'   宽度 >= w_max - region_tol（最宽区域）
      'narrow' 宽度 <= w_min + region_tol（最窄区域）
      'normal' 其余
    region_tol_mm 控制"区域"的宽容度（越大圈进的切片越多）。
    """
    valid = [p for p in profile if p['valid']]
    if not valid:
        return None
    wmax = max(valid, key=lambda p: p['width'])
    wmin = min(valid, key=lambda p: p['width'])
    spread_mm = (wmax['width'] - wmin['width']) * 1000.0
    widths_mm = [p['width'] * 1000.0 for p in valid]

    tol = region_tol_mm / 1000.0
    wide_th = wmax['width'] - tol
    narrow_th = wmin['width'] + tol
    wide_slices, narrow_slices = [], []
    for p in profile:
        if not p['valid']:
            p['region'] = 'normal'
            continue
        if p['width'] >= wide_th:
            p['region'] = 'wide'
            wide_slices.append(p)
        elif p['width'] <= narrow_th:
            p['region'] = 'narrow'
            narrow_slices.append(p)
        else:
            p['region'] = 'normal'

    def _span(slices):
        if not slices:
            return None
        ys = [s['y'] for s in slices]
        return (min(ys), max(ys))

    return {
        'w_max_mm': wmax['width'] * 1000.0, 'y_max': wmax['y'], 'max_slice': wmax,
        'w_min_mm': wmin['width'] * 1000.0, 'y_min': wmin['y'], 'min_slice': wmin,
        'spread_mm': spread_mm, 'median_mm': float(np.median(widths_mm)),
        'alert': bool(spread_mm > max_spread_mm), 'max_spread_mm': float(max_spread_mm),
        'n_valid': len(valid), 'n_total': len(profile),
        'wide_slices': wide_slices, 'narrow_slices': narrow_slices,
        'wide_span': _span(wide_slices), 'narrow_span': _span(narrow_slices),
        'region_tol_mm': float(region_tol_mm),
    }
