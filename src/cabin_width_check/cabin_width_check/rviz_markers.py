"""构造 RViz MarkerArray：标注最大/最小宽度位置、逐片宽度阶梯与总览文字。

坐标轴映射可配置（默认 宽度=X(0)、长轴=Y(1)、高度=Z(2)）。
"""
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


def _rgba(r, g, b, a=1.0):
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


def _compose(width_val, long_val, z_val, width_idx, long_idx, z_idx):
    p = Point()
    coords = [0.0, 0.0, 0.0]
    coords[width_idx] = float(width_val)
    coords[long_idx] = float(long_val)
    coords[z_idx] = float(z_val)
    p.x, p.y, p.z = coords[0], coords[1], coords[2]
    return p


def _line(frame, mid, pts, color, width=0.01):
    m = Marker()
    m.header.frame_id = frame
    m.ns = 'cabin_width'
    m.id = mid
    m.type = Marker.LINE_LIST
    m.action = Marker.ADD
    m.scale.x = width
    m.color = color
    m.points = pts
    m.pose.orientation.w = 1.0
    return m


def _cube(frame, mid, center, scale_xyz, color):
    m = Marker()
    m.header.frame_id = frame
    m.ns = 'cabin_region'
    m.id = mid
    m.type = Marker.CUBE
    m.action = Marker.ADD
    m.pose.position = center
    m.pose.orientation.w = 1.0
    m.scale.x, m.scale.y, m.scale.z = scale_xyz
    m.color = color
    return m


def _region_box(frame, mid, slices, span, color, long_idx, width_idx, z_idx,
                z_level, height, slice_pad):
    """覆盖一个区域(最宽/最窄)的半透明立方体，沿长轴跨 span，横跨壁间宽度。"""
    if not slices or span is None:
        return None
    x_left = min(s['x_left'] for s in slices)
    x_right = max(s['x_right'] for s in slices)
    long_len = (span[1] - span[0]) + slice_pad
    long_c = (span[0] + span[1]) / 2
    width_c = (x_left + x_right) / 2
    scale = [0.0, 0.0, 0.0]
    scale[width_idx] = max(x_right - x_left, 0.05)
    scale[long_idx] = max(long_len, slice_pad)
    scale[z_idx] = height
    center = _compose(width_c, long_c, z_level + height / 2, width_idx, long_idx, z_idx)
    return _cube(frame, mid, center, scale, color)


def _text(frame, mid, pos, text, color, height=0.15):
    m = Marker()
    m.header.frame_id = frame
    m.ns = 'cabin_width_text'
    m.id = mid
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD
    m.scale.z = height
    m.color = color
    m.pose.position = pos
    m.pose.orientation.w = 1.0
    m.text = text
    return m


def build_markers(frame, profile, agg, long_idx=1, width_idx=0, z_level=0.0,
                  region_height=2.6, slice_pad=0.5):
    """根据宽度曲线与聚合结果构造 MarkerArray。
    region_height: 区域立方体高度(m)；slice_pad: 区域沿长轴两端外扩(m，约一个切片)。"""
    z_idx = ({0, 1, 2} - {long_idx, width_idx}).pop()
    arr = MarkerArray()

    # 先清空旧标注，避免残留
    clr = Marker()
    clr.header.frame_id = frame
    clr.action = Marker.DELETEALL
    arr.markers.append(clr)

    if agg is None:
        return arr

    mid = 0
    # 逐片宽度阶梯（灰色），形成内净宽剖面
    step_pts = []
    for p in profile:
        if not p['valid']:
            continue
        step_pts.append(_compose(p['x_left'], p['y'], z_level, width_idx, long_idx, z_idx))
        step_pts.append(_compose(p['x_right'], p['y'], z_level, width_idx, long_idx, z_idx))
    if step_pts:
        arr.markers.append(_line(frame, mid, step_pts, _rgba(0.6, 0.6, 0.6, 0.6)))
        mid += 1

    # 最宽/最窄区域半透明立方体（红/蓝），沿长轴覆盖整段区域
    box_w = _region_box(frame, mid, agg.get('wide_slices'), agg.get('wide_span'),
                        _rgba(1, 0.1, 0.1, 0.18), long_idx, width_idx, z_idx,
                        z_level, region_height, slice_pad)
    if box_w is not None:
        arr.markers.append(box_w)
        mid += 1
    box_n = _region_box(frame, mid, agg.get('narrow_slices'), agg.get('narrow_span'),
                        _rgba(0.1, 0.35, 1, 0.18), long_idx, width_idx, z_idx,
                        z_level, region_height, slice_pad)
    if box_n is not None:
        arr.markers.append(box_n)
        mid += 1

    # 最大宽度（红）
    mx = agg['max_slice']
    arr.markers.append(_line(frame, mid, [
        _compose(mx['x_left'], mx['y'], z_level, width_idx, long_idx, z_idx),
        _compose(mx['x_right'], mx['y'], z_level, width_idx, long_idx, z_idx),
    ], _rgba(1, 0, 0), width=0.03))
    mid += 1
    arr.markers.append(_text(
        frame, mid,
        _compose((mx['x_left'] + mx['x_right']) / 2, mx['y'], z_level + 0.2,
                 width_idx, long_idx, z_idx),
        f"max {agg['w_max_mm']:.0f}mm @{mx['y']:.1f}m", _rgba(1, 0, 0)))
    mid += 1

    # 最小宽度（蓝）
    mn = agg['min_slice']
    arr.markers.append(_line(frame, mid, [
        _compose(mn['x_left'], mn['y'], z_level, width_idx, long_idx, z_idx),
        _compose(mn['x_right'], mn['y'], z_level, width_idx, long_idx, z_idx),
    ], _rgba(0, 0.3, 1), width=0.03))
    mid += 1
    arr.markers.append(_text(
        frame, mid,
        _compose((mn['x_left'] + mn['x_right']) / 2, mn['y'], z_level + 0.2,
                 width_idx, long_idx, z_idx),
        f"min {agg['w_min_mm']:.0f}mm @{mn['y']:.1f}m", _rgba(0, 0.3, 1)))
    mid += 1

    # 总览文字（绿=正常 / 红=超限）
    ok = not agg['alert']
    summary = (f"spread={agg['spread_mm']:.0f}mm  thr={agg['max_spread_mm']:.0f}mm  "
               f"{'OK' if ok else 'ALERT!'}")
    y_mid = (agg['y_max'] + agg['y_min']) / 2
    arr.markers.append(_text(
        frame, mid,
        _compose(0.0, y_mid, z_level + 0.6, width_idx, long_idx, z_idx),
        summary, _rgba(0.2, 0.9, 0.2) if ok else _rgba(1, 0.1, 0.1), height=0.25))
    return arr
