"""生成带变形的合成车厢点云，切成多站 PCD，供离线回放/调试。

输出到 <ws>/test_data/cabin_pcds/（稳定目录，不被 colcon clean 清理）。
站间固定 3m（匹配节点默认 station_interval_m=3.0）。

用法：
    python3 tools/gen_synthetic_cabin.py            # 仅生成 PCD + 打印分析
    python3 tools/gen_synthetic_cabin.py --view     # 另存 matplotlib 预览图
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cabin_width_check.width_core import measure_profile, aggregate  # noqa: E402

# ── 车厢参数 ────────────────────────────────────────────
L = 17.0        # 长 (Y)
BASE_W = 2.46   # 基准内宽 (X)
H = 2.6         # 高 (Z)
STATION_INTERVAL = 3.0
# 波纹板（参考 LuZhou 真实点云）：竖向肋，沿长轴 y 周期 ~17cm，幅度 ~18mm
CORR_PERIOD = 0.175
CORR_AMP = 0.018


def wall_x(y, side, corr=True):
    """左右壁内表面 x(y)，叠加波纹板纹路 + 两处变形。side=-1左壁/+1右壁。
    波纹相位左右壁错开半周期，更接近真实（两侧肋不一定对齐）。"""
    base = side * BASE_W / 2
    if side < 0:   # 左壁 y=6m 内凹 60mm（变窄）
        base += 0.060 * np.exp(-((y - 6.0) ** 2) / (2 * 0.8 ** 2))
    else:          # 右壁 y=12m 外凸 40mm（变宽）
        base += 0.040 * np.exp(-((y - 12.0) ** 2) / (2 * 1.0 ** 2))
    if corr:
        phase = 0.0 if side < 0 else np.pi
        # 波纹朝车厢内侧凸起（-side 方向），故乘 -side
        base += -side * CORR_AMP * np.sin(2 * np.pi * y / CORR_PERIOD + phase)
    return base


def make_cabin(seed=42, n_wall=120000):
    rng = np.random.default_rng(seed)
    pts = []
    for side in (-1, +1):
        ys = rng.uniform(0, L, n_wall)
        zs = rng.uniform(0.05, H, n_wall)
        xs = np.array([wall_x(y, side) for y in ys]) + rng.normal(0, 0.004, n_wall)
        pts.append(np.column_stack([xs, ys, zs]))
    ng = 30000  # 地面（验证内表面提取不被内部点污染）
    pts.append(np.column_stack([
        rng.uniform(-BASE_W / 2, BASE_W / 2, ng),
        rng.uniform(0, L, ng),
        rng.normal(0, 0.004, ng)]))
    return np.vstack(pts)


def _ws_root():
    # tools/ 上溯到 src/ 之上即 ws 根
    parts = os.path.abspath(__file__).split(os.sep)
    if 'src' in parts:
        return os.sep.join(parts[:parts.index('src')])
    return os.path.join(os.path.dirname(__file__), '..', '..', '..')


def save_stations(cloud, out_dir):
    """按固定 3m 站窗切片，存为 binary PCD（无 open3d 依赖的简易写入）。"""
    os.makedirs(out_dir, exist_ok=True)
    n_st = int(np.ceil(L / STATION_INTERVAL))
    for k in range(n_st):
        start = k * STATION_INTERVAL
        y0, y1 = start - 0.3, start + STATION_INTERVAL + 0.3   # 0.3m 重叠
        seg = cloud[(cloud[:, 1] >= y0) & (cloud[:, 1] < y1)].copy()
        seg[:, 1] -= start   # 转到本站局部坐标（拼接时再加回里程）
        _write_pcd_binary(os.path.join(out_dir, f'station_{k:02d}.pcd'), seg)
    return n_st


def _write_pcd_binary(path, xyz):
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    n = xyz.shape[0]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\nDATA binary\n")
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(xyz.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--view', action='store_true', help='另存 matplotlib 预览图')
    ap.add_argument('--out', default=None, help='输出目录（默认 <ws>/test_data/cabin_pcds）')
    args = ap.parse_args()

    out_dir = args.out or os.path.join(_ws_root(), 'test_data', 'cabin_pcds')
    cloud = make_cabin()
    print(f"合成点云总点数：{len(cloud)}")
    n_st = save_stations(cloud, out_dir)
    print(f"已存 {n_st} 站 PCD 到 {out_dir}（站间 {STATION_INTERVAL}m）")

    profile = measure_profile(cloud, slice_step=0.5, long_idx=1, width_idx=0,
                              min_pts=200, face_percentile=5.0)
    agg = aggregate(profile, max_spread_mm=60.0)
    print(f"中位宽={agg['median_mm']:.0f}mm  "
          f"max={agg['w_max_mm']:.0f}@{agg['y_max']:.1f}m  "
          f"min={agg['w_min_mm']:.0f}@{agg['y_min']:.1f}m  "
          f"spread={agg['spread_mm']:.0f}mm  告警={'是' if agg['alert'] else '否'}")

    if args.view:
        _save_view(cloud, profile, agg, out_dir)


def _save_view(cloud, profile, agg, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    valid = [p for p in profile if p['valid']]
    ys = [p['y'] for p in valid]
    ws = [p['width'] * 1000 for p in valid]
    yc = np.array([p['y'] for p in valid])
    reg = np.array([p['region'] for p in valid])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 9))
    sub = cloud[np.random.choice(len(cloud), 25000, replace=False)]
    # 按所在切片区域着色：最宽区=红、最窄区=蓝、正常=灰
    pidx = np.clip(np.searchsorted(yc, sub[:, 1]), 0, len(valid) - 1)
    preg = reg[pidx]
    cmap = {'wide': '#e51515', 'narrow': '#2659ff', 'normal': '0.6'}
    for name, col in cmap.items():
        m = preg == name
        if m.any():
            a1.scatter(sub[m, 1], sub[m, 0] * 1000, s=0.6, c=col,
                       label={'wide': '最宽区', 'narrow': '最窄区', 'normal': 'normal'}[name])
    a1.set_xlabel('Y (m)'); a1.set_ylabel('X (mm)')
    a1.set_title(f"top view  spread={agg['spread_mm']:.0f}mm thr=60mm {'ALERT' if agg['alert'] else 'OK'}"
                 f"  (red=widest blue=narrowest)")
    # 宽度曲线也按区域着色点
    a2.plot(ys, ws, '-', color='0.5', zorder=1)
    cw = [cmap[r] for r in reg]
    a2.scatter(ys, ws, c=cw, s=30, zorder=2)
    a2.axhline(agg['median_mm'], color='gray', ls='--')
    a2.plot(agg['y_max'], agg['w_max_mm'], 'r^', ms=12)
    a2.plot(agg['y_min'], agg['w_min_mm'], 'bv', ms=12)
    a2.set_xlabel('Y (m)'); a2.set_ylabel('width (mm)'); a2.set_title('width profile')
    png = os.path.join(out_dir, 'preview.png')
    plt.tight_layout(); plt.savefig(png, dpi=110)
    print(f"预览图已保存：{png}")


if __name__ == '__main__':
    main()
