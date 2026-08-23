#!/usr/bin/env python3
"""
test_placement.py — 垛序落点自动化验证脚本

用法：
    python test_placement.py [json文件路径 ...]
    python test_placement.py  # 默认加载同目录下所有 .json 文件

检测项：
  1. 落点越界：y 超出 [0, car_W]，z 超出 [0, car_H]
  2. 箱子干涉：当前抓 AABB 与已放置箱子 AABB 重叠
"""
import sys
import os
import json
import glob

# 路径初始化
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from robot_process_node import parse_planner_json
from rrt_env.environment_3D import build_robot_positions

# ─── AABB 工具 ────────────────────────────────────────────────────────────────

def action_aabbs(action):
    """
    返回本次抓取所有箱子的 AABB 列表，每项 (y0, y1, z0, z1)。
    gaps 已计入（intra-group 缝隙）。
    """
    pos   = action['pos']
    size  = action['size']
    l, w, h = size
    area  = action['area']
    nums  = action['num']    # list of sub-grab counts
    gaps  = action.get('gaps', [])  # list of gap values between sub-grabs

    y_cur = pos[1]
    result = []
    for i, n in enumerate(nums):
        if area == 'p1':
            y_span, z_span = w * n, h
        elif area == 'p3':
            y_span, z_span = h * n, w
        else:  # p2: 沿 x 方向叠放，y/z 固定
            y_span, z_span = w, h * n
        result.append((y_cur, y_cur + y_span, pos[2], pos[2] + z_span))
        y_cur += y_span
        if i < len(gaps):
            y_cur += gaps[i]
    return result


def aabb_overlap(a, b, tol=0.5):
    """两个 (y0,y1,z0,z1) 是否实质重叠（tol 容差 mm）"""
    return (a[0] < b[1] - tol and a[1] > b[0] + tol and
            a[2] < b[3] - tol and a[3] > b[2] + tol)


# ─── 单 block 验证 ─────────────────────────────────────────────────────────────

def check_block(rp, car_W, car_H, block_idx):
    errors  = []
    placed  = []  # 已放 AABB 列表（含 action id，便于报错定位）
    actions = [a for a in rp.ori_offsets if a != 'done']

    for action in actions:
        aabbs = action_aabbs(action)
        aid   = action['id']
        area  = action['area']
        nF    = action['num_F']

        for aabb in aabbs:
            y0, y1, z0, z1 = aabb

            # 1. 边界检查
            if y0 < -1 or y1 > car_W + 1:
                errors.append(
                    f'  [越界-Y] Block{block_idx} id={aid} area={area} numF={nF} '
                    f'y=[{y0:.0f},{y1:.0f}] 超出[0,{car_W}]'
                )
            if z0 < -1 or z1 > car_H + 1:
                errors.append(
                    f'  [越界-Z] Block{block_idx} id={aid} area={area} numF={nF} '
                    f'z=[{z0:.0f},{z1:.0f}] 超出[0,{car_H}]'
                )

            # 2. 干涉检查（与同面同类型的已放箱子）
            for prev_aabb, prev_id, prev_area, prev_nF in placed:
                if prev_nF != nF:
                    continue  # 不同面不检测（不同面可复用坐标空间）
                if aabb_overlap(aabb, prev_aabb):
                    errors.append(
                        f'  [干涉]   Block{block_idx} id={aid}(area={area},numF={nF}) '
                        f'与 id={prev_id}(area={prev_area}) 重叠 '
                        f'cur_y=[{y0:.0f},{y1:.0f}] z=[{z0:.0f},{z1:.0f}]'
                    )
                    break

        for aabb in aabbs:
            placed.append((aabb, aid, area, nF))

    return len(actions), errors


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def test_file(json_path):
    print(f'\n{"="*70}')
    print(f'文件：{os.path.basename(json_path)}')
    print(f'{"="*70}')

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 解析进 glob_data
    ok = parse_planner_json(data)
    if ok is False:
        print('  [ERROR] parse_planner_json 失败')
        return

    # 取解析结果
    import robot_process_node as rpn
    if not rpn.glob_data:
        print('  [ERROR] glob_data 为空')
        return

    car = rpn.glob_data[0]['car']
    car_W = car['size']['W']
    car_H = car['size']['H']

    total_grabs  = 0
    total_errors = []

    try:
        rp_list = build_robot_positions(rpn.glob_data)
    except Exception as e:
        print(f'  [ERROR] 完整垛序构造失败：{e}')
        return

    for bi, rp in enumerate(rp_list):
        box_type = rp.box_type
        n_grabs, errors = check_block(rp, car_W, car_H, bi + 1)
        total_grabs  += n_grabs
        total_errors += errors

        status = '✓' if not errors else f'✗ {len(errors)}个问题'
        print(f'  Block{bi+1} 箱型={box_type} 总箱={rp.box_count} '
              f'抓次={n_grabs}  [{status}]')

    print(f'\n  汇总：{total_grabs} 次抓取', end='')
    if total_errors:
        print(f'，发现 {len(total_errors)} 个问题：')
        for e in total_errors:
            print(e)
    else:
        print('，所有落点合法，无干涉 ✓')


def main():
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = sorted(glob.glob(os.path.join(_DIR, '*.json')))
        # 排除 config.json
        files = [f for f in files if os.path.basename(f) != 'config.json']

    if not files:
        print('未找到 JSON 文件，请传入文件路径或将 JSON 放在脚本同目录')
        return

    for path in files:
        test_file(path)


if __name__ == '__main__':
    main()
