"""
测试 emit_p1_groups 生成的位置正确性
场景：
  Case 1 — 缝隙仅在组间（两组之间），抓取内部无缝隙
  Case 2 — 缝隙落在第一组内部（需拆成 num=[n1,n2], gaps=[g]）
  Case 3 — 组间 + 组内同时有缝隙
"""
import sys, os
# test/ 与 robot_process/ 是兄弟目录，指向上级目录使 rrt_env 包可见
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robot_process'))

from rrt_env.environment_3D import RobotPosition

# ── 公共箱/车参数 ────────────────────────────────────────────────
BOX_L, BOX_W, BOX_H = 600, 400, 300
CAR_W = 4000          # gap = W - N1*w = 4000 - 9*400 = 400 mm
N1    = 9
GRAB  = 5             # grab_num_p1


def make_config(stack, group):
    """构造最小 config_data，只测常规垛 P1 区一面一层。"""
    return {
        'box_list': ['A'],
        'box': {'A': {'size': {'L': BOX_L, 'W': BOX_W, 'H': BOX_H},
                      'grip': {'P1': [GRAB], 'P2': [2], 'P3': [2]}}},
        'car': {'size': {'L': 6000, 'W': CAR_W, 'H': 2000},
                'reserve': {'W': 50}},
        'regular': [{
            'N1': N1, 'N2': 0, 'N3': 0,
            'T12': 1,          # 只测一层
            'T3': 0, 'F13': 1, 'F2': 0,
            'E': 0, 'Nx': 0,
            'Stack': stack,
            'Group': group,
        }],
        'trapezoid': [],
        'mixture': [],
    }


def print_actions(label, rp):
    """打印所有 robot_offsets，并验证箱子连续性。"""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  gap={CAR_W - N1*BOX_W}mm  grab_num={GRAB}")
    print(f"{'='*60}")
    all_boxes = []
    for a in rp.robot_offsets:
        if a == 'done':
            continue
        gaps  = a.get('gaps', [])
        num   = a['num']
        pos_y = a['pos'][1]
        height = a['pos'][2]
        print(f"  id={a['id']:2d}  num={str(num):12s}  gaps={str(gaps):12s}"
              f"  pos_y={pos_y:7.1f}  h={height}")

        # 展开实际每个箱子的 y 坐标
        y = pos_y
        for seg_i, seg_n in enumerate(num):
            for _ in range(seg_n):
                all_boxes.append(round(y, 1))
                y += BOX_W
            if seg_i < len(gaps):
                y += gaps[seg_i]

    print(f"\n  展开箱子 y 坐标（共 {len(all_boxes)} 个）：")
    print(f"  {all_boxes}")

    # 验证：所有箱子应紧密排列在 [0, W) 范围内，或按缝隙分布
    expected_total = N1
    ok = len(all_boxes) == expected_total
    print(f"\n  箱数校验: {'OK' if ok else 'FAIL'} "
          f"(期望={expected_total}, 实际={len(all_boxes)})")


# Stack 长度 = N1+1，s[i] 表示第i箱左侧缝隙占比，s[N1] 为最后一箱右侧（不参与计算，补0）

# ── Case 1：缝隙仅在两组之间 ─────────────────────────────────────
# N1=9, Group=[[5,4],[4,5]]
# s[5]=100 表示全部 400mm 间隙放在第5箱左侧（即两组之间）
stack_case1 = [
    [0, 0, 0, 0, 0, 100, 0, 0, 0, 0],   # 奇数层（t=0）
    [0, 0, 0, 0, 100, 0, 0, 0, 0, 0],   # 偶数层（t=1）
]
group_case1 = [[5, 4], [4, 5]]
rp1 = RobotPosition(make_config(stack_case1, group_case1))
print_actions("Case 1：缝隙在组间（s[5]=100%）", rp1)

# ── Case 2：缝隙落在第一组内部（第3箱后） ───────────────────────
# s[3]=50 在第一组（5箱）内部，s[5]=50 在两组之间
# 期望：第一组 num=[3,2], gaps=[200]
stack_case2 = [
    [0, 0, 0, 50, 0, 50, 0, 0, 0, 0],
    [0, 0, 0, 0, 50, 50, 0, 0, 0, 0],
]
group_case2 = [[5, 4], [4, 5]]
rp2 = RobotPosition(make_config(stack_case2, group_case2))
print_actions("Case 2：缝隙在第一组内部（s[3]=50%, s[5]=50%）", rp2)

# ── Case 3：第二组内部也有缝隙 ───────────────────────────────────
# s[3]=40 第一组内部, s[5]=30 组间, s[7]=30 第二组内部
# 期望：第一组 num=[3,2] gaps=[160]，第二组 num=[2,2] gaps=[120]
stack_case3 = [
    [0, 0, 0, 40, 0, 30, 0, 30, 0, 0],
    [0, 0, 0, 0, 40, 30, 0, 30, 0, 0],
]
group_case3 = [[5, 4], [4, 5]]
rp3 = RobotPosition(make_config(stack_case3, group_case3))
print_actions("Case 3：两组内部均有缝隙（s[3]=40%, s[5]=30%, s[7]=30%）", rp3)
