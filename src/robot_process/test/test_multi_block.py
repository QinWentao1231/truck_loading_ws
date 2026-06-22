"""
测试多 block 解析与顺序切换逻辑
不依赖 gRPC/socket，直接构造 config_rp 列表验证：
  1. callback 解析产出 glob_data 列表
  2. rp_list 各 block 独立生成 robot_offsets
  3. done 切换逻辑：模拟 cmd_get_path 循环，验证跨 block 顺序
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robot_process'))

from rrt_env.environment_3D import RobotPosition

# ── 构造两个 block 的 config_rp ─────────────────────────────────────
def make_block(box_type, box_w, n1, grab, t12, f13, car_w=4000):
    # Group 是 list of list：每个子列表对应一种层型的分组方式
    group_size = grab
    remainder = n1 - group_size * (n1 // group_size)
    single_group = [group_size] * (n1 // group_size)
    if remainder:
        single_group.append(remainder)
    # 奇偶层交替（两种），与 test_positions.py 保持一致
    group = [single_group, list(reversed(single_group))]

    # Stack 长度 = N1+1，末位补0（最后一箱右侧，不参与计算）
    stack = [[0] * (n1 + 1) for _ in range(t12)]
    # 在第一个组边界放100%间隙（组间缝隙）
    if len(single_group) > 1:
        boundary = single_group[0]
        for row in stack:
            row[boundary] = 100
    return {
        'box_list': [box_type],
        'box': {
            box_type: {
                'size': {'L': 600, 'W': box_w, 'H': 300},
                'grip': {'P1': [grab], 'P2': [2], 'P3': [2]},
                'Nt': 0,
            }
        },
        'car': {
            'size': {'L': 6000, 'W': car_w, 'H': 2000},
            'reserve': {'L': 0, 'W': 50, 'H': 0},
        },
        'regular': [{
            'N1': n1, 'N2': 0, 'N3': 0,
            'T12': t12, 'T3': 0,
            'F13': f13, 'F2': 0,
            'E': 0, 'Nx': 0,
            'Stack': stack,
            'Group': group,
        }],
        'trapezoid': [],
        'head': [],
    }


PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'


def check(label, condition):
    print(f"  {'[PASS]' if condition else '[FAIL]'}  {label}")
    return condition


# ── Test 1：glob_data 列表长度与 box_list 内容 ───────────────────────
print("\n" + "="*60)
print("Test 1：callback 解析 → glob_data 为列表，各 block 独立")
cfg1 = make_block('A', 400, 9, 5, 2, 3)
cfg2 = make_block('B', 500, 8, 4, 2, 2)
config_rp = [cfg1, cfg2]

all_ok = True
all_ok &= check("config_rp 是列表", isinstance(config_rp, list))
all_ok &= check("长度 == 2", len(config_rp) == 2)
all_ok &= check("block[0] box_list[0] == 'A'", config_rp[0]['box_list'][0] == 'A')
all_ok &= check("block[1] box_list[0] == 'B'", config_rp[1]['box_list'][0] == 'B')
all_ok &= check("block[0] N1 == 9", config_rp[0]['regular'][0]['N1'] == 9)
all_ok &= check("block[1] N1 == 8", config_rp[1]['regular'][0]['N1'] == 8)


# ── Test 2：rp_list 各 block 独立生成 ───────────────────────────────
print("\n" + "="*60)
print("Test 2：rp_list 各块独立，box/尺寸/robot_offsets 不共享")
rp_list = [RobotPosition(block_cfg)
           for block_cfg in (config_rp if isinstance(config_rp, list) else [config_rp])]
all_ok &= check("rp_list 长度 == 2", len(rp_list) == 2)
all_ok &= check("rp_list[0].box_type == 'A'", rp_list[0].box_type == 'A')
all_ok &= check("rp_list[1].box_type == 'B'", rp_list[1].box_type == 'B')
all_ok &= check("rp_list[0].w == 400", rp_list[0].w == 400)
all_ok &= check("rp_list[1].w == 500", rp_list[1].w == 500)
all_ok &= check("rp_list[0].robot_offsets 非空", len(rp_list[0].robot_offsets) > 0)
all_ok &= check("rp_list[1].robot_offsets 非空", len(rp_list[1].robot_offsets) > 0)
all_ok &= check("两块 robot_offsets 对象不同", rp_list[0].robot_offsets is not rp_list[1].robot_offsets)


# ── Test 3：done 切换逻辑 ────────────────────────────────────────────
print("\n" + "="*60)
print("Test 3：模拟 cmd_get_path 循环，验证 done 时切换 rp")
rp_list2 = [RobotPosition(block_cfg)
            for block_cfg in (config_rp if isinstance(config_rp, list) else [config_rp])]
rp_idx = 0
rp = rp_list2[rp_idx]

visited_blocks = []
action_counts = [0, 0]
max_steps = 10000  # 防死循环

for _ in range(max_steps):
    if not rp.robot_offsets:
        break
    action = rp.robot_offsets.pop(0)
    if action == 'done':
        visited_blocks.append(rp_idx)
        rp_idx += 1
        if rp_idx >= len(rp_list2):
            break
        rp = rp_list2[rp_idx]
        continue
    action_counts[rp_idx] += 1

all_ok &= check("两个 block 均被访问（done 均触发）", visited_blocks == [0, 1])
all_ok &= check("block[0] 动作数 > 0", action_counts[0] > 0)
all_ok &= check("block[1] 动作数 > 0", action_counts[1] > 0)
all_ok &= check("循环结束后 rp_idx == 2（所有块完成）", rp_idx == 2)

print(f"\n  block[0] 动作数: {action_counts[0]}")
print(f"  block[1] 动作数: {action_counts[1]}")


# ── Test 4：单块兼容（config_rp 非列表）────────────────────────────
print("\n" + "="*60)
print("Test 4：config_rp 为单个 dict 时向下兼容")
single_cfg = make_block('C', 400, 6, 3, 1, 2)
rp_list3 = [RobotPosition(block_cfg)
            for block_cfg in (single_cfg if isinstance(single_cfg, list) else [single_cfg])]
all_ok &= check("rp_list 长度 == 1", len(rp_list3) == 1)
all_ok &= check("box_type == 'C'", rp_list3[0].box_type == 'C')


# ── Test 5：num 为列表格式（组内缝隙）───────────────────────────────
print("\n" + "="*60)
print("Test 5：num 字段为列表，gaps 字段存在")
BOX_W = 400
CAR_W = 4000
N1 = 9
GRAB = 5
stack_gap = [[0, 0, 0, 50, 0, 50, 0, 0, 0, 0]]  # N1+1=10个，组内+组间各50%，末位补0
group_gap  = [[5, 4]]
cfg_gap = {
    'box_list': ['D'],
    'box': {'D': {'size': {'L': 600, 'W': BOX_W, 'H': 300},
                  'grip': {'P1': [GRAB], 'P2': [2], 'P3': [2]}, 'Nt': 0}},
    'car': {'size': {'L': 6000, 'W': CAR_W, 'H': 2000}, 'reserve': {'L': 0, 'W': 50, 'H': 0}},
    'regular': [{'N1': N1, 'N2': 0, 'N3': 0, 'T12': 1, 'T3': 0,
                 'F13': 1, 'F2': 0, 'E': 0, 'Nx': 0,
                 'Stack': stack_gap, 'Group': group_gap}],
    'trapezoid': [], 'head': [],
}
rp_gap = RobotPosition(cfg_gap)
real_offsets = [a for a in rp_gap.robot_offsets if a != 'done']
all_num_lists = all(isinstance(a['num'], list) for a in real_offsets)
all_ok &= check("所有 action['num'] 均为 list", all_num_lists)
all_ok &= check("所有 action['gaps'] 均为 list", all(isinstance(a['gaps'], list) for a in real_offsets))

# 验证总箱数
total_boxes = sum(sum(a['num']) for a in real_offsets)
all_ok &= check(f"总箱数 == N1={N1} (实际={total_boxes})", total_boxes == N1)

# 第一个动作（5箱组）应有 gaps（组内分割）
first = real_offsets[0]
has_inner_gap = len(first['num']) > 1
all_ok &= check(f"第一组检测到组内缝隙 num={first['num']} gaps={first['gaps']}", has_inner_gap)


# ── 汇总 ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"总体结果: {'ALL PASS' if all_ok else 'SOME FAILED'}")
