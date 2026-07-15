"""
可视化一层垛面的每一抓位置及 action 字段
用法：python test_visualize.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from rrt_env.environment_3D import RobotPosition

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── 测试垛型配置 ─────────────────────────────────────────────────────
# N1=10, Group=[3,3,4] (visit order: grab0=left-3, grab1=right-3, grab2=mid-4)
# Stack(N1+1=11): odd s[3]=50 s[7]=50, even s[7]=50 s[10]=50
BOX_W, BOX_H = 420, 280
CAR_W = round(10 * BOX_W + 420)   # gap = CAR_W - 10*BOX_W = 420mm

cfg = {
    'box_list': ['A'],
    'box': {'A': {
        'size': {'L': 600, 'W': BOX_W, 'H': BOX_H},
        'grip': {'P1': [4], 'P2': [2], 'P3': [2]},
        'Nt': 0,
    }},
    'car': {
        'size': {'L': 6000, 'W': CAR_W, 'H': 2000},
        'reserve': {'L': 0, 'W': 50, 'H': 0},
    },
    'regular': [{
        'N1': 10, 'N2': 0, 'N3': 0,
        'T12': 2,   # 两层
        'T3': 0, 'F13': 1, 'F2': 0,
        'E': 0, 'Nx': 0,
        # Stack N1+1=11: s[i]=left gap ratio of box i (%)
        # odd:  gap splits at box3(50%) and box7(50%)
        # even: gap splits at box7(50%) and box10(50%, unused last slot)
        'Stack': [
            [0, 0, 0, 50, 0, 0, 50, 0, 0, 0, 0],   # odd:  50% between group0&1, 50% between group1&2
            [0, 0, 0, 0, 0, 0, 50, 0, 0, 0, 50],   # even: 50% between group1&2, 50% right of group2
        ],
        'Group': [
            [3, 3, 4],   # physical: left=3, mid=3, right=4
            [3, 3, 4],   # visit_order=[0,2,1]: grab0=left-3 grab1=right-4 grab2=mid-3
        ],
    }],
    'trapezoid': [],
    'mixture': [],
}

rp = RobotPosition(cfg)

# ── 打印每一抓字段 ───────────────────────────────────────────────────
ACTION_LABEL = {0: 'normal', 1: 'face_end', 2: 'blk_end', 3: 'all_end'}
DIR_LABEL    = {1: '<-L', 2: 'R->'}

print(f"\n{'='*72}")
print(f"  N1=10  Group=[3,3,4]  gap={CAR_W - 10*BOX_W}mm  T12=2层  F13=1面")
print(f"  Expected: id1=3boxes left dir=1, id2=4boxes right dir=2, id3=3boxes mid dir=1(odd)/gap(even)")
print(f"{'='*72}")
print(f"  {'id':>3}  {'area':4}  {'num':<12}  {'gaps':<12}  {'pos_y':>6}  {'h':>4}  "
      f"{'dir':6}  {'num_F':>5}  {'action'}")
print(f"  {'-'*68}")

for a in rp.robot_offsets:
    if a == 'done':
        print(f"  {'---  done  ---':^68}")
        continue
    print(f"  {a['id']:>3}  {a['area']:4}  {str(a['num']):<12}  {str(a['gaps']):<12}  "
          f"{a['pos'][1]:>6.0f}  {a['pos'][2]:>4}  "
          f"{DIR_LABEL[a['dir']]:6}  {a['num_F']:>5}  "
          f"{a['action']} {ACTION_LABEL[a['action']]}")

# ── 可视化 ───────────────────────────────────────────────────────────
if not HAS_MPL:
    print("\n[跳过可视化：未安装 matplotlib]")
    sys.exit(0)

# 只画第一面第一层（num_F==1, pos[2]==0）
layer_actions = [a for a in rp.robot_offsets
                 if a != 'done' and a['num_F'] == 1 and a['pos'][2] == 0]

fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(-50, CAR_W + 50)
ax.set_ylim(-BOX_H * 0.8, BOX_H * 2.5)
ax.set_xlabel('Y (mm)')
ax.set_title('Face1 Layer1 — grab position & action  (color=grab order)')
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='black', linewidth=1.5, label='左墙')
ax.axvline(CAR_W, color='black', linewidth=1.5, label='右墙')

colors = plt.cm.tab10.colors
for step_i, a in enumerate(layer_actions):
    color = colors[step_i % 10]
    num_list = a['num']
    gaps = a.get('gaps', [])
    y = a['pos'][1]
    box_y_list = []

    # 展开各箱 y 坐标
    for seg_i, seg_n in enumerate(num_list):
        for _ in range(seg_n):
            box_y_list.append(y)
            y += BOX_W
        if seg_i < len(gaps):
            y += gaps[seg_i]

    for by in box_y_list:
        rect = mpatches.FancyBboxPatch(
            (by + 2, 2), BOX_W - 4, BOX_H - 4,
            boxstyle='round,pad=2', linewidth=1,
            edgecolor='white', facecolor=color, alpha=0.85
        )
        ax.add_patch(rect)

    # 标注：抓取序号 + dir + action
    center_y = (box_y_list[0] + box_y_list[-1]) / 2 + BOX_W / 2
    label = f"#{a['id']}\n{DIR_LABEL[a['dir']]}\nact={a['action']}"
    ax.text(center_y, BOX_H / 2, label,
            ha='center', va='center', fontsize=7, color='white', fontweight='bold')

    # 在箱子上方标出 gaps
    if gaps:
        for seg_i, g in enumerate(gaps):
            gap_y = sum(num_list[:seg_i+1]) * BOX_W + a['pos'][1] + sum(gaps[:seg_i])
            ax.annotate(f'gap\n{g}mm',
                        xy=(gap_y + g/2, BOX_H + 10),
                        ha='center', fontsize=7, color='red')

ax.set_yticks([0, BOX_H])
ax.set_yticklabels(['floor', f'H={BOX_H}'])
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'test_visualize.png'), dpi=120)
print("\n[已保存] test_visualize.png")
try:
    plt.show()
except:
    pass
