"""规则路径计算：与 robot_process 的非混装 cmd_get_path 保持同一口径。

这里只计算笛卡尔路径和原有离散检查，不运行 RRT，不承诺关节空间可达性。
旧协议不能携带校验状态，调用方发现 issues 时不得把失败路径作为正常路径下发。
"""

from dataclasses import dataclass
from itertools import product
import math

from .path_helpers import (
    _action_car_width, _action_p1_right_wall, _build_p1_p3_approach,
    _candidate_goal, _check_path_height,
)


@dataclass
class PathResult:
    path: list
    size: tuple
    issues: list
    y_offset_app: float


def row_clearance(action, rp):
    """按同行计算余量；前抓优先采用已下发落点，未下发时沿用理论位置。"""
    size_index = 2 if action['area'] == 'p3' else 1
    y_start = action['pos'][1]
    y_right = y_start + sum(action['num']) * action['size'][size_index]
    placed = [a for a in rp.ori_offsets if a != 'done'
              and a['area'] == action['area']
              and a['pos'][2] == action['pos'][2]
              and a['num_F'] == action['num_F'] and a['id'] < action['id']]
    def placed_y(a):
        return rp.sent_positions.get(a['id'], a['pos'])[1]

    def right_edge(a):
        return (placed_y(a) + sum(a['num']) * a['size'][size_index]
                + sum(a.get('gaps', [])))
    left_ends = [right_edge(a) for a in placed if right_edge(a) <= y_start]
    right_starts = [placed_y(a) for a in placed if placed_y(a) >= y_right]
    wall = (_action_p1_right_wall(action, rp) if action['area'] == 'p1'
            else _action_car_width(action, rp))
    return (y_start - (max(left_ends) if left_ends else 0),
            (min(right_starts) if right_starts else wall) - y_right,
            bool(left_ends or right_starts))


def record_sent_placement(action, result, rp, environment):
    """仅在路径发送成功后调用，使模型和邻箱余量使用同一实际下发落点。

    旧协议没有到位反馈，仍沿用“已下发视为已放置”的建模时序；此处不是
    机器人完成确认。保留原垛序与待发动作，补偿不回写、不累加。
    """
    position = tuple(result.path[-1])
    environment.step(dict(action, pos=list(position)))
    if action['action'] in (1, 2, 3):
        rp.sent_positions.clear()
    else:
        rp.sent_positions[action['id']] = position


def _obstacle_free(point, size, obstacles):
    """等价于新版 SearchSpace.obstacle_free 的八角点检查，但不依赖 Rtree。

    边界接触也算命中；同原算法一样，这不是完整的连续扫掠碰撞验证。
    """
    for switches in product((0, 1), repeat=3):
        corner = [point[i] + size[i] * switches[i] for i in range(3)]
        if any(all(obstacle[i] <= corner[i] <= obstacle[i + 3]
                   for i in range(3)) for obstacle in obstacles):
            return False
    return True


def calculate_path(action, rp, config, environment, corner_distance=0.0):
    """生成 x0/x1/APP/goal，既不推进队列，也不修改已放箱体环境。"""
    if len(action['num']) != 1 or action.get('gaps'):
        raise ValueError('旧机器人协议不支持一抓多段放置')
    reserve_grip = config['reserve_grip']
    y_offset = 0.0
    if action['area'] in ('p1', 'p3'):
        left_gap, right_gap, has_neighbor = row_clearance(action, rp)
        cap = float(config.get('min_side_gap', 100))
        y_offset = max(-cap, min(cap, (right_gap - left_gap) * 0.5))
        x0, x1, x_app, size = _build_p1_p3_approach(
            action, rp, config, reserve_grip, y_offset)
    elif action['area'] == 'p2':
        boxes = environment.to_box(action)
        offset = config['p1_app_offset_list'][0]
        x_app = tuple(action['pos'][i] + offset[i] for i in range(3))
        x1 = (config['p2_init_pos'][0], x_app[1] - action['size'][1],
              max(config['p2_init_pos'][2], x_app[2] + 10))
        x0 = x1
        size = (boxes[0].length * 4, boxes[0].width,
                boxes[0].height + reserve_grip[2])
    else:
        raise ValueError(f"不支持的区域: {action['area']}")
    if not math.isfinite(float(corner_distance)) or corner_distance < 0:
        raise ValueError('角点距离必须为有限非负数')
    goal = _candidate_goal(action, rp, config, corner_distance, action['dir'])
    path = [tuple(float(v) for v in point) for point in (x0, x1, x_app, goal)]
    if not all(math.isfinite(v) for point in path for v in point):
        raise ValueError('路径包含无效数值')
    issues = []
    if config.get('chk_enable', True):
        for issue in _check_path_height(path, size, rp.H):
            issues.append(
                f"{issue['label']}顶部超高{issue['over_height']:.1f}mm "
                f"(Z={issue['point_z']:.1f}+包围盒高{issue['carried_height']:.1f}, "
                f"车高={issue['car_height']:.1f})")
        if environment.objects:
            for index, point in enumerate(path[:3]):
                if not _obstacle_free(point, size, environment.objects):
                    issues.append(f'路径点{index}存在箱体干涉: {point}')
            if action['area'] == 'p1' and has_neighbor:
                if left_gap + right_gap < config.get('min_side_gap', 100):
                    issues.append(f'左右余量之和不足: {left_gap:.1f}+{right_gap:.1f}mm')
                expected = 1 if left_gap <= right_gap else 2
                if getattr(rp, 'sequence_strategy', None) == 'legacy':
                    # 什邡原版按整行方向放置；只核对原始垛序的dir，不用新版
                    # “贴近较小间隙一侧”的方向规则否决正常的旧版左→右顺序。
                    expected = next(a['dir'] for a in rp.ori_offsets
                                    if a != 'done' and a['id'] == action['id'])
                if action['dir'] != expected:
                    issues.append(f"放置方向不符: dir={action['dir']}, expected={expected}")
    return PathResult(path, tuple(size), issues, y_offset)
