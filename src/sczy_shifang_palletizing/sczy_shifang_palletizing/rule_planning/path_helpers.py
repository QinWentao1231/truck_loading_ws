"""从 robot_process 提取的纯路径函数（2026-10-15），不导入 ROS/点云/通信节点。"""

import numpy as np


def _action_car_width(action, rp) -> float:
    """返回当前抓所属面的车宽；非异形订单等于 ``rp.W``。"""
    return float(action.get('car_width', rp.W))


def _head_corner_shrink(action, rp) -> float:
    """返回当前面左右角点各自需要向车内收缩的距离（mm）。

    异形车头按车厢中心线对称处理，因此单侧收缩量为原始车宽与当前面
    可用车宽差值的一半。普通区域（包括已经超过 head.L 的面）固定返回0，
    避免把异形区域的补偿继续带到后续码垛面。
    """
    if action is None or not bool(action.get('is_head', False)):
        return 0.0
    return max(0.0, (float(rp.W) - _action_car_width(action, rp)) * 0.5)


def _action_p1_right_wall(action, rp) -> float:
    """返回当前面P1区域右边界，规则垛会扣除同面的P2区域。"""
    return float(action.get(
        'p1_right_wall',
        _action_car_width(action, rp) - rp.N2 * rp.l,
    ))


def _action_right_edge_y(action) -> float:
    """返回整抓箱体在车宽方向上的实际右边界。"""
    y_start = float(action['pos'][1])
    actual_num = sum(int(value) for value in action['num'])
    if action['area'] == 'p1':
        return (
            y_start
            + actual_num * float(action['size'][1])
            + sum(float(value) for value in action.get('gaps', []))
        )
    if action['area'] == 'p3':
        # P3侧立后，车宽方向占用的是箱子原始高度。
        return y_start + actual_num * float(action['size'][2])
    # P2的多箱沿车深方向排列，车宽方向只占一个箱宽。
    return y_start + float(action['size'][1])


def _is_row_rightmost_grab(action, rp) -> bool:
    """判断当前抓是否为所在行最右抓。

    同一 block 内以“同面、同区域、同底面高度”定义一行；按整抓箱体
    的实际 Y 右边界比较，兼容不同箱宽、P3侧立和P1多段放置。
    """
    row_actions = [
        item for item in rp.ori_offsets
        if item != 'done'
        and int(item['num_F']) == int(action['num_F'])
        and item['area'] == action['area']
        and np.isclose(
            float(item['pos'][2]), float(action['pos'][2]),
            rtol=0.0, atol=1e-3,
        )
    ]
    if not row_actions:
        return False
    rightmost_y = max(_action_right_edge_y(item) for item in row_actions)
    return bool(np.isclose(
        _action_right_edge_y(action), rightmost_y,
        rtol=0.0, atol=1e-3,
    ))


def _corner_compensation_applies(action, rp, config_be, direction) -> bool:
    """角点宽度补偿仅用于右半侧、方向2且为所在行最右侧的抓次。"""
    return (
        bool(config_be['use_corner'])
        and direction == 2
        and _is_row_rightmost_grab(action, rp)
        and float(action['pos'][1]) >= _action_car_width(action, rp) * 0.5
    )


def _build_p1_p3_approach(action, rp, config_be, reserve_grip,
                          y_offset_app, direction=None):
    """按现有规则生成 P1/P3 的 x0、x1、APP 和碰撞包围盒。"""
    direction = action['dir'] if direction is None else direction
    app_offset = config_be['p1_app_offset_list'][0]
    x_app = (
        action['pos'][0] + app_offset[0],
        action['pos'][1] + y_offset_app,
        action['pos'][2] + app_offset[2],
    )

    # P3 侧立时箱子 z 跨度为原始宽 w；P1 竖放时为原始高 h。
    box_z = action['size'][1] if action['area'] == 'p3' else action['size'][2]
    if action['pos'][2] + box_z <= config_be['p3_init_pos'][2]:
        x0 = [
            config_be['p1_init_pos'][0],
            config_be['p1_init_pos'][1],
            max(config_be['p1_init_pos'][2] - box_z, x_app[2]),
        ]
    elif action['area'] == 'p1':
        action_grab_p1 = action.get('grab_num_p1', rp.grab_num_p1)
        action_box_w = action['size'][1]
        car_width = _action_car_width(action, rp)
        x0 = [
            config_be['p1_init_pos'][0],
            max(car_width - (action_grab_p1 + 1) * action_box_w, x_app[1]),
            max(config_be['p3_init_pos'][2] - box_z, x_app[2]),
        ]
    else:
        x0 = [
            config_be['p3_init_pos'][0],
            config_be['p3_init_pos'][1],
            max(config_be['p3_init_pos'][2] - box_z, x_app[2]),
        ]

    # 第一层以箱体底面 Z=0 判断。不能用“底面Z < 当前箱高”判断：
    # 混装面箱高不一致时，较高箱可能把已经位于第二层的箱子误判为第一层。
    if np.isclose(float(action['pos'][2]), 0.0, atol=1e-3):
        x0[0] = 200

    if action['pos'][2] + box_z <= config_be['p3_init_pos'][2]:
        if direction == 1:
            x1_y = (x_app[1] + action['size'][1] / 2
                    if action['pos'][1] < action['size'][1]
                    and action['pos'][2] <= 2 * box_z
                    else x_app[1])
        else:
            x1_y = x_app[1]
        x1 = [x0[0], x1_y, x0[2]]
    else:
        x1 = [
            100 if action['area'] == 'p1' else x0[0],
            x0[1] if action['area'] == 'p1' else x_app[1],
            x0[2],
        ]

    actual_num = sum(action['num'])
    if action['area'] == 'p1':
        size = (
            action['size'][0],
            action['size'][1] * actual_num,
            action['size'][2] + reserve_grip[2],
        )
    else:
        size = (
            action['size'][0],
            action['size'][2] * actual_num,
            action['size'][1] + reserve_grip[2],
        )
    return x0, x1, x_app, size


def _candidate_goal(action, rp, config_be, dis_y, direction):
    """计算候选轨迹使用的目标点，不产生重复日志。"""
    if _corner_compensation_applies(
            action, rp, config_be, direction):
        correction = float(np.clip(
            dis_y - _action_car_width(action, rp), 0, 40))
        return (action['pos'][0], action['pos'][1] + correction, action['pos'][2])
    return tuple(action['pos'])


def _check_path_height(path, carried_size, car_height, tolerance=0.5):
    """检查各路径点处整抓包围盒顶部是否超过车厢物理高度。"""
    issues = []
    carried_height = float(carried_size[2])
    car_height = float(car_height)
    for point_index, point in enumerate(path):
        point_z = float(point[2])
        top_z = point_z + carried_height
        if top_z <= car_height + tolerance:
            continue
        if point_index == 0:
            label = 'x0'
        elif point_index == 1:
            label = 'x1'
        elif point_index == 2:
            label = 'APP'
        else:
            label = f'goal{point_index - 2}'
        issues.append({
            'index': point_index,
            'label': label,
            'point_z': point_z,
            'carried_height': carried_height,
            'top_z': top_z,
            'car_height': car_height,
            'over_height': top_z - car_height,
        })
    return issues


def _check_action_lateral_bounds(action, rp, tolerance=0.5):
    """检查当前抓的实际箱体是否超出所在区域的车宽边界。

    该检查只计算垛型落箱范围，不进行RRT或轨迹避障。P1多段抓按 ``num``
    和 ``gaps`` 逐段累计，避免负间隙时只看最终右端而漏掉中间段越界；规则垛
    P1使用扣除P2后的右边界，P2/P3及其他垛型使用当前面的动态车宽。
    """
    area = action['area']
    left_y = float(action['pos'][1])
    right_y = left_y

    if area == 'p1':
        cursor_y = left_y
        box_width = float(action['size'][1])
        segment_counts = [int(value) for value in action['num']]
        segment_gaps = [float(value) for value in action.get('gaps', [])]
        for segment_index, segment_count in enumerate(segment_counts):
            segment_start = cursor_y
            segment_end = segment_start + segment_count * box_width
            left_y = min(left_y, segment_start, segment_end)
            right_y = max(right_y, segment_start, segment_end)
            cursor_y = segment_end
            if segment_index < len(segment_gaps):
                cursor_y += segment_gaps[segment_index]
                left_y = min(left_y, cursor_y)
                right_y = max(right_y, cursor_y)
    elif area == 'p3':
        # P3侧立后，原始箱高是车宽方向的单箱跨度。
        right_y = left_y + sum(
            int(value) for value in action['num']) * float(action['size'][2])
        left_y, right_y = min(left_y, right_y), max(left_y, right_y)
    elif area == 'p2':
        # P2多箱沿车深方向排列，车宽方向只占一个箱宽。
        right_y = left_y + float(action['size'][1])
        left_y, right_y = min(left_y, right_y), max(left_y, right_y)
    else:
        raise ValueError(f"未知放置区域: {area}")

    car_width = _action_car_width(action, rp)
    if area == 'p1' and rp.block_type == 'regular':
        right_wall = _action_p1_right_wall(action, rp)
        boundary_name = 'P1可用右边界'
    else:
        right_wall = car_width
        boundary_name = '当前面右车壁'

    left_overhang = max(0.0, -left_y)
    right_overhang = max(0.0, right_y - right_wall)
    if (left_overhang <= tolerance and
            right_overhang <= tolerance):
        return None
    return {
        'area': area,
        'left_y': left_y,
        'right_y': right_y,
        'car_width': car_width,
        'right_wall': right_wall,
        'boundary_name': boundary_name,
        'left_overhang': left_overhang,
        'right_overhang': right_overhang,
    }
