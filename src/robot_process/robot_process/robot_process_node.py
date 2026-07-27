import os
import sys
current_file = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file)
sys.path.insert(0, current_directory)
import time
import struct
import json
import pickle
import numpy as np
from rrt_env.environment_3D import BinEnv, RobotPosition
from rrt_env.fanuc_kinematics import FanucKinematics, make_uf_transform
from auxiliary_methods.search_space import SearchSpace
from socket_pkg.socket_server import SocketApp
from socket_pkg import socket_client
from stacking_detection.stacking_detection_node import collect_dual_lidar_once, check_stacking, save_point_clouds
from grpc_pkg.grpc_server import GrpcServer
from auxiliary_methods.plotting import (
    Plot, resolve_output_dir, save_face_layout,
)
from auxiliary_methods.logging import Logger
from auxiliary_methods.resume_store import ResumeStore, resolve_resume_dir


# ── 机器人下发指令（16字节定长，机器人→服务端）────────────────────────
# 格式：start_word(2B) + version(2B) + action_id(2B) + reserved(10B)
# fefe = 起始字，0001 = 版本，01xx = 指令编号
cmd_get_pallet    = 'fefe0000010100000000000000000000'  # 请求垛型总体信息
cmd_get_per_count = 'fefe0000010200000000000000000000'  # 请求当前码垛面箱数
cmd_get_box       = 'fefe0000010300000000000000000000'  # 请求下一箱来料配方
cmd_get_path      = 'fefe0000010400000000000000000000'  # 请求下一抓放置路径
cmd_chk_path      = 'fefe0000010500000000000000000000'  # 触发路径预取（批量拉取剩余路径）
cmd_stacking      = 'fefe0000010600000000000000000000'  # 触发堆叠检测（双雷达采集+宽度计算）

# ── 响应报文头（服务端→机器人）────────────────────────────────────────
# 固定7字节：start_word(2B)=fefe + version(1B)=01 + reserved(1B)=00
#            trans_to(2B)=1023 + block_type(1B)=01
_MSG_HEADER = b'\xfe\xfe\x01\x00\x10\x23\x01'

# ── 路径点 area 编号（info_block 第1浮点）────────────────────────────
# p1=主堆区  p2=侧壁区  p3=侧立区
_AREA_NUM = {'p1': 1, 'p2': 2, 'p3': 3}

# ── 垛面检测拍照位 J1 角（度），用于点云偏航补偿 ────────────────────────
# 雷达装在桅杆上、桅杆装在机器人 J1 轴上；不同箱型在各自拍照位拍照，
# 与"正对箱面"的 J1 有固定偏差 → 点云相应偏航，需补偿。
# 箱型首位：1=常规、2=细支、3=中支。
#   常规(1xx)/中支(3xx) → 拍照位 -50°（偏 +10°）
#   细支(2xx)          → 拍照位 -55°（偏 +5°）
# 偏航补偿角 = 拍照位 J1 − 正对 J1
_J1_FACE_DEG = -60.5            # 正对（雷达垂直箱面）时的 J1
_J1_PHOTO_REGULAR_DEG = -53.1  # 常规箱(1xx) / 中支箱(3xx) 拍照位 J1
_J1_PHOTO_SLIM_DEG = -56.6     # 细支烟箱(2xx) 拍照位 J1


def _yaw_offset_for_box(box_type) -> float:
    """根据箱型首位判定烟支类型，返回点云偏航补偿角(度) = 拍照位 J1 − 正对 J1。
    首位 '2'=细支 用 -55°，其余(1xx常规/3xx中支) 用 -50°。"""
    j1_photo = _J1_PHOTO_SLIM_DEG if str(box_type)[:1] == '2' else _J1_PHOTO_REGULAR_DEG
    return j1_photo - _J1_FACE_DEG


def _build_msg(num_blocks: int, payload: bytes) -> bytes:
    """拼装完整响应报文。
    结构：_MSG_HEADER(7B) + num_blocks(1B) + err_code(1B)=00
          + reserved(3B)=000000 + mes_data_num(1B)=91 + payload
    num_blocks：本次报文携带的数据块数量（每块41字节）。"""
    return _MSG_HEADER + num_blocks.to_bytes(1, 'big') + b'\x00\x00\x00\x00\x00\x91' + payload


def _data_block(*floats) -> bytes:
    """编码一个通用数据块（41字节）。
    结构：index(1B)=00 + class_id(4B)=00000000 + float32 * N（大端）+ 零填充至9个float槽
    float 槽共9个（36字节），不足部分补零。"""
    return (
        b'\x00\x00\x00\x00\x00'
        + struct.pack('!' + 'f' * len(floats), *floats)
        + b'\x00' * (4 * (9 - len(floats)))
    )


def _find_mixture_block_position(rp_list) -> int:
    """返回首个混装 block 在 rp_list 中从 1 开始的位置；不存在时返回 0。"""
    return next(
        (index for index, item in enumerate(rp_list, start=1)
         if item.block_type == 'mixture'),
        0,
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
    if action['pos'][2] + box_z < config_be['p3_init_pos'][2]:
        x0 = [
            config_be['p1_init_pos'][0],
            config_be['p1_init_pos'][1],
            max(config_be['p1_init_pos'][2] - box_z, x_app[2]),
        ]
    elif action['area'] == 'p1':
        action_grab_p1 = action.get('grab_num_p1', rp.grab_num_p1)
        action_box_w = action['size'][1]
        x0 = [
            config_be['p1_init_pos'][0],
            max(rp.W - (action_grab_p1 + 1) * action_box_w, x_app[1]),
            max(config_be['p3_init_pos'][2] - box_z, x_app[2]),
        ]
    else:
        x0 = [
            config_be['p3_init_pos'][0],
            config_be['p3_init_pos'][1],
            max(config_be['p3_init_pos'][2] - box_z, x_app[2]),
        ]

    if action['pos'][2] < box_z:
        x0[0] = 200

    if action['pos'][2] + box_z < config_be['p3_init_pos'][2]:
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
    if config_be['use_corner'] and direction == 2:
        correction = float(np.clip(dis_y - rp.W, 0, 40))
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


def _save_face_visualizations(rp, block_number, face_number, stamp,
                              issue_ids=None, save_face=True,
                              save_mixture=False):
    """保存面级PNG；混装面可额外保存独立的 mixture_pallet 文件。"""
    actions = [
        action for action in rp.ori_offsets
        if action != 'done' and action['num_F'] == face_number
    ]
    paths = []
    common_title = (
        f'Block {block_number} Face {face_number} - {rp.block_type}')
    if save_face:
        paths.append(save_face_layout(
            actions,
            {'L': rp.L, 'W': rp.W, 'H': rp.H},
            f'face_block{block_number:02d}_face{face_number:02d}_{stamp}',
            title=common_title,
            issue_ids=issue_ids,
        ))
    if save_mixture and rp.block_type == 'mixture':
        paths.append(save_face_layout(
            actions,
            {'L': rp.L, 'W': rp.W, 'H': rp.H},
            f'mixture_pallet_block{block_number:02d}_face{face_number:02d}_{stamp}',
            title=f'Mixture pallet - Block {block_number} Face {face_number}',
            issue_ids=issue_ids,
        ))
    return paths


def _save_parsed_order(config_rp, rp_list, source='online'):
    """垛序构造成功后保存规范化订单字段；仅接收阶段不调用。"""
    now_text = time.strftime('%Y-%m-%d %H:%M:%S')
    day_text = time.strftime('%Y%m%d')
    stamp = (
        time.strftime('%Y%m%d_%H%M%S')
        + f'_{time.time_ns() % 1_000_000_000:09d}'
    )
    output_dir = os.path.join(resolve_output_dir(day_text), 'orders')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'parsed_order_{stamp}.json')
    normalized_blocks = (
        config_rp if isinstance(config_rp, list) else [config_rp]
    )
    block_summaries = []
    for block_index, rp_item in enumerate(rp_list, start=1):
        actions = [item for item in rp_item.ori_offsets if item != 'done']
        block_summaries.append({
            'block': block_index,
            'block_type': rp_item.block_type,
            'box_types': list(rp_item.box_configs.keys()),
            'box_count': int(rp_item.box_count),
            'grab_count': len(actions),
            'face_count': max(
                (int(item['num_F']) for item in actions), default=0),
        })
    record = {
        'schema_version': 1,
        'parsed_at': now_text,
        'source': source,
        'block_count': len(rp_list),
        'total_box_count': sum(item['box_count'] for item in block_summaries),
        'total_grab_count': sum(item['grab_count'] for item in block_summaries),
        'mixture_block_position': _find_mixture_block_position(rp_list),
        'block_summaries': block_summaries,
        'order_fields': normalized_blocks,
    }
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
    return output_path


def _collect_mixture_fields(config_rp):
    """将内部规范化结构还原为接口收到的扁平 mixture 字段。"""
    blocks = config_rp if isinstance(config_rp, list) else [config_rp]
    result = []
    for block_index, block in enumerate(blocks, start=1):
        mixture_items = []
        for face in block.get('mixture', []) or []:
            items = _json_mixture_items(face)
            if items is None:
                continue
            for item in items:
                pos = _json_value(item, 'Pos', {}) or {}
                mixture_items.append({
                    'Type': _json_value(item, 'Type', ''),
                    'Num': _json_value(item, 'Num', 0),
                    'Pos': {
                        'X': _json_value(pos, 'X', 0.0),
                        'Y': _json_value(pos, 'Y', 0.0),
                        'Z': _json_value(pos, 'Z', 0.0),
                    },
                })
        if mixture_items:
            result.append({
                'block': block_index,
                'mixture': mixture_items,
            })
    return result


def _write_chk_path_summary(session):
    """将 cmd_chk_path 批量检查结果写入 TXT/JSON，并返回统计信息。"""
    results = session['results']
    abnormal = [item for item in results if item['issues']]
    summary = {
        'started_at': session['started_at'],
        'finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_grabs': len(results),
        'normal_grabs': len(results) - len(abnormal),
        'abnormal_grabs': len(abnormal),
        'path_htmls': session.get('path_htmls', []),
        'face_images': session.get('face_images', []),
        'system_issues': session.get('system_issues', []),
        'mixture_fields': session.get('mixture_fields', []),
        'results': results,
    }
    output_dir = resolve_output_dir(session['stamp'][:8])
    prefix = os.path.join(
        output_dir, f'cmd_chk_path_summary_{session["stamp"]}')
    json_path = prefix + '.json'
    txt_path = prefix + '.txt'
    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    lines = [
        'cmd_chk_path 路径检查汇总',
        f'开始时间: {summary["started_at"]}',
        f'结束时间: {summary["finished_at"]}',
        f'总抓数: {summary["total_grabs"]}',
        f'正常抓数: {summary["normal_grabs"]}',
        f'异常抓数: {summary["abnormal_grabs"]}',
    ]
    if abnormal:
        lines.append('异常定位:')
        for item in abnormal:
            lines.append(
                f'  Block {item["block"]} / 面 {item["face"]} / '
                f'第 {item["grab"]} 抓 / 箱型 {item["box_type"]}: '
                + '；'.join(item['issues']))
    else:
        lines.append('检查结果: 未发现异常')
    if summary['mixture_fields']:
        lines.append('收到的mixture字段:')
        lines.extend(json.dumps(
            summary['mixture_fields'], ensure_ascii=False, indent=2).splitlines())
    if summary['path_htmls'] or summary['face_images']:
        lines.append('可视化文件:')
        lines.extend(f'  {path}' for path in summary['path_htmls'])
        lines.extend(f'  {path}' for path in summary['face_images'])
    if summary['system_issues']:
        lines.append('非路径异常:')
        lines.extend(f'  {issue}' for issue in summary['system_issues'])
    with open(txt_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines) + '\n')
    summary['json_path'] = json_path
    summary['txt_path'] = txt_path
    summary['report_lines'] = lines
    return summary


def _log_chk_path_summary(summary, logger):
    """把TXT汇总的同一份内容输出到终端/运行日志。"""
    logger.info('========== cmd_chk_path 检查汇总 ==========')
    warning_section = False
    for line in summary.get('report_lines', []):
        if line in ('异常定位:', '非路径异常:'):
            warning_section = True
            logger.warning(line)
            continue
        if line in ('收到的mixture字段:', '可视化文件:'):
            warning_section = False
        if warning_section:
            logger.warning(line)
        else:
            logger.info(line)
    logger.info(f'汇总TXT: {summary["txt_path"]}')
    logger.info(f'汇总JSON: {summary["json_path"]}')
    logger.info('===========================================')


def _parse_stack_group(src, stack_attr='Stack', group_attr='Group'):
    """将 protobuf Stack/Group 转为二维列表"""
    return {
        'Stack': [list(x.Width) for x in getattr(src, stack_attr)],
        'Group': [list(x.Unit) for x in getattr(src, group_attr)],
    }


def _parse_regular(src_list):
    """解析 Regular 列表，跳过 N1==0 的项"""
    result = []
    for item in src_list:
        if item.N1 == 0:
            continue
        result.append({
            'N1': item.N1, 'N2': item.N2, 'N3': item.N3,
            'T12': item.T12, 'T3': item.T3,
            'F13': item.F13, 'F2': item.F2,
            'E': item.E, 'Nx': item.Nx,
            'Type': item.Type,
            **_parse_stack_group(item),
        })
    return result


def _parse_trapezoid(src_list):
    """解析 Trapezoid 列表，跳过 N1==0 的项"""
    result = []
    for item in src_list:
        if item.N1 == 0:
            continue
        result.append({
            'N1': item.N1, 'N3': item.N3,
            'T1': item.T1, 'T3': item.T3,
            'Nx': item.Nx,
            'Isdoor': item.Isdoor,
            'Type': item.Type,
            **_parse_stack_group(item),
        })
    return result


def _proto_mixture_items(face):
    """返回 protobuf 混装条目；兼容旧版 Items 嵌套结构。"""
    items = getattr(face, 'Items', None)
    if items is not None:
        return items
    if all(hasattr(face, name) for name in ('Type', 'Num', 'Pos')):
        return (face,)
    return None


def _parse_mixture(src_list):
    """将新版扁平 Mixture 列表归一成内部单面 Items 结构。"""
    flat_items = []
    legacy_faces = []
    for entry in src_list:
        nested_items = getattr(entry, 'Items', None)
        items = nested_items if nested_items is not None else (entry,)
        parsed_items = []
        for item in items:
            parsed_items.append({
                'Type': item.Type,
                'Num': item.Num,
                'Pos': {'X': item.Pos.X, 'Y': item.Pos.Y, 'Z': item.Pos.Z},
            })
        if nested_items is None:
            flat_items.extend(parsed_items)
        else:
            legacy_faces.append({'Items': parsed_items})

    # 新版协议中 Block.mixture 的全部条目共同组成唯一一个混装面。
    if flat_items:
        return [{'Items': flat_items}, *legacy_faces]
    return legacy_faces


def _json_mixture_items(face):
    """返回 JSON 混装条目；兼容旧版 Items 嵌套结构。"""
    for key in ('Items', 'items'):
        if key in face:
            return face[key]
    if any(key in face for key in ('Type', 'type')):
        return (face,)
    return None


def _json_value(data, upper_name, default=None):
    """读取 Type/type、Num/num、Pos/pos、X/x 等大小写形式。"""
    return data.get(upper_name, data.get(upper_name.lower(), default))


def _parse_mixture_json(src_list):
    """将新版扁平 Mixture 列表归一成内部单面 Items 结构。"""
    flat_items = []
    legacy_faces = []
    for entry in src_list:
        nested_items = None
        for key in ('Items', 'items'):
            if key in entry:
                nested_items = entry[key]
                break
        items = nested_items if nested_items is not None else (entry,)
        parsed_items = []
        for item in items:
            pos = _json_value(item, 'Pos', {}) or {}
            parsed_items.append({
                'Type': _json_value(item, 'Type', ''),
                'Num': _json_value(item, 'Num', 0),
                'Pos': {
                    'X': _json_value(pos, 'X', 0.0),
                    'Y': _json_value(pos, 'Y', 0.0),
                    'Z': _json_value(pos, 'Z', 0.0),
                },
            })
        if nested_items is None:
            flat_items.extend(parsed_items)
        else:
            legacy_faces.append({'Items': parsed_items})

    if flat_items:
        return [{'Items': flat_items}, *legacy_faces]
    return legacy_faces



def parse_planner_json(data: dict):
    """解析规划器 JSON 格式响应，产出与 callback() 相同的 glob_data 列表。
    Stack/Group 项为 dict 格式：{"Width": [...]} / {"Unit": [...]}"""
    global glob_data
    try:
        cond     = data['condition']
        car_orig = cond['car']['original']
        car_rsv  = cond['car']['reserve']
        car_info = {
            'size':    {'L': car_orig['L'], 'W': car_orig['W'], 'H': car_orig['H']},
            'reserve': {'L': car_rsv['L'],  'W': car_rsv['W'],  'H': car_rsv['H']},
        }


        # 按箱型 type 建立索引（忽略 orderId，避免类型不一致问题）
        type_map = {}
        for order in cond['orders']:
            for box in order['boxes']:
                btype = box.get('type') or box.get('Type', '')
                if btype and btype not in type_map:
                    orig = box['original']
                    rsv  = box.get('reserve', {})
                    grip = box['grip']
                    type_map[btype] = {
                        'size': {'L': orig['L'], 'W': orig['W'], 'H': orig['H']},
                        'reserve': {'L': rsv.get('L', 0), 'W': rsv.get('W', 0), 'H': rsv.get('H', 0)},
                        'grip': {'P1': list(grip['P1']), 'P2': list(grip['P2']), 'P3': list(grip['P3'])},
                        'Nt': box.get('Nt', 0),
                    }

        idx = data['answer']['index']
        index_info = {
            'Ns': idx['Ns'], 'Us': idx['Us'], 'Um': idx['Um'],
            'Gn': list(idx.get('Gn', [])), 'Tp': idx.get('Tp', 0),
        }

        glob_data = []
        for blk in data['answer']['blocks']:
            # 收集 block 内所有用到的箱型
            all_items = blk.get('regular', []) + blk.get('trapezoid', []) + blk.get('mixture', [])
            blk_types = []
            for x in all_items:
                for key in ('Type',):
                    t = x.get(key, '')
                    if t and t not in blk_types:
                        blk_types.append(t)
                mixture_items = _json_mixture_items(x)
                if mixture_items is not None:
                    for mixture_item in mixture_items:
                        t = _json_value(mixture_item, 'Type', '')
                        if t and t not in blk_types:
                            blk_types.append(t)
            box_dict = {t: type_map[t] for t in blk_types if t in type_map}
            box_info = {'box_list': blk_types, 'box': box_dict}

            regular = []
            for item in blk.get('regular', []):
                if item.get('N1', 0) == 0:
                    continue
                regular.append({
                    'N1': item['N1'], 'N2': item['N2'], 'N3': item['N3'],
                    'T12': item['T12'], 'T3': item['T3'],
                    'F13': item['F13'], 'F2': item['F2'],
                    'E': item['E'], 'Nx': item['Nx'],
                    'Type': item.get('Type', ''),
                    'Stack': [list(x['Width']) for x in item['Stack']],
                    'Group': [list(x['Unit'])  for x in item['Group']],
                })

            trapezoid = []
            for item in blk.get('trapezoid', []):
                if item.get('N1', 0) == 0:
                    continue
                trapezoid.append({
                    'N1': item['N1'],
                    'N3': item.get('N3', 0),
                    'T1': item['T1'],
                    'T3': item.get('T3', 0),
                    'Nx': item.get('Nx', 0),
                    'Isdoor': item.get('Isdoor', False),
                    'Type': item.get('Type', ''),
                    'Stack': [list(x['Width']) for x in item.get('Stack', [])],
                    'Group': [list(x['Unit'])  for x in item.get('Group', [])],
                })

            mixture = _parse_mixture_json(blk.get('mixture', []))

            glob_data.append({
                'box_list':  box_info['box_list'],
                'box':       box_info['box'],
                'car':       car_info,
                'regular':   regular,
                'trapezoid': trapezoid,
                'mixture':   mixture,
                'index':     index_info,
            })
    except Exception as e:
        logs.error(f"解析垛型JSON失败: {e}")
        return False, "receive failed!"
    logs.info("已接受垛型数据，请将机器人进行连接或重新下发垛型数据")
    return True, "receive succeed!"


def callback(data):
    global glob_data
    try:
        car_orig = data.condition.car.original
        car_rsv  = data.condition.car.reserve
        car_info = {
            'size':    {'L': car_orig.L, 'W': car_orig.W, 'H': car_orig.H},
            'reserve': {'L': car_rsv.L,  'W': car_rsv.W,  'H': car_rsv.H},
        }

        # 按箱型 type 建立索引（忽略 orderId）
        type_map = {}
        for order in data.condition.orders:
            for box in order.boxes:
                if box.type and box.type not in type_map:
                    type_map[box.type] = {
                        'size': {'L': box.original.L, 'W': box.original.W, 'H': box.original.H},
                        'reserve': {'L': box.reserve.L, 'W': box.reserve.W, 'H': box.reserve.H},
                        'grip': {'P1': list(box.grip.P1), 'P2': list(box.grip.P2), 'P3': list(box.grip.P3)},
                        'Nt': box.Nt,
                    }

        # 全局统计参数（Answer 级别，所有 block 共享）
        idx = data.answer.index
        index_info = {
            'Ns': idx.Ns, 'Us': idx.Us, 'Um': idx.Um,
            'Gn': list(idx.Gn), 'Tp': idx.Tp,
        }
        # 每个 block 按 Type 收集箱型信息
        glob_data = []
        for blk in data.answer.blocks:
            all_items = list(blk.regular) + list(blk.trapezoid) + list(blk.mixture)
            blk_types = []
            for x in all_items:
                for t in [getattr(x, 'Type', '')]:
                    if t and t not in blk_types:
                        blk_types.append(t)
                mixture_items = _proto_mixture_items(x)
                if mixture_items is not None:
                    for mixture_item in mixture_items:
                        t = getattr(mixture_item, 'Type', getattr(mixture_item, 'type', ''))
                        if t and t not in blk_types:
                            blk_types.append(t)
            box_dict = {t: type_map[t] for t in blk_types if t in type_map}
            box_info = {'box_list': blk_types, 'box': box_dict}
            glob_data.append({
                'box_list':  box_info['box_list'],
                'box':       box_info['box'],
                'car':       car_info,
                'regular':   _parse_regular(blk.regular),
                'trapezoid': _parse_trapezoid(blk.trapezoid),
                'mixture':   _parse_mixture(blk.mixture),
                'index':     index_info,
            })
    except Exception as e:
        logs.error(f"解析垛型失败: {e}")
        return False, "receive failed!"
    logs.info("已接受垛型数据，请将机器人进行连接或重新下发垛型数据")
    return True, "receive succeed!"


logs = Logger(level="debug").get_log()


def _fast_forward(rp_list, be, cursor):
    """断点续传快进：按游标跳过已完成部分，重放当前 block 重建 BinEnv。
    返回 (rp_idx, rp, last_grab_action, last_box_id, last_path_id)。"""
    rp_idx = int(cursor['rp_idx'])
    box_id = int(cursor['box_id'])
    path_id = int(cursor['path_id'])
    # 已完成 block：清空队列
    for i in range(rp_idx):
        rp_list[i].boxes.clear()
        rp_list[i].robot_offsets.clear()
    rp = rp_list[rp_idx]
    # 当前 block：boxes 弹到 box_id
    while rp.boxes and rp.boxes[0]['id'] <= box_id:
        rp.boxes.pop(0)
    # 当前 block：robot_offsets 弹到 path_id，逐抓重放 be.step 重建障碍物
    last_act = None
    while rp.robot_offsets and rp.robot_offsets[0] != 'done' and rp.robot_offsets[0]['id'] <= path_id:
        act = rp.robot_offsets.pop(0)
        last_act = act
        # 混装断点重放时同步恢复动态 dir，保证 reserve_object 的外扩方向
        # 与正常连续运行一致。失败保持垛序原 dir，只记录，不影响续传。
        if rp.block_type == 'mixture' and act['area'] == 'p1':
            try:
                clearance = be.mixture_side_clearance(
                    act, left_wall=0, right_wall=rp.W)
                if clearance['blocking']:
                    raise ValueError(
                        f"目标区域与 {len(clearance['blocking'])} 个已码箱体重叠")
                act['dir'] = (
                    1 if clearance['left_gap'] <= clearance['right_gap'] else 2)
            except Exception as exc:
                logs.warning(
                    f"[MIX-APP] 续传重放 Round.{act['id']} 动态方向恢复失败，"
                    f"保持原方向: {exc}")
        be.step(act)   # action 0 累加障碍；1/2/3 自动清空，与正常流程一致
    # 当前 block 已全部消费完 → 前进到下一 block（处理断电恰在 block 边界）
    while rp_idx + 1 < len(rp_list) and (not rp.robot_offsets or all(x == 'done' for x in rp.robot_offsets)):
        rp.robot_offsets.clear()
        rp.boxes.clear()
        rp_idx += 1
        rp = rp_list[rp_idx]
        box_id = path_id = 0
        last_act = None
    return rp_idx, rp, last_act, box_id, path_id


def main():
    global glob_data
    # 日志配置
    logs.info("***垛序及机器人路径规划程序启动 ver.0.5.6（支持混码）-alpha-for济南烟厂***")
    try:
        try:
            from ament_index_python.packages import get_package_share_directory
            _cfg_path = os.path.join(get_package_share_directory('robot_process'), 'config.json')
        except Exception:
            _cfg_path = os.path.join(current_directory, 'config.json')
        with open(_cfg_path, "r", encoding="utf-8") as f:
            config_be = json.load(f)
        # 配置文件参数获取
        ip = config_be['ip']
        port = config_be['port']
        level = config_be['level']
        reserve_grip = config_be['reserve_grip']
        off_line_mode = config_be['off_line_mode']
        show_env = config_be['show_env']
    except Exception as e:
        logs.error(f"读取参数文件失败,请检查项目根目录config.json文件: {e}")
        sys.exit(1)
    logs.info("读取参数文件成功！")
    # 运动学辅助（可选）：验证路径点是否接近奇异/超关节限位
    # 启用方式：config.json 中添加 "use_kinematics": true
    # User Frame 配置：config.json 中添加 "kin_uf_offset": [x, y, z, rx_deg, ry_deg, rz_deg]
    kin = None
    if config_be.get('use_kinematics', False):
        T_uf = make_uf_transform(config_be.get('kin_uf_offset'))
        kin = FanucKinematics(uf_transform=T_uf)
        logs.info("运动学辅助已启用（Fanuc R-1000iA/120F），注意核对 DH 参数与 User Frame 偏置")
    # 断点续传：两个独立开关
    #   resume_save       —— 是否持久化进度（保存；开销小，建议常开。失败只告警不中断）
    #   resume_on_restart —— 重启时是否自动检测并续码（关了则忽略旧进度，走正常等垛型）
    resume_save = config_be.get('resume_save', True)
    resume_on_restart = config_be.get('resume_on_restart', True)
    resume_need_confirm = config_be.get('resume_need_confirm', True)
    # 只要任一开关开启就建 store（存/读分别由各自开关控制）
    store = ResumeStore(resolve_resume_dir(current_file), logger=logs) \
        if (resume_save or resume_on_restart) else None
    # 主循环
    while True:
        # 断点续传：有未完成进度则跳过等待垛型，直接用磁盘保存的计划恢复
        resume_data = store.load() if (store and not off_line_mode and resume_on_restart) else None
        resume_cursor = None
        if resume_data:
            config_rp, resume_cursor = resume_data
            order_source = 'resume'
            logs.warning("检测到未完成码垛进度，进入断点续传（跳过等待垛型）")
        # 选择读取垛型参数的方式
        elif off_line_mode:
            try:
                order_source = 'offline'
                # 离线读取文件模式
                logs.warning("已启用离线模式，垛型参数文件夹为 {}".format(os.getcwd() + '/pkl_data/'))
                time.sleep(1)
                if 'file' not in locals():
                    file = input("请输入垛型文件名称：")
                config_rp = pickle.load(file=open('{0}/pkl_data/{1}'.format(current_directory, file), 'rb'))
                logs.info("读取垛型配置文件成功！配置文件路径为 {}".format(os.getcwd() + '/pkl_data/' + file))
            except Exception as e:
                logs.error(f"{os.getcwd() + '/pkl_data/' + file} 垛型配置文件读取失败: {e}")
                sys.exit(1)
        else:
            # 在线读取垛型模式
            order_source = 'online'
            glob_data = None
            if 'svr' not in vars():
                svr = GrpcServer(5007, callback)
                svr.run()
            logs.warning("已启用在线模式，服务器已启动，等待接收垛型数据...")
        try:
            # logs.info("等待机器人连接中...")
            if 'server' not in vars():
                server = SocketApp()
            server.start(ip, port)
        except Exception as e:
            logs.error(f"创建机器人服务器失败，请检查项目根目录config.json文件: {e}")
            sys.exit(1)
        logs.info("连接机器人成功！")
        # 停止接受垛型数据（仅在线、非续传时才启动过 gRPC）
        if not off_line_mode and not resume_data:
            config_rp = glob_data
            svr.stop()
            del svr
            logs.info("停止接受垛型")
        # 创建环境
        be = BinEnv(config_be)
        logs.info("码垛环境初始化成功！")
        # 计算垛序（逐 block 生成，保留各自 rp 对象）
        try:
            rp_list = [RobotPosition(block_cfg)
                       for block_cfg in (config_rp if isinstance(config_rp, list) else [config_rp])]
            mixture_block_position = _find_mixture_block_position(rp_list)
            # 最后一个 block 的末条目 action 升为 3（全部结束），其余 block 末条目保持 2（block 结束）
            rp_list[-1].boxes[-1]['action'] = 3
            last_offset = next(x for x in reversed(rp_list[-1].robot_offsets) if x != 'done')
            last_offset['action'] = 3
            logs.info(
                f"计算垛序成功！共 {len(rp_list)} 个 block，"
                f"混装 block 位置={mixture_block_position}")
        except Exception as e:
            logs.error(f"垛序计算失败，请检查垛型数据: {e}")
            sys.exit(1)
        # 只有所有 Block 均成功构造 RobotPosition 后才保存；gRPC 仅接收时不落盘。
        try:
            parsed_order_path = _save_parsed_order(
                config_rp, rp_list, source=order_source)
            logs.info(f"订单解析字段已保存: {parsed_order_path}")
        except Exception as order_save_error:
            # 订单归档属于诊断信息，保存失败不能影响机器人主流程。
            logs.warning(f"订单解析字段保存失败，继续执行: {order_save_error}")
        # 断点续传：首次收到垛型时落盘保存计划（续传时计划已在磁盘，跳过）
        if store and not off_line_mode and resume_save and resume_cursor is None:
            store.save_plan(config_rp)
            logs.info("已保存垛型计划用于断点续传")
        rp_idx = 0
        rp = rp_list[rp_idx]
        # 环境初始化
        be.reset()
        dis_x = 0          # 角点激光补偿：x方向偏差（暂未使用）
        dis_y = 0          # 角点激光补偿：y方向偏差，用于修正 x_goal
        # chk_value: 批量预取计数器，0=正常逐抓模式，>0=预取剩余抓次（cmd_chk_path触发）
        chk_value = 0
        # chk_session: cmd_chk_path 本轮逐抓结果、面级图片及最终汇总。
        chk_session = None
        # last_grab_action: get_path 刚下发的那一抓，供 cmd_stacking 计算"当前抓"宽度
        # （cmd_stacking 在 get_path 之后触发，robot_offsets[0] 已是下一抓，不能用）
        # last_grab_box_type: 那一抓的实际箱型（Mixture 内可按 Items 切换）
        last_grab_action = None
        last_grab_box_type = None
        # cur_box_id/cur_path_id: 当前已下发的 box/path 抓号，每次"先存后发"写入游标
        cur_box_id = 0
        cur_path_id = 0
        # 断点续传：快进到游标 + 操作员确认
        if resume_cursor is not None:
            rp_idx, rp, last_grab_action, cur_box_id, cur_path_id = _fast_forward(rp_list, be, resume_cursor)
            if last_grab_action is not None:
                last_grab_box_type = last_grab_action.get('box_type', rp.box_type)
            logs.warning("===== 断点续传待确认 =====")
            logs.warning(f"将从 block {rp_idx + 1}/{len(rp_list)} 继续，"
                         f"已完成至 box_id≤{cur_box_id}, path_id≤{cur_path_id}")
            logs.warning("请核对机器人当前实际已码位置（可能与服务端相差最多 1 抓）。")
            if resume_need_confirm:
                try:
                    _ans = input("核对无误输入 y 回车开始续传，其它退出: ").strip().lower()
                except EOFError:
                    _ans = ''
                if _ans != 'y':
                    logs.error("未确认续传，程序退出。")
                    sys.exit(1)
            logs.warning("断点续传已确认，继续码垛。")
        # 指令集循环
        # try:
        while True:
            if chk_value == 0:
                try:
                    mes_hex = server.receive_message(byte_size=16)
                except Exception:
                    logs.warning("机器人已断开连接 ！")
                    break
                byte_data = bytes.fromhex(mes_hex)
                if byte_data == b'':
                    logs.warning("机器人已断开连接 ！")
                    break
                if mes_hex == cmd_chk_path:
                    chk_value = len(rp.robot_offsets) - 1
                    _chk_stamp = time.strftime('%Y%m%d_%H%M%S')
                    chk_session = {
                        'stamp': _chk_stamp,
                        'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'results': [],
                        'path_htmls': [],
                        'face_images': [],
                        'system_issues': [],
                        'mixture_fields': _collect_mixture_fields(config_rp),
                    }
                    logs.info(
                        f"cmd_chk_path 批量检查开始，当前 Block {rp_idx + 1}，"
                        f"本 Block 待检查 {chk_value} 抓")
                    if chk_value <= 0:
                        try:
                            _empty_summary = _write_chk_path_summary(chk_session)
                            logs.info("cmd_chk_path 当前无待检查路径")
                            _log_chk_path_summary(_empty_summary, logs)
                        except Exception as _empty_summary_error:
                            logs.warning(
                                f"cmd_chk_path 空任务汇总保存失败: {_empty_summary_error}")
                        finally:
                            chk_session = None
            else:
                mes_hex = cmd_get_path
            if mes_hex == cmd_get_pallet:
                # 发送总体信息（3块：垛型参数 + 箱型参数 + 混装 block 位置）
                payload = (
                    _data_block(rp.box_count, rp.ori_offsets[-2]['num_F'], rp.W)
                    + _data_block(rp.l, rp.w, rp.h)
                    + _data_block(float(mixture_block_position))
                    + b'\x00\x00'
                )
                server.send_message(_build_msg(3, payload))
                floor_n = rp.ori_offsets[-2]['num_F']
                logs.debug(
                    f'总箱数: {rp.box_count}, 码垛面数：{floor_n}, '
                    f'车厢宽度：{round(rp.W, 2)}, '
                    f'箱子尺寸：长{rp.l} 宽{rp.w} 高{rp.h}, '
                    f'混装 block 位置：{mixture_block_position}')
            elif mes_hex == cmd_get_per_count:
                # 发送单面信息
                pallet_cnt = rp.cal_floor_count()
                if config_be['use_corner']:
                    dis_y, dis_x, count = socket_client.create_tcp_client("192.168.10.100", 9002,
                                                                            b'\xff\xfe\x01\x00\x01\xfd')
                    logs.warning(f"角点个数：{count}")
                    logs.warning(f"value between corners and car width: {dis_y - rp.W:.2f}")
                # 每行抓数：按当前面(boxes[0]所属)实时查，多条 regular 各面精确
                try:
                    _n_per_row = rp.cal_n_per_row()
                except Exception as _e:
                    logs.error(f'每行抓数解析异常：{type(_e).__name__}: {_e}，已发 0 兜底')
                    _n_per_row = 0
                _face_box_type = (rp.boxes[0].get('box_type', rp.box_type)
                                  if rp.boxes else rp.box_type)
                server.send_message(_build_msg(1, _data_block(
                    float(pallet_cnt), float(_face_box_type), float(_n_per_row)
                ) + b'\x00\x00'))
                logs.debug(f'单面码垛放置次数：{pallet_cnt}，每行抓数={_n_per_row}')
            elif mes_hex == cmd_get_box:
                # 单次来料箱子样式（2块：来料配方 + 当前箱型尺寸）
                # box['action'] 语义：
                #   0 → 普通，继续
                #   1 → 当前码垛面最后一箱，机器人换面
                #   2 → 当前 block 最后一箱，等待下一 block
                #   3 → 所有 block 最后一箱，码垛结束
                if not rp.boxes:
                    logs.warning("boxes 队列已空，无更多来料配方可发（请重连并重发垛型）")
                    continue
                box = rp.boxes.pop(0)
                box_cfg = box['num']
                area_cfg = box.get('area_cfg', 1)
                box_type_cfg = box.get('box_type', rp.box_type)
                box_size_cfg = box.get('size', [rp.l, rp.w, rp.h])
                # 断点续传：先存游标后发送（断电宁可漏一抓也不重码）
                cur_box_id = box['id']
                if store and not off_line_mode and resume_save:
                    store.save_cursor(rp_idx, cur_box_id, cur_path_id)
                payload = (
                    _data_block(float(box_cfg), float(int(box_type_cfg)), float(area_cfg))
                    + _data_block(*map(float, box_size_cfg))
                    + b'\x00\x00'
                )
                server.send_message(_build_msg(2, payload))
                logs.debug(
                    f'来料箱子配方号：{box_cfg}，位置编号：{area_cfg}，'
                    f'尺寸L/W/H：{box_size_cfg}，动作标志：{box["action"]}')
            elif mes_hex == cmd_get_path:
            # 单次路径点
                if not rp.robot_offsets:
                    logs.warning("robot_offsets 队列已空，无更多路径可发（请重连并重发垛型）")
                    chk_value = 0   # 退出批量预取，避免空队列死循环
                    continue
                action = rp.robot_offsets.pop(0)
                if action == 'done':
                    # 正常不应再弹出 done，切换已在 action==2 时提前完成；此处兜底
                    chk_value = 0   # 同样退出批量预取
                    continue
                path_issues = []
                physical_support = None
                last_grab_action = action          # 记录当前抓，供随后 cmd_stacking 对齐序号
                last_grab_box_type = action.get('box_type', rp.box_type)
                # 断点续传：先存游标后发送（在计算/发送路径之前落盘）
                cur_path_id = action['id']
                if store and not off_line_mode and resume_save:
                    store.save_cursor(rp_idx, cur_box_id, cur_path_id)
                if action['id'] == 1:
                    logs.info("*** Block.{} NO.{} 码垛面 开始！***".format(rp_idx + 1, 1))
                logs.debug("Round.{}/{} start planning...".format(action['id'], len(rp.ori_offsets) - 1))
                start_time = time.time()
                obstacles = be.objects
                boxs = be.to_box(action)
                # 路径点语义：x0=初始悬停点，x1=过渡点（y向偏移避障），x_app=放箱接近点，x_goal=目标落箱点
                # size=(L, Y跨度, Z高度+夹爪余量)，仅用于碰撞检测包围盒，不下发给机器人
                mixture_clearance = None
                if action['area'] in ('p1', 'p3'):
                    # dir: 1=从右往左放（贴左侧已放箱），2=从左往右放（贴右侧已放箱）
                    # P1 接近点 Y 偏移：approach 侧到边界距离 * 0.5，clamp [50, 100]；p3 沿用配置固定值
                    # 接近点 Y 偏移：取左右两侧空隙的中点，offset=(右-左)/2，
                    # 正值向右(+y)、负值向左(-y)，使两侧余量均等；幅值上限 min_side_gap，无下限
                    _cap = config_be['min_side_gap']
                    if action['area'] == 'p1':
                        _y_start      = action['pos'][1]
                        _y_grip_right = _y_start + sum(action['num']) * action['size'][1]
                        _cur_h        = action['pos'][2]
                        _p1_right_wall = rp.W - rp.N2 * rp.l
                        def _phys_right_a(a):
                            return a['pos'][1] + sum(a['num']) * a['size'][1] + sum(a.get('gaps', []))
                        _placed = [a for a in rp.ori_offsets
                                   if a != 'done' and a['area'] == 'p1'
                                   and a['pos'][2] == _cur_h
                                   and a['num_F'] == action['num_F']
                                   and a['id'] < action['id']]
                        _right_ends = [a['pos'][1] for a in _placed if a['pos'][1] >= _y_grip_right]
                        _dist_right = (min(_right_ends) if _right_ends else _p1_right_wall) - _y_grip_right
                        _left_ends_a = [_phys_right_a(a) for a in _placed if _phys_right_a(a) <= _y_start]
                        _dist_left = _y_start - (max(_left_ends_a) if _left_ends_a else 0)
                    else:  # p3
                        _y_start       = action['pos'][1]
                        _y_grip_right  = _y_start + sum(action['num']) * action['size'][2]  # p3 y步长=h
                        _cur_h         = action['pos'][2]
                        _p3_right_wall = rp.W
                        def _phys_right_p3(a):
                            return a['pos'][1] + sum(a['num']) * a['size'][2] + sum(a.get('gaps', []))
                        _placed_p3 = [a for a in rp.ori_offsets
                                      if a != 'done' and a['area'] == 'p3'
                                      and a['pos'][2] == _cur_h
                                      and a['num_F'] == action['num_F']
                                      and a['id'] < action['id']]
                        _right_ends_p3 = [a['pos'][1] for a in _placed_p3 if a['pos'][1] >= _y_grip_right]
                        _dist_right = (min(_right_ends_p3) if _right_ends_p3 else _p3_right_wall) - _y_grip_right
                        _left_ends_p3 = [_phys_right_p3(a) for a in _placed_p3 if _phys_right_p3(a) <= _y_start]
                        _dist_left = _y_start - (max(_left_ends_p3) if _left_ends_p3 else 0)
                    y_offset_app = max(-_cap, min(_cap, (_dist_right - _dist_left) * 0.5))
                    fallback_dir = action['dir']
                    fallback_y_offset_app = y_offset_app
                    mixture_clearance = None

                    # 混装面不能使用“底部 Z 完全相等”识别同层。按已经实际码放的
                    # 箱体 AABB 查找与当前抓 X/Z 有效重叠的左右障碍，并动态修正 dir。
                    if rp.block_type == 'mixture' and action['area'] == 'p1':
                        try:
                            mixture_clearance = be.mixture_side_clearance(
                                action,
                                left_wall=0,
                                right_wall=rp.W,
                                min_z_overlap=float(
                                    config_be.get('mixture_z_overlap_min', 20.0)),
                            )
                            if mixture_clearance['blocking']:
                                raise ValueError(
                                    f"目标区域与 {len(mixture_clearance['blocking'])} 个已码箱体重叠")
                            left_gap = mixture_clearance['left_gap']
                            right_gap = mixture_clearance['right_gap']
                            if left_gap < 0 or right_gap < 0:
                                raise ValueError(
                                    f"左右间隙为负数: left={left_gap:.1f}, right={right_gap:.1f}")
                            y_offset_app = max(
                                -_cap,
                                min(_cap, (right_gap - left_gap) * 0.5),
                            )
                            action['dir'] = 1 if left_gap <= right_gap else 2
                        except Exception as _mix_neighbor_error:
                            mixture_clearance = None
                            action['dir'] = fallback_dir
                            y_offset_app = fallback_y_offset_app
                            path_issues.append(
                                f"混装空间邻箱分析失败，已回退原APP: {_mix_neighbor_error}")
                            logs.warning(
                                f"[MIX-APP] Round.{action['id']} 空间邻箱分析失败，"
                                f"保持原APP策略: {_mix_neighbor_error}")

                    x0, x1, x_app, size = _build_p1_p3_approach(
                        action, rp, config_be, reserve_grip,
                        y_offset_app, direction=action['dir'])
                elif action['area'] == 'p2':
                    x_app = tuple([action['pos'][0] + config_be['p1_app_offset_list'][0][0],
                                action['pos'][1] + config_be['p1_app_offset_list'][0][1],
                                action['pos'][2] + config_be['p1_app_offset_list'][0][2]])
                    x1 = tuple([config_be['p2_init_pos'][0],
                                x_app[1] - action['size'][1],
                                max(config_be['p2_init_pos'][2], x_app[2]+10)])
                    x0 = x1
                    size = (boxs[0].length * 4, boxs[0].width, boxs[0].height + reserve_grip[2])
                else:
                    raise Exception('unknown area ! ')

                # 角点信息引入补正。混装候选轨迹也使用补正后的目标点验证。
                x_goal = _candidate_goal(
                    action, rp, config_be, dis_y, action['dir'])

                # 混装面 APP 候选：连续验证 x0→x1→APP→goal。动态策略任何异常
                # 或所有候选均碰撞时，恢复旧 dir/APP，保持原流程继续下发，仅记录日志。
                if mixture_clearance is not None:
                    try:
                        raw_candidates = [y_offset_app]
                        raw_candidates.extend(config_be.get(
                            'mixture_app_y_candidates', [0, 50, -50, 100, -100]))
                        raw_candidates.append(fallback_y_offset_app)
                        candidates = []
                        for value in raw_candidates:
                            value = max(-_cap, min(_cap, float(value)))
                            if not any(abs(value - old) < 1e-6 for old in candidates):
                                candidates.append(value)

                        chosen = None
                        last_collision = None
                        sample_step = float(config_be.get(
                            'mixture_path_sample_step', 10.0))
                        for candidate_offset in candidates:
                            candidate_x0, candidate_x1, candidate_app, candidate_size = \
                                _build_p1_p3_approach(
                                    action, rp, config_be, reserve_grip,
                                    candidate_offset, direction=action['dir'])
                            # 整抓在 APP 处不得越过车厢左右边界。
                            if (candidate_app[1] < 0
                                    or candidate_app[1] + candidate_size[1] > rp.W):
                                continue
                            candidate_goal = _candidate_goal(
                                action, rp, config_be, dis_y, action['dir'])
                            candidate_path = [
                                candidate_x0, candidate_x1,
                                candidate_app, candidate_goal,
                            ]
                            is_safe, collision = be.trajectory_collision_free(
                                candidate_path,
                                candidate_size,
                                sample_step=sample_step,
                            )
                            if is_safe:
                                chosen = (
                                    candidate_offset, candidate_x0, candidate_x1,
                                    candidate_app, candidate_size, candidate_goal,
                                )
                                break
                            last_collision = collision

                        if chosen is None:
                            raise RuntimeError(
                                f"没有无碰撞APP候选，最后碰撞信息={last_collision}")

                        (y_offset_app, x0, x1, x_app,
                         size, x_goal) = chosen
                        logs.info(
                            f"[MIX-APP] Round.{action['id']} 空间邻箱={mixture_clearance['relevant_count']}"
                            f" left_gap={mixture_clearance['left_gap']:.1f}mm"
                            f" right_gap={mixture_clearance['right_gap']:.1f}mm"
                            f" dir={fallback_dir}->{action['dir']}"
                            f" APP_Y偏移={fallback_y_offset_app:.1f}->{y_offset_app:.1f}mm")
                    except Exception as _mix_path_error:
                        action['dir'] = fallback_dir
                        y_offset_app = fallback_y_offset_app
                        x0, x1, x_app, size = _build_p1_p3_approach(
                            action, rp, config_be, reserve_grip,
                            y_offset_app, direction=action['dir'])
                        x_goal = _candidate_goal(
                            action, rp, config_be, dis_y, action['dir'])
                        path_issues.append(
                            f"混装候选轨迹失败，已回退原APP: {_mix_path_error}")
                        logs.warning(
                            f"[MIX-APP] Round.{action['id']} 候选轨迹规划失败，"
                            f"保持原APP策略: {_mix_path_error}")

                if config_be['use_corner'] and action['dir'] == 2:
                    logs.warning(
                        'Goal has been corrected, value:{:.2f}'.format(
                            np.clip((dis_y - rp.W), 0, 40)))

                # 运动学辅助：奇异性/限位检查 + x_app 自动调整
                if kin is not None:
                    rid = action['id']
                    # 先验证全路径（x0/x_init/x_app/x_goal）
                    kin_results = kin.validate_path([x0, x1, x_app, x_goal])
                    kin_labels = ['x0(过渡)', 'x_init(接近2)', 'x_app(接近1)', 'x_goal(目标)']

                    for ki, (label, kr) in enumerate(zip(kin_labels, kin_results)):
                        if not kr['reachable'] or kr['near_singularity']:
                            q_prev_kin = kin_results[ki - 1]['q'] if ki > 0 else None
                            q_info = str(kin.q_deg(kr['q'])) if kr.get('q') is not None else 'N/A'
                            margin_info = str(kin.joint_limit_margin_deg(kr['q'])) if kr.get('q') is not None else 'N/A'
                            logs.warning(f"[KIN] Round.{rid} {label} {kr['msg']} | q={q_info} | 限位余量={margin_info}")
                            path_issues.append(f"运动学异常 {label}: {kr['msg']}")

                            # ── x_app 自动规避（x_goal / x_init / x0 不改动）──
                            if label == 'x_app(接近1)':
                                new_xyz, new_res = kin.resolve_approach_point(
                                    x_app,
                                    kr['singularity_type'],
                                    q_seed=q_prev_kin,
                                )
                                if new_xyz is not None:
                                    logs.warning(
                                        f"[KIN] Round.{rid} x_app 已自动调整 "
                                        f"{tuple(round(v,1) for v in x_app)} → {tuple(round(v,1) for v in new_xyz)} | "
                                        f"新姿态: {new_res['msg']}"
                                    )
                                    x_app = new_xyz
                                else:
                                    logs.warning(f"[KIN] Round.{rid} x_app 无法自动规避，保持原值")

                # 多段放置：num=[n1,n2,...] gaps=[g1,...] 时，追加各段落点
                # 最终 path = [x0, x1, x_app, x_goal_seg1, x_goal_seg2, ...]
                # 机器人侧：前3点为过渡/接近点（区域 flag），之后每点为一段落点
                # 末点（最后一段）使用动作 flag，其余各段落点使用区域 flag
                # 多段放置：组内存在缝隙时 num=[n1,n2,...] gaps=[g1,...]，每段有独立落箱点
                num_segs = action['num']
                gaps = action.get('gaps', [])
                extra_goals = []
                if len(num_segs) > 1:
                    # 第二段落点 y = x_goal_y + gap（x_goal 是第一段参考点，第一段箱子向反方向延伸）
                    y = x_goal[1]
                    for seg_i in range(len(num_segs) - 1):
                        y += gaps[seg_i]
                        extra_goals.append((x_goal[0], y, x_goal[2]))
                #  开始验证
                path = [x0, x1, x_app, x_goal] + extra_goals
                if chk_session is not None:
                    try:
                        _continuous_safe, _continuous_detail = \
                            be.trajectory_collision_free(
                                path[:4], size,
                                sample_step=float(config_be.get(
                                    'mixture_path_sample_step', 10.0)))
                        if not _continuous_safe:
                            path_issues.append(
                                "连续轨迹干涉: "
                                f"segment={_continuous_detail['segment']}, "
                                f"ratio={_continuous_detail['ratio']:.3f}, "
                                f"point={tuple(round(v, 1) for v in _continuous_detail['point'])}")
                    except Exception as _continuous_error:
                        _issue = f"连续轨迹检查执行失败: {_continuous_error}"
                        chk_session['system_issues'].append(
                            f"Block {rp_idx + 1} / 面 {action['num_F']} / "
                            f"第 {action['id']} 抓: {_issue}")
                        logs.warning(_issue)
                if config_be['chk_enable']:
                    logs.info(f"x_goal:{x_goal[0]:.2f}, {x_goal[1]:.2f}, {x_goal[2]:.2f}"
                              + (f"  extra_goals({len(extra_goals)}段): " +
                                 ", ".join(f"[{g[0]:.2f},{g[1]:.2f},{g[2]:.2f}]" for g in extra_goals)
                                 if extra_goals else ""))
                    # 混装面物理支撑分析只提示和记录，不阻断路径下发。
                    # 使用当前抓之前已经写入 BinEnv 的真实箱体，检查底面支撑率、
                    # 单箱最低支撑率以及单箱中心是否落在支撑面内。
                    if rp.block_type == 'mixture' and action['area'] == 'p1':
                        try:
                            physical_support = be.analyze_mixture_support(action)
                            action['_physical_support'] = physical_support
                            if physical_support['risk']:
                                risk_name = (
                                    '完全悬空'
                                    if physical_support['risk_level'] == 'floating'
                                    else '支撑面积不足')
                                risk_boxes = physical_support['risk_box_indices']
                                issue_text = (
                                    f"混装物理结构风险({risk_name}): "
                                    f"整抓支撑={physical_support['support_ratio'] * 100:.1f}%，"
                                    f"悬空={physical_support['unsupported_ratio'] * 100:.1f}%，"
                                    f"最低单箱支撑="
                                    f"{physical_support['min_box_support_ratio'] * 100:.1f}%，"
                                    f"风险单箱={risk_boxes or '无'}")
                                path_issues.append(issue_text)
                                logs.warning(
                                    f"[CHK-SUPPORT] Round.{action['id']} "
                                    f"{issue_text}")
                            else:
                                logs.info(
                                    f"[CHK-SUPPORT] Round.{action['id']} 支撑正常："
                                    f"整抓={physical_support['support_ratio'] * 100:.1f}%，"
                                    f"最低单箱="
                                    f"{physical_support['min_box_support_ratio'] * 100:.1f}%")
                        except Exception as _support_error:
                            issue_text = (
                                f"混装物理支撑分析失败: {_support_error}")
                            path_issues.append(issue_text)
                            logs.warning(
                                f"[CHK-SUPPORT] Round.{action['id']} "
                                f"{issue_text}")
                    # 规划器只保证箱体垛型在车厢内；路径抬高和手爪高度包围盒
                    # 由本节点叠加。最终 path（含运动学调整后的 APP）必须重新查顶。
                    for height_issue in _check_path_height(path, size, rp.H):
                        issue_text = (
                            f"路径{height_issue['label']}顶部超出车厢: "
                            f"点Z={height_issue['point_z']:.1f}mm + "
                            f"整抓高度={height_issue['carried_height']:.1f}mm = "
                            f"{height_issue['top_z']:.1f}mm，"
                            f"车厢高度={height_issue['car_height']:.1f}mm，"
                            f"超出={height_issue['over_height']:.1f}mm")
                        path_issues.append(issue_text)
                        logs.warning(f"[CHK-HEIGHT] Round.{action['id']} {issue_text}")
                    if obstacles:
                        X_chk = SearchSpace(np.array([(0, rp.L), (0, rp.W), (0, rp.H)]), obstacles)
                        # 只检查过渡/接近点（x0,x1,x_app），落箱点本身不做障碍检测
                        for pt_idx in range(3):
                            if not X_chk.obstacle_free((path[pt_idx][0], path[pt_idx][1], path[pt_idx][2]), size):
                                path_issues.append(
                                    f"路径点{pt_idx}干涉风险: "
                                    f"x={path[pt_idx][0]:.1f}, y={path[pt_idx][1]:.1f}, "
                                    f"z={path[pt_idx][2]:.1f}")
                                logs.warning (
                                    f"第{pt_idx}个点可能存在干涉风险！x: {path[pt_idx][0]}, y: {path[pt_idx][1]}, z: {path[pt_idx][2]}")
                        # P1 抓取余量检查：计算手抓左右间隙，任意一侧不足阈值时告警
                        _MIN_SIDE_GAP = config_be.get('min_side_gap', 100)
                        if action['area'] == 'p1':
                            y_start = action['pos'][1]
                            # y_grip_right：手抓实际占宽右边界，不含放置后的开缝间隙
                            y_grip_right = y_start + sum(action['num']) * action['size'][1]
                            cur_h = action['pos'][2]
                            if rp.block_type == 'mixture' and mixture_clearance is not None:
                                left_gap = mixture_clearance['left_gap']
                                right_gap = mixture_clearance['right_gap']
                                left_src = '箱' if mixture_clearance['left_is_box'] else '墙'
                                right_src = '箱' if mixture_clearance['right_is_box'] else '墙'
                                has_side_reference = True
                            else:
                                # 规则垛保持原来的同底面 Z 判断。
                                placed = [a for a in rp.ori_offsets
                                          if a != 'done'
                                          and a['area'] == 'p1'
                                          and a['pos'][2] == cur_h
                                          and a['num_F'] == action['num_F']
                                          and a['id'] < action['id']]
                                def _phys_right(a):
                                    return (a['pos'][1]
                                            + sum(a['num']) * a['size'][1]
                                            + sum(a.get('gaps', [])))
                                left_ends = [
                                    _phys_right(a) for a in placed
                                    if _phys_right(a) <= y_start]
                                right_starts = [
                                    a['pos'][1] for a in placed
                                    if a['pos'][1] >= y_grip_right]
                                p1_left_wall = 0
                                p1_right_wall = rp.W - rp.N2 * rp.l
                                y_placed_right = (
                                    y_grip_right + sum(action.get('gaps', [])))
                                left_gap = (y_start - max(left_ends)
                                            if left_ends else y_start - p1_left_wall)
                                right_gap = (min(right_starts) - y_grip_right
                                             if right_starts
                                             else p1_right_wall - y_placed_right)
                                left_src = '箱' if left_ends else '墙'
                                right_src = '箱' if right_starts else '墙'
                                has_side_reference = bool(left_ends or right_starts)
                            if has_side_reference:
                                # 期望方向：间隙小的一侧贴紧（left<=right → dir=1贴左，否则 dir=2贴右）
                                expected_dir = 1 if left_gap <= right_gap else 2
                                dir_ok = action['dir'] == expected_dir
                                dir_info = (f"dir={action['dir']}({'<-L' if action['dir']==1 else 'R->'})"
                                            f" 期望={expected_dir} {'✓' if dir_ok else '✗ 方向不符'}")
                                if left_gap + right_gap < _MIN_SIDE_GAP:
                                    path_issues.append(
                                        f"两侧间隙不足: left={left_gap:.1f}mm, "
                                        f"right={right_gap:.1f}mm, "
                                        f"阈值={_MIN_SIDE_GAP}mm")
                                    logs.warning(
                                        f"[CHK] Round.{action['id']} 两侧间隙之和不足！"
                                        f" left_gap={left_gap:.1f}mm({left_src})"
                                        f"  right_gap={right_gap:.1f}mm({right_src})"
                                        f"  之和={left_gap + right_gap:.1f}mm"
                                        f"  阈值={_MIN_SIDE_GAP}mm  {dir_info}")
                                else:
                                    log_fn = logs.info if dir_ok else logs.warning
                                    log_fn(
                                        f"[CHK] Round.{action['id']} 间隙 ok"
                                        f"  left_gap={left_gap:.1f}mm({left_src})"
                                        f"  right_gap={right_gap:.1f}mm({right_src})"
                                        f"  {dir_info}")
                                if not dir_ok:
                                    path_issues.append(
                                        f"放置方向不符: dir={action['dir']}, "
                                        f"expected={expected_dir}")
                    if path_issues:
                        logs.warning(
                            f"Round.{action['id']} check path发现"
                            f" {len(path_issues)} 项异常")
                    else:
                        logs.info("check path ok!")
                end_time = time.time()
                logs.debug("it cost {:.2f} s".format(end_time - start_time))

                if chk_session is not None:
                    chk_result = {
                        'block': rp_idx + 1,
                        'block_type': rp.block_type,
                        'face': int(action['num_F']),
                        'grab': int(action['id']),
                        'box_type': str(action.get('box_type', rp.box_type)),
                        'area': action['area'],
                        'dir': int(action['dir']),
                        'app': [round(float(value), 2) for value in x_app],
                        'goal': [round(float(value), 2) for value in x_goal],
                        'cost_sec': round(end_time - start_time, 4),
                        'issues': list(dict.fromkeys(path_issues)),
                        'physical_support': physical_support,
                    }
                    chk_session['results'].append(chk_result)

                # 发送路径数据：每个路径点编码为41字节块
                # 中间点 flag = 区域码，末点 flag = 动作码
                # action=2 表示当前 block 最后一抓
                # 若已是最后一个 block 发 3（全部结束）
                act_val = action['action']
                path_payload = b''.join(
                    b'\x01\x00\x00\x00\x00'
                    + struct.pack('!fff', round(pt[0], 2), round(pt[1], 2), round(pt[2], 2))
                    + b'\x00' * 24
                    for pt in path
                )
                info_block = _data_block(
                    float(_AREA_NUM[action['area']]),
                    float(act_val),
                    float(action['num'][0]),
                )
                server.send_message(_build_msg(len(path) + 1, path_payload + info_block + b'\x00\x00'))
    
                # 正常任务由 show_env 控制单抓 HTML；cmd_chk_path 检查混装面时
                # 即使 show_env=false 也必须逐抓保存，便于定位混装路径异常。
                _save_path_html = (
                    show_env
                    or (chk_session is not None and rp.block_type == 'mixture')
                )
                if _save_path_html:
                    try:
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        plot = Plot(
                            f"path_block{rp_idx + 1:02d}_face{action['num_F']:02d}_"
                            f"{ts}_id{action['id']}")
                        # Plot 方法只用 X.dimensions，可视化无需 rtree，用轻量对象代替
                        _X3 = type('_X', (), {'dimensions': 3})()
                        if path is not None:
                            # x0=浅绿 / x1=青 / APP=蓝 / goal=红 / 后续落点=橙
                            path_colors = (
                                ['lightgreen', 'cyan', 'blue', 'red']
                                + ['orange'] * (len(path) - 4))
                            plot.plot_path(_X3, path, size, colors=path_colors)
                        if be.display_objects:
                            plot.plot_obstacles(_X3, be.display_objects)
                        plot.plot_start(_X3, x0)
                        plot.plot_goal(_X3, x_goal)
                        plot.draw(auto_open=False)
                        if chk_session is not None:
                            chk_session['path_htmls'].append(plot.filename)
                        logs.info(f"单抓路径可视化已保存: {plot.filename}")
                    except Exception as _path_plot_error:
                        _plot_issue = f"单抓路径可视化保存失败: {_path_plot_error}"
                        if chk_session is not None:
                            chk_session['system_issues'].append(
                                f"Block {rp_idx + 1} / 面 {action['num_F']} / "
                                f"第 {action['id']} 抓: {_plot_issue}")
                        logs.warning(_plot_issue)

                # 面完成时保存整面PNG。show_env=true 时正常任务也保存；
                # cmd_chk_path 批量检查时无论 show_env 开关都保存。
                if action['action'] in (1, 2, 3) and (show_env or chk_session is not None):
                    try:
                        _face_issue_ids = set()
                        if chk_session is not None:
                            _face_issue_ids = {
                                item['grab'] for item in chk_session['results']
                                if item['block'] == rp_idx + 1
                                and item['face'] == int(action['num_F'])
                                and item['issues']
                            }
                        _face_stamp = (
                            chk_session['stamp'] if chk_session is not None
                            else time.strftime('%Y%m%d_%H%M%S'))
                        _saved_images = _save_face_visualizations(
                            rp,
                            block_number=rp_idx + 1,
                            face_number=int(action['num_F']),
                            stamp=_face_stamp,
                            issue_ids=_face_issue_ids,
                            save_face=True,
                            save_mixture=(rp.block_type == 'mixture'),
                        )
                        if chk_session is not None:
                            chk_session['face_images'].extend(_saved_images)
                        for _image_path in _saved_images:
                            logs.info(f"面级可视化已保存: {_image_path}")
                    except Exception as _face_plot_error:
                        _plot_issue = f"面级可视化保存失败: {_face_plot_error}"
                        if chk_session is not None:
                            chk_session['system_issues'].append(
                                f"Block {rp_idx + 1} / 面 {action['num_F']}: "
                                f"{_plot_issue}")
                        logs.warning(_plot_issue)

                # 更新环境
                be.step(action)
                if action['action'] == 1:
                    logs.info("*** Block.{} NO.{} 码垛面 结束！***".format(rp_idx + 1, action['num_F']))
                    logs.info("*** Block.{} NO.{} 码垛面 开始！***".format(rp_idx + 1, action['num_F'] + 1))
                elif action['action'] == 2:
                    logs.info("*** Block.{} NO.{} 码垛面 结束！***".format(rp_idx + 1, action['num_F']))
                    logs.info("当前 block 路径规划结束，切换到下一 block !")
                    # 发完最后一抓路径后立即切换，下次 get_path 直接返回新 block 第一条
                    rp.robot_offsets.clear()   # 清掉剩余的 'done'
                    rp_idx += 1
                    if rp_idx < len(rp_list):
                        rp = rp_list[rp_idx]
                        # 断点续传：切到新 block，游标重置并落盘（新 block 尚未消费）
                        cur_box_id = cur_path_id = 0
                        if store and not off_line_mode and resume_save:
                            store.save_cursor(rp_idx, cur_box_id, cur_path_id)
                        # chk_value > 0 说明当前处于批量预取模式，继续对新 block 批量处理
                        if chk_value > 0:
                            # len-1 排除末尾 'done'，+1 补偿本次循环末尾的 chk_value-=1
                            chk_value = len(rp.robot_offsets) - 1 + 1
                            logs.info("已切换到 block {}，继续批量 check path（共{}抓）".format(rp_idx + 1, chk_value - 1))
                        else:
                            logs.info("已切换到 block {}".format(rp_idx + 1))
                elif action['action'] == 3:
                    logs.info("*** Block.{} NO.{} 码垛面 结束！***".format(rp_idx + 1, action['num_F']))
                    logs.info("所有 block 路径规划结束 !")
                    # 断点续传：全部完成，清除续传文件，避免下次误触发
                    if store and not off_line_mode and resume_save:
                        store.clear()
                        logs.info("码垛全部完成，已清除断点续传记录")
                if chk_value != 0:
                    chk_value -= 1
                    if chk_value == 0 and chk_session is not None:
                        try:
                            _chk_summary = _write_chk_path_summary(chk_session)
                            _log_chk_path_summary(_chk_summary, logs)
                        except Exception as _chk_summary_error:
                            logs.warning(
                                f"cmd_chk_path 汇总保存失败，仅保留运行日志: "
                                f"{_chk_summary_error}")
                        finally:
                            chk_session = None
            elif mes_hex == cmd_stacking:
                # 双雷达采集 → 堆叠检测 → 发送结果 → 保存点云（分段计时）
                pc1 = pc2 = None
                t_collect = t_compute = t_save = 0.0
                try:
                    # 当前抓箱子总宽度：箱数 × 单箱宽（p3侧立时宽度方向为h=size[2]，p1为w=size[1]）
                    # 用 last_grab_action 自带的 size/箱型（多 block 边界后 rp 已切换，不能用 rp.w/h/box_type）
                    # 保护：未触发过 get_path（无当前抓）时理论宽度发 -1，仍照常测量并上报实测
                    _rel_top_h = None    # 当前行顶面距地板高度(米)，供测宽锁定当前行 Z
                    _box_h_m = None      # 当前抓箱子竖向高度(米)
                    _single_box_width = None  # 当前姿态下单箱沿垛面宽度方向的尺寸(mm)
                    if last_grab_action is None or last_grab_action == 'done':
                        _grab_width = -1
                        _box_type = (rp.robot_offsets[0].get('box_type', rp.box_type)
                                     if rp.robot_offsets and rp.robot_offsets[0] != 'done'
                                     else rp.box_type)
                    else:
                        _cur = last_grab_action
                        _n = sum(_cur['num'])
                        _single_box_width = (
                            _cur['size'][2] if _cur['area'] == 'p3' else _cur['size'][1])
                        _grab_width = _n * _single_box_width
                        _box_type = last_grab_box_type
                        # 当前行高度：pos[2] 为放置点底面距地板高度(mm)，加箱竖向跨度=顶面相对高度
                        # p3 侧立时竖向跨度为原始宽 size[1]，p1 竖放时为原始高 size[2]
                        _box_vspan = _cur['size'][1] if _cur['area'] == 'p3' else _cur['size'][2]
                        _rel_top_h = (_cur['pos'][2] + _box_vspan) / 1000.0
                        _box_h_m = _box_vspan / 1000.0
                    _t0 = time.time()
                    pc1, pc2 = collect_dual_lidar_once('/lidar/JT128_1', '/lidar/JT128_2', frames=3)
                    t_collect = time.time() - _t0
                    # payload: [status(1=计算成功/2=计算失败或报错), 理论宽度(mm), 测量宽度(mm)]
                    if pc1 is None or pc2 is None:
                        # 雷达 topic 未发布 / 采集超时 → 按计算失败处理
                        logs.warning(f'堆叠检测：雷达点云采集超时（topic 未发布），理论宽度={_grab_width:.1f}mm，发送 status=2')
                        server.send_message(_build_msg(
                            1, _data_block(2.0, float(_grab_width), 0.0) + b'\x00\x00'))
                        measured = None
                    else:
                        _t1 = time.time()
                        # 偏航补偿角：由该抓所属 block 箱型对应的拍照位 J1 与正对 J1 之差决定
                        _yaw_off = _yaw_offset_for_box(_box_type)
                        measured = check_stacking(_grab_width, pc1, pc2, yaw_offset_deg=_yaw_off,
                                                  rel_top_h=_rel_top_h, box_h=_box_h_m,
                                                  box_width_mm=_single_box_width,
                                                  log_callback=logs.warning, view=False,
                                                  box_type=_box_type)
                        t_compute = time.time() - _t1
                        if measured is None:
                            # 宽度计算失败：发送 status=2（不再依赖异常兜底）
                            logs.warning(f'堆叠检测：宽度计算失败，理论宽度={_grab_width:.1f}mm，发送 status=2')
                            server.send_message(_build_msg(
                                1, _data_block(2.0, float(_grab_width), 0.0) + b'\x00\x00'))
                        else:
                            server.send_message(_build_msg(
                                1, _data_block(1.0, float(_grab_width), float(measured)) + b'\x00\x00'))
                            logs.info(f'堆叠检测：计算成功，理论宽度={_grab_width:.1f}mm，'
                                      f'测量宽度={measured:.1f}mm，发送 status=1')
                except Exception as _e:
                    logs.error(f'堆叠检测异常：{type(_e).__name__}: {_e}')
                    try:
                        server.send_message(_build_msg(1, _data_block(2.0, 0.0, 0.0) + b'\x00\x00'))
                    except Exception as _e2:
                        logs.error(f'堆叠检测异常兜底发送也失败（机器人可能已断连）：{type(_e2).__name__}: {_e2}')
                # 点云保存（不影响已发送结果；只要采到点云就保存便于事后分析）
                if pc1 is not None and pc2 is not None:
                    try:
                        _t2 = time.time()
                        save_point_clouds(pc1, pc2)
                        t_save = time.time() - _t2
                    except Exception as _e:
                        logs.error(f'点云保存异常：{type(_e).__name__}: {_e}')
                logs.info(f'cmd_stacking 耗时：采集拼接={t_collect:.2f}s '
                          f'计算={t_compute:.2f}s 保存={t_save:.2f}s '
                          f'总={t_collect + t_compute + t_save:.2f}s')

if __name__ == '__main__':
    main()
