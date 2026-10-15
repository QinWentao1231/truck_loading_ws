"""什邡原版垛序适配：只转换动作结构，不改变抓次、方向、坐标或尾料姿态。"""

import copy
import math

from .legacy_sequence import OriginalRobotPosition


def _size(message):
    return {key: float(getattr(message, key)) for key in ('L', 'W', 'H')}


def parse_legacy_request(request):
    """旧接口仍使用单箱型、regular[0]和后续trapezoid，不增加proto字段。"""
    if len(request.condition.boxes) != 1:
        raise ValueError('旧接口仅支持一个单字符箱型')
    if request.answer.mixture:
        raise ValueError('旧版垛序没有混装面处理规则')
    if not request.answer.regular:
        raise ValueError('原版垛序需要 regular[0]（仅梯形订单也需提供常规区占位配置）')
    box, car = request.condition.boxes[0], request.condition.car
    entry = request.answer.regular[0]
    regular = {key: int(getattr(entry, key)) for key in
               ('N1', 'N2', 'N3', 'T12', 'T3', 'F13', 'F2', 'E', 'Nx')}
    regular.update(Stack=[list(v.Width) for v in entry.Stack],
                   Group=[list(v.Unit) for v in entry.Group])
    config = {
        'box_list': [box.type],
        'box': {box.type: {'size': _size(box.original)}},
        'car': {'size': _size(car.original), 'reserve': _size(car.reserve)},
        'regular': [regular], 'trapezoid': [],
    }
    for entry in request.answer.trapezoid:
        # 沿用旧callback只保留N1非零条目的规则。
        if entry.N1 == 0:
            continue
        values = {key: int(getattr(entry, key)) for key in ('N1', 'N3', 'T1', 'T3', 'Nx')}
        values.update(Stack=[list(v.Width) for v in entry.Stack],
                      Group=[list(v.Unit) for v in entry.Group], Isdoor=bool(entry.Isdoor))
        config['trapezoid'].append(values)
    return normalize_legacy_order(config)


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{label} 必须是非负整数')
    if not math.isfinite(value) or int(value) != value or value < 0:
        raise ValueError(f'{label} 必须是非负整数')
    return int(value)


def _pattern(entry, label):
    n1 = entry['N1']
    if len(entry.get('Group', [])) != 2 or len(entry.get('Stack', [])) != 2:
        raise ValueError(f'{label} 必须提供两套 Group/Stack')
    for index, (group, stack) in enumerate(zip(entry['Group'], entry['Stack'])):
        if not group or sum(group) != n1:
            raise ValueError(f'{label}.Group[{index}] 箱数合计必须等于N1={n1}')
        if any(_integer(v, label) <= 0 or v > 4 for v in group):
            raise ValueError(f'{label}.Group 单抓必须在1～4箱内')
        if len(stack) != n1 + 1 or any(not math.isfinite(v) or v < 0 for v in stack):
            raise ValueError(f'{label}.Stack 长度必须为N1+1且为有限非负数')


def normalize_legacy_order(config):
    """字段检查不替代原版生成规则；不叠加箱体reserve，不读取订单grip/frame。"""
    data = copy.deepcopy(config)
    types = data.get('box_list', [])
    if len(types) != 1 or not isinstance(types[0], str) or len(types[0]) != 1:
        raise ValueError('旧机器人箱型必须为单字符，例如A')
    if data.get('mixture'):
        raise ValueError('原版垛序不支持Mixture')
    for label, item in (('box', data['box'][types[0]]), ('car', data['car'])):
        for key in ('L', 'W', 'H'):
            value = float(item['size'][key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{label}.size.{key} 必须为有限正数')
            item['size'][key] = value
    if not data.get('regular'):
        raise ValueError('原版需要regular[0]；仅梯形时提供F13=0的占位配置')
    data['regular'] = [data['regular'][0]]
    data.setdefault('trapezoid', [])
    for kind, fields in (
            ('regular', ('N1', 'N2', 'N3', 'T12', 'T3', 'F13', 'F2', 'E', 'Nx')),
            ('trapezoid', ('N1', 'N3', 'T1', 'T3', 'Nx'))):
        for index, entry in enumerate(data[kind]):
            for key in fields:
                if key not in entry:
                    raise ValueError(f'{kind}[{index}]缺少{key}，不能推测旧垛型')
                entry[key] = _integer(entry[key], f'{kind}.{key}')
            if kind == 'regular' or not entry.get('Isdoor', False):
                _pattern(entry, f'{kind}[{index}]')
            if kind == 'trapezoid':
                entry['Isdoor'] = bool(entry.get('Isdoor', False))
    return data


class RobotPosition(OriginalRobotPosition):
    """使用原版read_robot_offset；num变成单元素列表以供规则路径计算。"""

    sequence_strategy = 'legacy'

    def __init__(self, config):
        data = normalize_legacy_order(config)
        super().__init__(data)
        for action in self.robot_offsets:
            if action == 'done':
                continue
            action['num'] = [action['num']]
            action['gaps'] = []
            action['box_type'] = self.box_type
            action['grab_num_p1'] = self.grab_num_p1
            action['car_width'] = float(self.W)
            action['p1_right_wall'] = float(self.W)
        for action, box in zip(self.robot_offsets, self.boxes):
            box['size'] = list(action['size'])
            box['box_type'] = self.box_type
        actual_count = sum(sum(a['num']) for a in self.robot_offsets if a != 'done')
        if actual_count != self.box_count:
            # 继续保留上一轮迁移已有的数量一致性检查，不把旧版遗漏的P2/E3静默下发。
            raise ValueError(
                f'原版垛序箱数不一致：订单{self.box_count}箱，实际生成{actual_count}箱；'
                '请检查P2、E=3或尾料配置，原版规则不完整支持这些情况')
        self.face_count = self.robot_offsets[-2]['num_F']
        self.ori_offsets = copy.deepcopy(self.robot_offsets)
        # 当前面成功下发的落点；不覆盖原始垛序，避免重复补偿或改变最右抓判定。
        self.sent_positions = {}
