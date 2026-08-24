"""垛序动作生成与箱体碰撞环境。

``RobotPosition`` 把 regular、trapezoid 或 mixture block 展开为逐抓动作和来料配方；
``BinEnv`` 把已放动作转换为真实/外扩 AABB，用于路径、侧向间隙和混装物理支撑
检查。该模块的长度统一使用毫米。
"""

import copy
import logging
import math
from collections import defaultdict

_logger = logging.getLogger(__name__)


def _mixture_items(mixture_face):
    """返回 Mixture 的放置信息列表。"""
    return mixture_face.get('Items', mixture_face.get('items', []))


def _decode_mixture_placement(box_type, encoded_num):
    """解码混装条目的放置区域和真实箱数。

    常规箱(1XX)的 Num>=10 表示按 P3 侧立方式放置，个位数是实际箱数；
    Num 原值仍需保留给 cmd_get_box。其他箱型和数值维持 P1 语义。
    """
    encoded_num = int(encoded_num)
    if str(box_type).startswith('1') and encoded_num >= 10:
        return 'p3', encoded_num % 10
    return 'p1', encoded_num


def _mixture_box_count(mixture_face):
    """统计一个混装面的真实箱数，去除 P3 数量编码中的十位标志。"""
    return sum(
        _decode_mixture_placement(
            item.get('Type', item.get('type', '')),
            item.get('Num', item.get('num', 0)),
        )[1]
        for item in _mixture_items(mixture_face)
    )


def _is_head_entry(entry):
    """读取下一层垛型条目的异形车头标志，兼容常见键名。"""
    return bool(entry.get(
        'Ishead', entry.get('isHead', entry.get('ishead', False))))


def _physical_box_length(block, box_type):
    """返回箱型沿车厢纵深方向的物理长度，不叠加建模 reserve。"""
    box_configs = block.get('box', {}) or {}
    box_type = str(box_type or '')
    if not box_type:
        box_list = block.get('box_list', []) or []
        if box_list:
            box_type = str(box_list[0])
        elif len(box_configs) == 1:
            box_type = str(next(iter(box_configs)))
    if box_type not in box_configs:
        raise ValueError(f"异形车头纵深计算找不到箱型 {box_type or '空'}")
    size = box_configs[box_type].get('size', {}) or {}
    length = float(size.get('L', size.get('l', 0.0)))
    if not math.isfinite(length) or length <= 0:
        raise ValueError(f"箱型 {box_type} 的物理长度L无效: {length}")
    return length


def _block_face_specs(block):
    """按真实生成顺序返回一个 block 的各面纵深和显式 Ishead 标记。

    常规垛每个 F13 面以及 E=1 的前置尾料面各占一个箱长；梯形每条
    Trapezoid 占一个箱长；混装面取所有条目 ``Pos.X + 箱长`` 的最大值，
    因而不同箱型或前后错位时不会被平均箱长低估。
    """
    regular = block.get('regular', []) or []
    trapezoid = block.get('trapezoid', []) or []
    mixture = block.get('mixture', []) or []
    specs = []

    if regular:
        for entry in regular:
            length = _physical_box_length(block, entry.get('Type', ''))
            is_head = _is_head_entry(entry)
            for _ in range(max(0, int(entry.get('F13', 0)))):
                specs.append({'depth': length, 'explicit_is_head': is_head})
            if (int(entry.get('E', 0)) == 1 and
                    int(entry.get('Nx', 0)) > 0):
                specs.append({'depth': length, 'explicit_is_head': is_head})
        return specs

    if trapezoid:
        for entry in trapezoid:
            specs.append({
                'depth': _physical_box_length(
                    block, entry.get('Type', '')),
                'explicit_is_head': _is_head_entry(entry),
            })
        return specs

    if mixture:
        for face_index, face in enumerate(mixture, start=1):
            max_end = 0.0
            is_head = _is_head_entry(face)
            items = _mixture_items(face)
            for item in items:
                pos = item.get('Pos', item.get('pos', {})) or {}
                pos_x = float(pos.get('X', pos.get('x', 0.0)))
                if not math.isfinite(pos_x) or pos_x < 0:
                    raise ValueError(
                        f"Mixture 面{face_index}的Pos.X无效: {pos_x}")
                max_end = max(
                    max_end,
                    pos_x + _physical_box_length(
                        block, item.get('Type', item.get('type', ''))),
                )
                is_head = is_head or _is_head_entry(item)
            specs.append({
                'depth': max_end,
                'explicit_is_head': is_head,
            })
        return specs

    return specs


def _attach_head_face_geometry(blocks):
    """为完整订单的每个面附加异形车头几何信息。

    block/面按规划顺序视为从车头向车尾排列。第一个面从纵深0开始，下一面
    起点由上一面真实占用纵深推进。``CarCondition.head`` 有效时，未显式标记
    Ishead 的前部面也按累计纵深自动识别；Regular/Mixture/Trapezoid 下的
    Ishead 则作为显式标记。宽度取当前面靠车头一侧的最窄值。
    """
    if not blocks:
        return
    first_car = blocks[0].get('car', {}) or {}
    body_size = first_car.get('size', {}) or {}
    body_width = float(body_size.get('W', body_size.get('w', 0.0)))
    if not math.isfinite(body_width) or body_width <= 0:
        raise ValueError(f"车厢原始宽度W无效: {body_width}")

    head_cfg = first_car.get('head', {}) or {}
    head_length = float(head_cfg.get('L', head_cfg.get('l', 0.0)))
    head_width = float(head_cfg.get('W', head_cfg.get('w', 0.0)))
    head_height = float(head_cfg.get('H', head_cfg.get('h', 0.0)))
    head_configured = any(
        abs(value) > 1e-6 for value in
        (head_length, head_width, head_height))

    block_specs = [_block_face_specs(block) for block in blocks]
    explicit_head = any(
        spec['explicit_is_head']
        for specs in block_specs for spec in specs)
    if head_configured or explicit_head:
        if (not math.isfinite(head_length) or head_length <= 0 or
                not math.isfinite(head_width) or head_width <= 0 or
                head_width > body_width):
            raise ValueError(
                "异形车头尺寸无效：需要 head.L>0、0<head.W<=original.W，"
                f"当前head.L={head_length}, head.W={head_width}, "
                f"original.W={body_width}")
        has_head_geometry = True
    else:
        has_head_geometry = False

    depth_x = 0.0
    for block, specs in zip(blocks, block_specs):
        geometry = {}
        for face_number, spec in enumerate(specs, start=1):
            inferred = (
                has_head_geometry and depth_x < head_length - 1e-6)
            explicit = bool(spec['explicit_is_head'])
            is_head = explicit or inferred
            if is_head and has_head_geometry:
                ratio = min(max(depth_x / head_length, 0.0), 1.0)
                car_width = head_width + (
                    body_width - head_width) * ratio
            else:
                car_width = body_width
            geometry[face_number] = {
                'depth_x': depth_x,
                'face_depth': float(spec['depth']),
                'car_width': float(car_width),
                'is_head': bool(is_head),
                'explicit_is_head': explicit,
                'source': (
                    'explicit+depth' if explicit and inferred
                    else 'explicit' if explicit
                    else 'depth' if inferred
                    else 'normal'),
            }
            depth_x += float(spec['depth'])
        block['_head_face_geometry'] = geometry


def build_robot_positions(config_data):
    """按完整订单构造 RobotPosition 列表，并跨 block 连续计算异形宽度。"""
    blocks = config_data if isinstance(config_data, list) else [config_data]
    prepared = copy.deepcopy(blocks)
    _attach_head_face_geometry(prepared)
    return [RobotPosition(block) for block in prepared]


class Box:
    """按放置姿态保存单箱尺寸和左下前角位置。"""

    def __init__(self, l, w, h, rotation_type, position):
        """创建箱体；rotation_type 0/1/2 分别表示原向、平转和 P3 侧立。"""
        if rotation_type == 0:
            self.length = l
            self.width = w
            self.height = h
        elif rotation_type == 1:
            self.length = w
            self.width = l
            self.height = h
        elif rotation_type == 2:
            # 侧立：箱子绕 x 轴旋转 90°，原高度 h 变为 y 跨度（width），原宽度 w 变为 z 跨度（height）
            self.length = l
            self.width = h
            self.height = w
        else:
            raise ValueError(f"不支持的 rotation_type: {rotation_type}")
        self.position = position


class RobotPosition:
    """单个 block 的垛型解析器。block 类型互斥：regular / trapezoid / mixture，
    仅对应字段会被读取和解析。"""

    def __init__(self, config_data):
        """读取单个 block 配置并立即生成完整的逐抓队列。"""
        # 单独构造 RobotPosition（测试/离线工具）时也生成面级信息；正式主流程
        # 使用 build_robot_positions 一次处理完整订单，保证纵深跨 block 连续。
        if '_head_face_geometry' not in config_data:
            prepared = copy.deepcopy(config_data)
            _attach_head_face_geometry([prepared])
            config_data = prepared
        # 箱型尺寸会叠加订单中的 reserve，后续动作和碰撞环境均使用有效尺寸。
        self.box_configs = config_data['box']
        self.box_type = config_data['box_list'][0]
        box_cfg = self.box_configs[self.box_type]
        rsv = box_cfg.get('reserve', {})
        self.l = box_cfg['size']['L'] + rsv.get('L', 0)
        self.w = box_cfg['size']['W'] + rsv.get('W', 0)
        self.h = box_cfg['size']['H'] + rsv.get('H', 0)
        self.Nt = box_cfg.get('Nt', 0)
        self.box_size = [self.l, self.w, self.h]
        grip = box_cfg.get('grip', {})
        self.grab_num_p1 = grip['P1'][0] if grip.get('P1') else 4
        self.grab_num_p2 = grip['P2'][0] if grip.get('P2') else 2
        self.grab_num_p3 = grip['P3'][0] if grip.get('P3') else 2
        # 车厢尺寸和侧向预留，单位均为毫米。
        self.L = round(config_data['car']['size']['L'])
        self.W = round(config_data['car']['size']['W'])
        self.H = round(config_data['car']['size']['H'])
        self.RW = round(config_data['car']['reserve']['W'])
        head_cfg = config_data['car'].get('head', {}) or {}
        self.head = {
            'L': float(head_cfg.get('L', head_cfg.get('l', 0.0))),
            'W': float(head_cfg.get('W', head_cfg.get('w', 0.0))),
            'H': float(head_cfg.get('H', head_cfg.get('h', 0.0))),
        }

        # block 类型与对应字段（互斥）
        self.regular   = config_data.get('regular',   []) or []
        self.trapezoid = config_data.get('trapezoid', []) or []
        self.mixture   = config_data.get('mixture',   []) or []
        self.block_type = self._detect_block_type()
        self.head_face_geometry = {
            int(face_number): dict(values)
            for face_number, values in
            (config_data.get('_head_face_geometry', {}) or {}).items()
        }
        # Block 本身没有 Ishead；它位于 Regular/Mixture/Trapezoid 下一层。
        # block 是否处于异形区域由完整订单预处理后的面级结果汇总。
        self.is_head = any(
            bool(values.get('is_head'))
            for values in self.head_face_geometry.values())

        # regular 参数（仅 regular block 有效，其他类型置 0 以兼容 box_count 等公式）
        # N1/N2/N3: P1每层/P2每列/P3每行箱数；T12/T3: P1-P2/P3层数；
        # F13/F2: P1-P3/P2 面数；E: 尾料方式(0无 1前置P1 2顶置P3 3顶置P1)；Nx: 尾料箱总数
        # Stack: 缝隙比例列表；Group: 每抓箱数列表（奇/偶层各一套）
        if self.block_type == 'regular':
            reg0 = self.regular[0]
            self.N1  = reg0['N1']
            self.N2  = reg0['N2']
            self.N3  = reg0['N3']
            self.T12 = reg0['T12']
            self.T3  = reg0['T3']
            self.F13 = reg0['F13']
            self.F2  = reg0['F2']
            self.E   = reg0['E']
            self.Nx  = reg0['Nx']
            self.Stack = reg0['Stack']
            self.Group = reg0['Group']
            self.reg_type = reg0.get('Type', '')
        else:
            self.N1 = self.N2 = self.N3 = 0
            self.T12 = self.T3 = self.F13 = self.F2 = 0
            self.E = self.Nx = 0
            self.Stack = []
            self.Group = []
            self.reg_type = ''

        # 当前 block 箱数（block 互斥；regular 累加所有条，支持多条 regular）
        self.box_count = (
            sum(r['N1'] * r['T12'] * r['F13'] + r['N2'] * r['T12'] * r['F2']
                + r['N3'] * r['T3'] * r['F13'] + r['Nx'] for r in self.regular)
            + sum(x['N1'] * x['T1'] + x['N3'] * x['T3'] + x['Nx'] for x in self.trapezoid)
            + sum(_mixture_box_count(y) for y in self.mixture))

        # robot_offsets: 运行时抓取队列；ori_offsets: 完整快照（间隙计算用）
        # boxes: 来料配方队列（与robot_offsets一一对应，供cmd_get_box消费）
        self.robot_offsets = []
        self.fit_offsets = []
        self.ori_offsets = []
        self.boxes = []
        self.paths = []
        self._id = 0
        self._face_p1_right_walls = {}

        self.read_robot_offset()

    def _detect_block_type(self):
        """block 类型互斥：按优先级返回唯一命中的类型。"""
        if self.regular:
            return 'regular'
        if self.trapezoid:
            return 'trapezoid'
        if self.mixture:
            return 'mixture'
        return None

    # ── 辅助方法 ────────────────────────────────────────────────

    def _box_params(self, box_type):
        """返回指定箱型的有效尺寸与抓取配置（已叠加 reserve）。"""
        if box_type not in self.box_configs:
            raise ValueError(f"未找到箱型 {box_type} 的尺寸配置")
        cfg = self.box_configs[box_type]
        rsv = cfg.get('reserve', {})
        size = [
            cfg['size']['L'] + rsv.get('L', 0),
            cfg['size']['W'] + rsv.get('W', 0),
            cfg['size']['H'] + rsv.get('H', 0),
        ]
        grip = cfg.get('grip', {})
        return {
            'size': size,
            'grab_p1': grip['P1'][0] if grip.get('P1') else 4,
            'grab_p2': grip['P2'][0] if grip.get('P2') else 2,
            'grab_p3': grip['P3'][0] if grip.get('P3') else 2,
        }

    def face_car_width(self, num_F):
        """返回指定面的可用车宽；普通区域保持原始车厢宽度。"""
        geometry = self.head_face_geometry.get(int(num_F), {})
        return float(geometry.get('car_width', self.W))

    def current_face_car_width(self):
        """返回当前待执行面的车宽，队列为空时回退原始车宽。"""
        if self.boxes:
            return float(self.boxes[0].get('car_width', self.W))
        for action in self.robot_offsets:
            if action != 'done':
                return float(action.get('car_width', self.W))
        return float(self.W)

    def _emit(self, area, num, num_F, dir_, pos, is_tail=False,
              box_type=None, box_size=None, box_num_signal=None):
        """向 robot_offsets 和 boxes 同步追加一条动作记录。
        robot_offsets 中 num 存列表形式（方便机器人侧按段读取），
        boxes 中 num 存整数，并保存当前抓的箱型和有效尺寸；箱型信号（+10）：
          - 20x / 30x 箱型：任意区域 +10
          - 10x 箱型：仅 P3 区域 +10
        box_num_signal 非空时，boxes 中保留该原始数字供 cmd_get_box 下发。
        """
        self._id += 1
        num_int = sum(num) if isinstance(num, list) else num
        num_list = num if isinstance(num, list) else [num]
        action_box_type = self.box_type if box_type is None else box_type
        params = self._box_params(action_box_type)
        action_box_size = list(params['size'] if box_size is None else box_size)
        if box_num_signal is None:
            box_first = str(action_box_type)[0]
            if box_first in ('2', '3') or (box_first == '1' and area == 'p3'):
                num_int += 10
        else:
            num_int = int(box_num_signal)
        face_geometry = self.head_face_geometry.get(int(num_F), {})
        car_width = float(face_geometry.get('car_width', self.W))
        p1_right_wall = float(
            self._face_p1_right_walls.get(int(num_F), car_width))
        head_depth_x = float(face_geometry.get('depth_x', 0.0))
        is_head = bool(face_geometry.get('is_head', False))
        self.robot_offsets.append({
            'id': self._id, 'area': area, 'num': num_list, 'gaps': [], 'num_F': num_F,
            'action': 0, 'dir': dir_, 'pos': pos, 'size': action_box_size,
            'box_type': action_box_type, 'grab_num_p1': params['grab_p1'],
            'car_width': car_width, 'p1_right_wall': p1_right_wall,
            'head_depth_x': head_depth_x, 'is_head': is_head,
        })
        self.boxes.append({
            'id': self._id, 'area': area, 'num': num_int, 'num_F': num_F,
            'action': 0, 'area_cfg': 0, 'is_tail': is_tail,
            'is_two_grab_row_last': False,
            'box_type': action_box_type, 'size': action_box_size,
            'car_width': car_width, 'head_depth_x': head_depth_x,
            'is_head': is_head,
        })

    @staticmethod
    def _split_grabs(N, grab_num, min2=False):
        """将 N 个箱子按 grab_num 分批，返回每批数量列表。
        min2=True 时首批最小为 2（防止单箱抓取）。
        首批取余数使后续批次恰好整除，余量不足一批时单独成批。"""
        if N == 0:
            return []
        first = N % grab_num if N % grab_num != 0 else grab_num
        if min2 and first == 1:
            first = 2
        result = [first]
        rem = N - first
        result += [grab_num] * (rem // grab_num)
        leftover = rem % grab_num
        if leftover:
            result.append(leftover)
        return result

    @staticmethod
    def _split_grabs_tail(N, grab_num):
        """前置尾料专用分批规则：
        - 第一批 = N % grab_num（余0则取满批 grab_num）
        - 后续批 = grab_num（满抓）
        - 余数==1时首批补为2、次批为grab_num-1，避免单箱抓取
        - N <= grab_num 时整批一次抓（不强制拆分）
        """
        if N <= 0:
            return []
        if N <= grab_num:
            return [N]
        remainder = N % grab_num
        full = N // grab_num
        if remainder == 0:
            return [grab_num] * full
        elif remainder == 1:
            # 余1：首批补2，次批grab_num-1，避免单箱抓取
            if full == 1:
                if grab_num - 1 <= 1:
                    return [N]
                return [2, grab_num - 1]
            return [2, grab_num - 1] + [grab_num] * (full - 1)
        else:
            return [remainder] + [grab_num] * full

    @staticmethod
    def _layer_pattern(f, t, stack_src, group_src):
        """按面号+层号选取 Stack/Group 模板（奇偶层交替，相邻面起始模板相反）。
        idx = (f + t) % 2：面0层0→0，面0层1→1，面1层0→1，面1层1→0，以此类推。"""
        idx = (f + t) % 2
        return copy.deepcopy(group_src[idx]), copy.deepcopy(stack_src[idx])

    def _emit_p1_groups(self, g, s, gap_, height, num_F, n1,
                        box_type=None, box_size=None, pattern_index=0):
        """按 Stack/Group 分组方式发出一层 P1 的所有抓取动作。
        n1: 本条 regular/梯形的 N1（P1 每层箱数），用于算右边界（多条 regular 时各条不同）。
        通常顺序：左(0) → 右(last) → 中间(1..last-1)。仅第二套模板
        Group[1]/Stack[1] 恰好为两抓时，使用右(1) → 左(0)。
        g 按物理位置从左到右给出每组箱数，visit_order 决定实际访问顺序。
        Stack 长度为 N1+1：s[i] 表示第i箱左侧缝隙占比（s[0]=左墙到第0箱，s[N1]=最后一箱到右墙，不参与位置计算）。
        pos = gap * sum(s[:base+1]) * 0.01 + base * w
        若 Stack 在组内部有非零比例（缝隙落在一次抓取中间），num 按缝隙位置拆成子数组，
        gaps 存每段之间的缝隙 mm，机器人侧按 num/gaps 分段放置。"""
        action_box_size = self.box_size if box_size is None else box_size
        box_w = action_box_size[1]
        n_groups = len(g)
        if n_groups == 2 and pattern_index == 1:
            visit_order = [1, 0]
        elif n_groups <= 2:
            visit_order = list(range(n_groups))
        else:
            # 先固定左右两端，再从左往右补齐中间。
            # 例：3组→[0,2,1]，4组→[0,3,1,2]，5组→[0,4,1,2,3]
            visit_order = [0, n_groups - 1] + list(range(1, n_groups - 1))
        placed = set()
        for step, n in enumerate(visit_order):
            base = sum(g[:n])
            num_boxes = g[n]
            pos = gap_ * sum(s[:base + 1]) * 0.01 + base * box_w
            # 第二套两抓模板改为右先左后，两抓均结合已放邻组和两侧边界，
            # 向间隙更小的一侧对齐（相等取 dir_=1）。第一套模板保持原规则。
            if n_groups == 2 and pattern_index == 1:
                p1_right_wall = gap_ + n1 * box_w
                if (n - 1) in placed:
                    base_left = sum(g[:n - 1])
                    left_boundary = (
                        gap_ * sum(s[:base_left + 1]) * 0.01
                        + (base_left + g[n - 1]) * box_w
                    )
                else:
                    left_boundary = 0
                if (n + 1) in placed:
                    base_right = sum(g[:n + 1])
                    right_boundary = (
                        gap_ * sum(s[:base_right + 1]) * 0.01
                        + base_right * box_w
                    )
                else:
                    right_boundary = p1_right_wall
                left_gap = pos - left_boundary
                right_gap = right_boundary - (pos + g[n] * box_w)
                dir_ = 1 if left_gap <= right_gap else 2
            # 三抓及以上保持原规则：先左、再右，后续抓根据间距决定。
            elif step == 0:
                dir_ = 1
            elif step == 1 and n_groups > 2:
                dir_ = 2
            else:
                left_placed = (n - 1) in placed
                right_placed = (n + 1) in placed
                p1_right_wall = gap_ + n1 * box_w  # P1 区右边界（y=0 为左墙）
                if left_placed and right_placed:
                    # 两侧均已放，计算物理间隙，向间隙小的一侧对齐；相等时取 dir_=1
                    base_left = sum(g[:n - 1])
                    left_end = gap_ * sum(s[:base_left + 1]) * 0.01 + (base_left + g[n - 1]) * box_w
                    left_gap = pos - left_end

                    base_right = sum(g[:n + 1])
                    right_start = gap_ * sum(s[:base_right + 1]) * 0.01 + base_right * box_w
                    right_gap = right_start - (pos + g[n] * box_w)

                    dir_ = 1 if left_gap <= right_gap else 2
                elif left_placed:
                    # 右侧无邻组，以车厢壁为右边界
                    base_left = sum(g[:n - 1])
                    left_end = gap_ * sum(s[:base_left + 1]) * 0.01 + (base_left + g[n - 1]) * box_w
                    left_gap = pos - left_end
                    right_gap = p1_right_wall - (pos + g[n] * box_w)
                    dir_ = 1 if left_gap <= right_gap else 2
                elif right_placed:
                    # 左侧无邻组，以 y=0 左墙为左边界
                    left_gap = pos
                    base_right = sum(g[:n + 1])
                    right_start = gap_ * sum(s[:base_right + 1]) * 0.01 + base_right * box_w
                    right_gap = right_start - (pos + g[n] * box_w)
                    dir_ = 1 if left_gap <= right_gap else 2
                else:
                    dir_ = 1
            placed.add(n)
            self._emit('p1', g[n], num_F, dir_, [0, pos, height],
                       box_type=box_type, box_size=action_box_size)
            # 扫描组内每个箱位间隙，将 num 拆段、gaps 填入缝隙 mm
            seg_counts, seg_gaps = [], []
            seg_start = 0
            for k in range(1, num_boxes):
                idx = base + k
                if idx < len(s) and s[idx] != 0:
                    seg_counts.append(k - seg_start)
                    seg_gaps.append(round(gap_ * s[idx] * 0.01))
                    seg_start = k
            if seg_counts:
                seg_counts.append(num_boxes - seg_start)
                self.robot_offsets[-1]['num'] = seg_counts
                self.robot_offsets[-1]['gaps'] = seg_gaps

        # Group/Stack 当前行恰好两抓时，记录执行顺序中的最后一抓。
        # _finalize 会在原左右位置码上加10，供 cmd_get_box 通知机器人
        # 这一抓完成当前行。简单行（尾料、Isdoor）不经过本函数，不受影响。
        if n_groups == 2:
            self.boxes[-1]['is_two_grab_row_last'] = True

    def _emit_p1_simple(self, N, grab_num, height, num_F, y_start=0, tail=False):
        """顺序放置 N 个箱子（无间隙分组），用于尾料/梯形门口区。
        tail=True 时使用前置尾料分批规则（余数首批、末抓满抓、禁止单箱）。"""
        cum = 0
        grabs = self._split_grabs_tail(N, grab_num) if tail else self._split_grabs(N, grab_num, min2=True)
        for num in grabs:
            self._emit('p1', num, num_F, 1, [0, y_start + cum * self.w, height], is_tail=tail)
            cum += num

    def _emit_p3_row(self, N3, z, num_F):
        """发出 P3 区一行的所有抓取动作，y 从 0 起按箱高累积。
        P3 不扣 RW、不留左边距（直接贴左壁）。"""
        y_start = 0
        cum = 0
        for num in self._split_grabs(N3, self.grab_num_p3):
            self._emit('p3', num, num_F, 1, [0, y_start + cum * self.h, z])
            cum += num

    # ── 入口与分派 ─────────────────────────────────────────────

    def read_robot_offset(self):
        """按 block 类型分派到对应解析器，生成 robot_offsets / ori_offsets / boxes。"""
        if self.block_type == 'regular':
            self._parse_regular()
        elif self.block_type == 'trapezoid':
            self._parse_trapezoid()
        elif self.block_type == 'mixture':
            self._parse_mixture()
        else:
            raise ValueError("未知 block 类型：regular/trapezoid/mixture 字段均为空")
        self._finalize()

    def _parse_regular(self):
        """常规垛：可含多条 regular（同箱型，连续切块），逐条处理，num_F 跨条连续编号。
        每条 F13 个面，每面 T12 层 P1 + P2 侧壁 + P3 侧立，可选 E==1/2/3 尾料。
        brick 奇偶（相邻面错位码放）按全局常规面号连续计，不随条重置（连续垛被切块）。"""
        num_F_base = 0    # 已用面号偏移（含 E1 尾料面，用于 num_F 连续编号）
        reg_face_base = 0  # 已用常规面数（不含尾料面，用于 brick 奇偶连续）
        for reg in self.regular:
            N1, N2, N3 = reg['N1'], reg['N2'], reg['N3']
            T12, T3, F13, F2 = reg['T12'], reg['T3'], reg['F13'], reg['F2']
            E, Nx, Stack, Group = reg['E'], reg['Nx'], reg['Stack'], reg['Group']
            p2_filled, grab_p2, p2_f = False, 0, 0
            E_ = Nx
            last_face_width = float(self.W)
            for f in range(F13):
                num_F_reg = num_F_base + f + 1
                face_width = self.face_car_width(num_F_reg)
                last_face_width = face_width
                # P1 可分配的Y向缝隙：当前面车宽-P2占用-P1箱体总宽。
                gap = face_width - N2 * self.l - N1 * self.w
                self._face_p1_right_walls[num_F_reg] = (
                    face_width - N2 * self.l)
                for t in range(T12):
                    # brick 奇偶用全局常规面号 reg_face_base+f，保证跨条连续
                    pattern_index = (reg_face_base + f + t) % 2
                    g, s = self._layer_pattern(reg_face_base + f, t, Stack, Group)
                    # P2：侧壁区，按面推进条件触发
                    if p2_f < F2 and (f + 1) * self.l > p2_f * self.w:
                        p2_filled = True
                        grab_p2 = self.grab_num_p2 if (p2_f + self.grab_num_p2) < F2 else F2 - p2_f
                        for i in range(N2):
                            self._emit(
                                'p2', grab_p2, num_F_reg, 2,
                                [0, face_width - (i + 1) * self.l,
                                 t * self.h])
                    # P1：主区，按 Stack/Group 分组放置
                    self._emit_p1_groups(
                        g, s, gap, t * self.h, num_F_reg, N1,
                        pattern_index=pattern_index)
                    if p2_filled:
                        p2_f += grab_p2
                        p2_filled = False
                # P3：侧立区，T12 层全部完成后统一发出（每面 T3 行）
                if N3 != 0:
                    for t3 in range(T3):
                        self._emit_p3_row(N3, T12 * self.h + t3 * self.w, num_F_reg)
                # 顶置尾料 z：在常规 P3 行之上（P3 行占 T3*w 高，N3==0 时无行）
                _z_tail = T12 * self.h + (T3 if N3 != 0 else 0) * self.w
                # E==2 顶置尾料（每面完成后追加到 P3 顶层，箱子侧立，仅1xx允许）
                if E == 2 and E_ != 0:
                    N3_ = int(face_width // self.h) if N3 == 0 else N3
                    Nx_ = min(N3_, E_)
                    self._emit_p3_row(Nx_, _z_tail, num_F_reg)
                    E_ -= Nx_
                # E==3 顶置尾料（每面完成后追加到 P1 顶层，箱子竖放，仅2xx/3xx）
                elif E == 3 and E_ != 0:
                    N1_ = int((face_width - self.RW) // self.w)
                    Nx_ = min(N1_, E_)
                    self._emit_p1_simple(Nx_, self.grab_num_p1, _z_tail, num_F_reg,
                                         tail=True)
                    E_ -= Nx_
                self.robot_offsets[-1]['action'] = 1
                self.boxes[-1]['action'] = 1

            if E == 2 and E_ != 0:
                N3_cap = int(last_face_width // self.h) if N3 == 0 else N3
                _logger.warning(
                    f"E==2 尾料未完全分配，剩余 {E_} 个（Nx={Nx}, F13={F13}, N3_={N3_cap}）")
            if E == 3 and E_ != 0:
                _logger.warning(
                    f"E==3 尾料未完全分配，剩余 {E_} 个（Nx={Nx}, F13={F13}, "
                    f"N1_={int((last_face_width - self.RW) // self.w)}）")

            faces_used = F13
            # E==1 前置尾料（本条额外占 1 面）
            if Nx != 0 and E == 1:
                num_F_tail = num_F_base + F13 + 1
                tail_width = self.face_car_width(num_F_tail)
                self._face_p1_right_walls[num_F_tail] = tail_width
                N1_tail = int((tail_width - self.RW) // self.w)
                if N1_tail <= 0:
                    raise ValueError(
                        f"常规尾料面可用宽度不足：面{num_F_tail}，"
                        f"车宽={tail_width:.1f}, RW={self.RW}, 箱宽={self.w}")
                faces_used += 1
                Nx_t, Nx_rem = divmod(Nx, N1_tail)
                for t in range(Nx_t):
                    self._emit_p1_simple(N1_tail, self.grab_num_p1, t * self.h, num_F_tail,
                                         tail=True)
                if Nx_rem:
                    self._emit_p1_simple(Nx_rem, self.grab_num_p1, Nx_t * self.h, num_F_tail,
                                         tail=True)
                # 尾料面末抓标换面（多条时切到下一条；_finalize 会把全局最后一抓覆盖为 2）
                self.robot_offsets[-1]['action'] = 1
                self.boxes[-1]['action'] = 1

            num_F_base += faces_used
            reg_face_base += F13   # brick 奇偶只按常规面累计（不含尾料面）

    def _parse_trapezoid(self):
        """梯形垛：逐面处理，每面 P1 + P3 + 尾料。门口区走简单顺序放置。"""
        for f, trap in enumerate(self.trapezoid):
            num_F_trap = f + 1
            face_width = self.face_car_width(num_F_trap)
            self._face_p1_right_walls[num_F_trap] = face_width
            # 梯形 P1
            for t in range(trap['T1']):
                if trap['Isdoor']:
                    # 门口区：简单顺序放置，无间隙分组；用 tail 分批规则确保顺序为升序（小→大）
                    self._emit_p1_simple(trap['N1'], self.grab_num_p1, t * self.h, num_F_trap, tail=True)
                else:
                    trap_gap = (face_width - trap['N1'] * self.w -
                                (self.l - self.w) * (trap['Group'][0][0] // 10 + trap['Group'][0][1] // 10))
                    pattern_index = (f + t) % 2
                    g, s = self._layer_pattern(f, t, trap['Stack'], trap['Group'])
                    self._emit_p1_groups(
                        g, s, trap_gap, t * self.h, num_F_trap, trap['N1'],
                        pattern_index=pattern_index)
            # 梯形 P3
            for t3 in range(trap['T3']):
                self._emit_p3_row(trap['N3'], trap['T1'] * self.h + t3 * self.w, num_F_trap)
            # 梯形尾料：常规箱/细支箱/中支箱统一P1顶置（竖放），
            # 仅放一层，超出当前P1单层容量的部分丢弃。
            if trap['Nx'] != 0:
                z_tail = trap['T1'] * self.h + trap['T3'] * self.w
                N_cap = int((face_width - self.RW) // self.w)
                Nx_ = min(N_cap, trap['Nx'])
                if trap['Nx'] > N_cap:
                    _logger.warning(
                        f"梯形P1尾料超出单行容量，Nx={trap['Nx']} > N1_={N_cap}"
                        f"（面序={f}），超出 {trap['Nx'] - N_cap} 个丢弃")
                self._emit_p1_simple(
                    Nx_, self.grab_num_p1, z_tail, num_F_trap, tail=True)
            self.robot_offsets[-1]['action'] = 1
            self.boxes[-1]['action'] = 1

    def _parse_mixture(self):
        """按混装 Items 生成动作；1XX 的 Num>=10 解码为 P3 放置。"""
        for face_idx, mix in enumerate(self.mixture):
            num_F = face_idx + 1
            face_width = self.face_car_width(num_F)
            self._face_p1_right_walls[num_F] = face_width
            emitted_before = len(self.robot_offsets)

            items = _mixture_items(mix)
            for item_idx, item in enumerate(items):
                box_type = item.get('Type', item.get('type', ''))
                encoded_num = item.get('Num', item.get('num', 0))
                pos = item.get('Pos', item.get('pos', {})) or {}
                try:
                    encoded_num = int(encoded_num)
                    pos_x = float(pos.get('X', pos.get('x', 0.0)))
                    pos_y = float(pos.get('Y', pos.get('y', 0.0)))
                    pos_z = float(pos.get('Z', pos.get('z', 0.0)))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Mixture 面{num_F}第{item_idx + 1}项数值格式错误") from exc

                if not box_type:
                    raise ValueError(f"Mixture 面{num_F}第{item_idx + 1}项缺少箱型 Type")
                if encoded_num <= 0:
                    raise ValueError(
                        f"Mixture 面{num_F}第{item_idx + 1}项 Num 必须大于 0，"
                        f"当前为 {encoded_num}")
                area, actual_num = _decode_mixture_placement(
                    box_type, encoded_num)
                if actual_num <= 0:
                    raise ValueError(
                        f"Mixture 面{num_F}第{item_idx + 1}项 Num={encoded_num} "
                        f"解码后的真实数量必须大于 0")
                if not all(math.isfinite(v) for v in (pos_x, pos_y, pos_z)):
                    raise ValueError(
                        f"Mixture 面{num_F}第{item_idx + 1}项 Pos 含非有限数值")
                if min(pos_x, pos_y, pos_z) < 0:
                    raise ValueError(
                        f"Mixture 面{num_F}第{item_idx + 1}项 Pos 不能为负数")

                params = self._box_params(box_type)
                box_size = params['size']
                box_l, box_w, box_h = box_size
                grab_limit = (
                    params['grab_p3'] if area == 'p3'
                    else params['grab_p1'])
                if actual_num > grab_limit:
                    raise ValueError(
                        f"Mixture 面{num_F}第{item_idx + 1}项 "
                        f"Num={encoded_num}(真实数量={actual_num}) 超出"
                        f"箱型 {box_type} 的 {area.upper()} 单抓能力 {grab_limit}")
                y_span = box_h if area == 'p3' else box_w
                z_span = box_w if area == 'p3' else box_h
                if pos_y + actual_num * y_span > face_width:
                    raise ValueError(
                        f"Mixture 面{num_F}第{item_idx + 1}项沿车宽超界："
                        f"Y({pos_y}) + 真实数量({actual_num})×"
                        f"{area.upper()}宽度({y_span}) > {face_width:.1f}")
                if pos_x + box_l > self.L:
                    raise ValueError(
                        f"Mixture 面{num_F}第{item_idx + 1}项沿车深超界："
                        f"X({pos_x}) + 箱长({box_l}) > {self.L}")
                if pos_z + z_span > self.H:
                    raise ValueError(
                        f"Mixture 面{num_F}第{item_idx + 1}项沿车高超界："
                        f"Z({pos_z}) + {area.upper()}高度({z_span}) > {self.H}")

                # 接口与内部坐标一致：X=车深、Y=车宽、Z=车高。
                internal_pos = [pos_x, pos_y, pos_z]
                center_y = pos_y + actual_num * y_span * 0.5
                # 这里只生成失败回退用的初始方向；cmd_get_path 会结合当时已经
                # 码放的空间邻箱和左右间隙，动态修正混装 P1 抓的实际方向。
                dir_ = 1 if center_y <= face_width * 0.5 else 2
                self._emit(
                    area, actual_num, num_F, dir_, internal_pos,
                    box_type=box_type, box_size=box_size,
                    box_num_signal=(
                        encoded_num if area == 'p3' else None))

            if len(self.robot_offsets) == emitted_before:
                raise ValueError(f"Mixture 面{num_F}的 Items 为空")
            self.robot_offsets[-1]['action'] = 1
            self.boxes[-1]['action'] = 1

    def _mixture_area_cfg_map(self):
        """按混装面的已码箱体空间关系生成 area_cfg。

        area_cfg=4 只表示“当前高度的墙边收尾抓”：当前抓在该高度的
        有效箱体中最靠近一侧车壁，并且其内侧已有箱体的顶面超过当前
        放置底面50mm。其他情况均返回1，不再使用area_cfg=5。
        """
        result = {}
        placed_by_face = defaultdict(list)
        eps = 1e-6
        min_top_above = 50.0
        all_by_face = defaultdict(list)
        for action in self.robot_offsets:
            if action != 'done' and action['area'] == 'p1':
                all_by_face[action['num_F']].append(action)

        for current in self.robot_offsets:
            if current == 'done':
                continue
            if current['area'] != 'p1':
                result[current['id']] = 1
                continue

            cur_x0 = float(current['pos'][0])
            cur_x1 = cur_x0 + float(current['size'][0])
            cur_y0 = float(current['pos'][1])
            cur_y1 = (
                cur_y0 + sum(current['num']) * float(current['size'][1]))
            cur_z = float(current['pos'][2])

            # 在当前放置底面高度上仍有实体截面的箱体，用来确定当前抓是否
            # 位于该高度最靠左壁或最靠右壁的位置。未来更高层箱体不会参与。
            active_at_height = []
            for action in all_by_face[current['num_F']]:
                action_z0 = float(action['pos'][2])
                action_top = action_z0 + float(action['size'][2])
                if (action_z0 > cur_z + eps or
                        action_top <= cur_z + min_top_above + eps):
                    continue
                action_x0 = float(action['pos'][0])
                action_x1 = action_x0 + float(action['size'][0])
                if min(cur_x1, action_x1) - max(cur_x0, action_x0) <= eps:
                    continue
                active_at_height.append(action)

            leftmost_y = min(
                float(action['pos'][1]) for action in active_at_height)
            rightmost_y = max(
                float(action['pos'][1]) +
                sum(action['num']) * float(action['size'][1])
                for action in active_at_height)
            nearest_left_wall = cur_y0 <= leftmost_y + eps
            nearest_right_wall = cur_y1 >= rightmost_y - eps

            left_has_box = False
            right_has_box = False

            for other in placed_by_face[current['num_F']]:
                other_top = (
                    float(other['pos'][2]) + float(other['size'][2]))
                if other_top <= cur_z + min_top_above + eps:
                    continue

                other_x0 = float(other['pos'][0])
                other_x1 = other_x0 + float(other['size'][0])
                if min(cur_x1, other_x1) - max(cur_x0, other_x0) <= eps:
                    continue

                other_y0 = float(other['pos'][1])
                other_y1 = (
                    other_y0 + sum(other['num']) *
                    float(other['size'][1]))
                if other_y1 <= cur_y0 + eps:
                    left_has_box = True
                elif other_y0 >= cur_y1 - eps:
                    right_has_box = True

            if ((nearest_left_wall and right_has_box) or
                    (nearest_right_wall and left_has_box)):
                result[current['id']] = 4
            else:
                result[current['id']] = 1

            placed_by_face[current['num_F']].append(current)

        return result

    def _finalize(self):
        """末尾处理：标记 block 结束、填充 ori_offsets 快照、计算 area_cfg 位置编号。"""
        self.robot_offsets[-1]['action'] = 2
        self.boxes[-1]['action'] = 2
        self.robot_offsets.append('done')

        # 常规/梯形 area_cfg：按同面、同区域、同高度的 y 顺序确定左/中/右；
        # 异形车头 P1 三抓行的最右抓固定为1。Group/Stack 两抓行的执行
        # 末抓在原位置码上加10（11或13）。
        # 混装面使用三维邻箱关系：4=当前高度的墙边收尾抓，其余为1。
        groups = defaultdict(list)
        for offset in self.robot_offsets:
            if offset == 'done':
                continue
            groups[(offset['num_F'], offset['area'], offset['pos'][2])].append(offset)
        id_to_cfg = {}
        for offsets in groups.values():
            ordered = sorted(offsets, key=lambda o: o['pos'][1])
            n = len(ordered)
            for rank, offset in enumerate(ordered):
                oid = offset['id']
                if n == 1 or rank == 0:
                    id_to_cfg[oid] = 1
                elif (n == 3 and rank == n - 1
                      and offset['area'] == 'p1'
                      and bool(offset.get('is_head', False))):
                    # 异形车头三抓的最右抓不使用常规右侧位置码3。
                    id_to_cfg[oid] = 1
                elif rank == n - 1:
                    id_to_cfg[oid] = 3
                else:
                    id_to_cfg[oid] = 2
        mixture_cfg = (
            self._mixture_area_cfg_map()
            if self.block_type == 'mixture' else {})
        for box in self.boxes:
            if self.block_type == 'mixture':
                box['area_cfg'] = mixture_cfg.get(box['id'], 1)
            elif box.get('is_tail') or box['area'] == 'p3':
                box['area_cfg'] = 1
            else:
                box['area_cfg'] = id_to_cfg.get(box['id'], 1)
            if box.get('is_two_grab_row_last'):
                box['area_cfg'] += 10

        # 每面单层抓数映射：num_F → 该面 P1 底层(最低z)的抓取次数
        # 直接从生成的 offsets 数，对多条 regular / 梯形 / 尾料面均精确，不依赖单一 Group
        p1_by_face = defaultdict(list)
        for od in self.robot_offsets:
            if od != 'done' and od['area'] == 'p1':
                p1_by_face[od['num_F']].append(od)
        self.n_per_row_map = {}
        for nf, offs in p1_by_face.items():
            zmin = min(o['pos'][2] for o in offs)
            self.n_per_row_map[nf] = sum(1 for o in offs if o['pos'][2] == zmin)

        self.ori_offsets = self.robot_offsets.copy()

    def cal_floor_count(self):
        """返回当前来料所属面的总抓数；无待处理来料时返回1。"""
        if len(self.boxes) == 0:
            return 1
        x = self.boxes[0]['num_F']
        count = 0
        for od in self.ori_offsets:
            if od != 'done':
                if od['num_F'] == x:
                    count += 1
        return count

    def cal_n_per_row(self):
        """当前面（boxes[0] 所属）P1 底层单层抓数；无 boxes 或该面无 P1 时返回 1。"""
        if not self.boxes:
            return 1
        return self.n_per_row_map.get(self.boxes[0]['num_F'], 1)


class BinEnv:
    """维护当前码垛面的箱体 AABB，并提供路径与支撑检查。"""

    def __init__(self, config_data):
        """读取安全余量和混装诊断参数，初始化空环境。"""
        self.reserve_grip = config_data['reserve_grip']
        self.reserve_object = config_data['reserve_object']
        self.mixture_z_overlap_min = float(
            config_data.get('mixture_z_overlap_min', 20.0))
        self.mixture_support_z_tolerance = float(
            config_data.get('mixture_support_z_tolerance', 20.0))
        self.mixture_support_min_ratio = float(
            config_data.get('mixture_support_min_ratio', 0.80))
        self.mixture_support_min_box_ratio = float(
            config_data.get('mixture_support_min_box_ratio', 0.60))
        # 面积阈值随支撑结果输出，当前只作诊断；风险由重心稳定性判定。
        self.objects = []          # 含 reserve_object 外扩的 AABB，用于安全段碰撞检测
        self.display_objects = []  # 真实尺寸 AABB，用于落箱段、支撑分析和可视化

    def reset(self):
        """清空当前面的安全 AABB 与真实 AABB。"""
        self.objects = []
        self.display_objects = []

    @staticmethod
    def _aabb_intersects(a, b, tol=0.5):
        """判断两个三维 AABB 是否存在实质重叠，边界接触不算碰撞。"""
        return all(a[i] < b[i + 3] - tol and a[i + 3] > b[i] + tol
                   for i in range(3))

    def side_clearance(self, action, left_wall, right_wall,
                       min_z_overlap=None):
        """按真实已码箱体计算P1/P3抓取目标左右两侧的可用空间。

        与规则垛的“底部 Z 完全相等”不同，这里使用 X/Z 包围盒的有效重叠
        判断某个已码箱体是否会影响当前抓的横向进入。返回的 blocking 表示
        目标位置本身已经与已有箱体重叠，由调用方决定回退或报警。
        """
        if action['area'] not in ('p1', 'p3'):
            raise ValueError("侧向空间分析仅支持 P1/P3 区域")
        if min_z_overlap is None:
            min_z_overlap = self.mixture_z_overlap_min

        current_boxes = self.to_box(action)
        if not current_boxes:
            raise ValueError("当前动作没有可用箱体")

        cur_x0 = min(box.position[0] for box in current_boxes)
        cur_y0 = min(box.position[1] for box in current_boxes)
        cur_z0 = min(box.position[2] for box in current_boxes)
        cur_x1 = max(box.position[0] + box.length for box in current_boxes)
        cur_y1 = max(box.position[1] + box.width for box in current_boxes)
        cur_z1 = max(box.position[2] + box.height for box in current_boxes)

        left_edges = []
        right_starts = []
        blocking = []
        relevant = []
        for obstacle in self.display_objects:
            obs_x0, obs_y0, obs_z0, obs_x1, obs_y1, obs_z1 = obstacle
            x_overlap = min(cur_x1, obs_x1) - max(cur_x0, obs_x0)
            z_overlap = min(cur_z1, obs_z1) - max(cur_z0, obs_z0)
            if x_overlap <= 0 or z_overlap < min_z_overlap:
                continue

            relevant.append(obstacle)
            if obs_y1 <= cur_y0:
                left_edges.append(obs_y1)
            elif obs_y0 >= cur_y1:
                right_starts.append(obs_y0)
            else:
                blocking.append(obstacle)

        left_edge = max(left_edges) if left_edges else float(left_wall)
        right_start = min(right_starts) if right_starts else float(right_wall)
        return {
            'current_aabb': (cur_x0, cur_y0, cur_z0, cur_x1, cur_y1, cur_z1),
            'left_edge': left_edge,
            'right_start': right_start,
            'left_gap': cur_y0 - left_edge,
            'right_gap': right_start - cur_y1,
            'left_is_box': bool(left_edges),
            'right_is_box': bool(right_starts),
            'blocking': blocking,
            'relevant_count': len(relevant),
        }

    def mixture_side_clearance(self, action, left_wall, right_wall,
                               min_z_overlap=None):
        """兼容混装动态APP调用，仅允许P1，内部复用通用侧向空间分析。"""
        if action['area'] != 'p1':
            raise ValueError("混装动态 APP 当前仅支持 P1 区域")
        return self.side_clearance(
            action,
            left_wall=left_wall,
            right_wall=right_wall,
            min_z_overlap=min_z_overlap,
        )

    @staticmethod
    def mixture_gripper_wall_clearance(action, left_wall, right_wall):
        """检查混装 P1 固定长度手抓在目标位置是否越过车厢侧壁。

        P1 手抓按箱型的最大单抓数确定固定宽度，并与本抓箱体左边缘对齐；
        尾抓箱数少于最大单抓数时，空余手抓长度仍向 +Y 方向伸出。
        """
        if action['area'] != 'p1':
            raise ValueError("混装固定手抓车壁检查当前仅支持 P1 区域")

        actual_num = int(sum(action['num']))
        if actual_num <= 0:
            raise ValueError("当前混装动作的实际箱数必须大于0")
        grip_capacity = int(action.get('grab_num_p1', actual_num))
        if grip_capacity <= 0:
            raise ValueError("当前箱型的P1最大单抓数必须大于0")
        grip_capacity = max(grip_capacity, actual_num)

        box_width = float(action['size'][1])
        if box_width <= 0:
            raise ValueError("当前混装动作的P1箱宽必须大于0")

        box_left = float(action['pos'][1])
        box_right = box_left + actual_num * box_width
        gripper_left = box_left
        gripper_right = gripper_left + grip_capacity * box_width
        left_wall = float(left_wall)
        right_wall = float(right_wall)
        left_overhang = max(0.0, left_wall - gripper_left)
        right_overhang = max(0.0, gripper_right - right_wall)
        return {
            'actual_num': actual_num,
            'grip_capacity': grip_capacity,
            'box_width': box_width,
            'box_left': box_left,
            'box_right': box_right,
            'gripper_left': gripper_left,
            'gripper_right': gripper_right,
            'gripper_width': grip_capacity * box_width,
            'left_wall': left_wall,
            'right_wall': right_wall,
            'left_overhang': left_overhang,
            'right_overhang': right_overhang,
            'collision': left_overhang > 0.0 or right_overhang > 0.0,
        }

    @staticmethod
    def _rectangle_union_area(rectangles):
        """计算若干轴对齐二维矩形的并集面积。"""
        rectangles = [
            tuple(map(float, rect)) for rect in rectangles
            if rect[2] > rect[0] and rect[3] > rect[1]
        ]
        if not rectangles:
            return 0.0
        x_edges = sorted({value for rect in rectangles
                          for value in (rect[0], rect[2])})
        area = 0.0
        for x0, x1 in zip(x_edges, x_edges[1:]):
            if x1 <= x0:
                continue
            intervals = sorted(
                (rect[1], rect[3]) for rect in rectangles
                if rect[0] < x1 and rect[2] > x0)
            if not intervals:
                continue
            covered_y = 0.0
            start, end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start <= end:
                    end = max(end, next_end)
                else:
                    covered_y += end - start
                    start, end = next_start, next_end
            covered_y += end - start
            area += (x1 - x0) * covered_y
        return area

    @staticmethod
    def _convex_hull(points):
        """返回二维点集的凸包顶点（逆时针，使用单调链算法）。"""
        points = sorted({
            (float(point[0]), float(point[1])) for point in points
        })
        if len(points) <= 1:
            return points

        def _cross(origin, first, second):
            """返回二维三点转向的有符号叉积。"""
            return (
                (first[0] - origin[0]) * (second[1] - origin[1]) -
                (first[1] - origin[1]) * (second[0] - origin[0]))

        lower = []
        for point in points:
            while (len(lower) >= 2 and
                   _cross(lower[-2], lower[-1], point) <= 0):
                lower.pop()
            lower.append(point)
        upper = []
        for point in reversed(points):
            while (len(upper) >= 2 and
                   _cross(upper[-2], upper[-1], point) <= 0):
                upper.pop()
            upper.append(point)
        return lower[:-1] + upper[:-1]

    @staticmethod
    def _point_in_convex_polygon(point, polygon, tolerance=1e-6):
        """判断点是否位于凸多边形内部或边界上。"""
        if not polygon:
            return False
        px, py = map(float, point)
        if len(polygon) == 1:
            return (
                abs(px - polygon[0][0]) <= tolerance and
                abs(py - polygon[0][1]) <= tolerance)
        if len(polygon) == 2:
            first, second = polygon
            cross = (
                (second[0] - first[0]) * (py - first[1]) -
                (second[1] - first[1]) * (px - first[0]))
            if abs(cross) > tolerance:
                return False
            return (
                min(first[0], second[0]) - tolerance <= px <=
                max(first[0], second[0]) + tolerance and
                min(first[1], second[1]) - tolerance <= py <=
                max(first[1], second[1]) + tolerance)

        cross_sign = 0
        for index, first in enumerate(polygon):
            second = polygon[(index + 1) % len(polygon)]
            cross = (
                (second[0] - first[0]) * (py - first[1]) -
                (second[1] - first[1]) * (px - first[0]))
            if abs(cross) <= tolerance:
                continue
            current_sign = 1 if cross > 0 else -1
            if cross_sign and current_sign != cross_sign:
                return False
            cross_sign = current_sign
        return True

    @classmethod
    def _support_hull_contains(cls, point, rectangles):
        """判断重心投影是否落在全部接触面的联合支撑凸包内。"""
        corners = [
            corner
            for x0, y0, x1, y1 in rectangles
            for corner in (
                (x0, y0), (x0, y1), (x1, y0), (x1, y1))
        ]
        return cls._point_in_convex_polygon(
            point, cls._convex_hull(corners))

    def analyze_mixture_support(self, action):
        """分析混装P1当前抓底面受到已码箱体支撑的面积比例。

        只使用 display_objects 中已经实际码放的箱体。支撑面高度与当前
        放置底面的差值不超过配置容差时视为接触；地面层直接按完全支撑。
        稳定性按重心投影是否落在所有接触面的联合凸包内判断：上层箱跨在
        两个下层箱上时，即使每个接触面较小，只要两侧能共同托住重心就稳定。
        面积比例保留为诊断信息，不再单独触发“支撑面积不足”。
        返回值仅供检查、日志和可视化使用，不参与路径放行。
        """
        if action['area'] != 'p1':
            raise ValueError("混装物理支撑分析当前仅支持P1区域")

        current_boxes = self.to_box(action)
        if not current_boxes:
            raise ValueError("当前混装动作没有可分析箱体")

        z_tolerance = max(0.0, self.mixture_support_z_tolerance)
        min_total_ratio = min(max(self.mixture_support_min_ratio, 0.0), 1.0)
        min_box_ratio = min(
            max(self.mixture_support_min_box_ratio, 0.0), 1.0)
        current_z = min(float(box.position[2]) for box in current_boxes)
        on_floor = current_z <= z_tolerance
        per_box = []
        total_area = 0.0
        total_supported_area = 0.0
        supporter_indices = set()

        for box_index, box in enumerate(current_boxes, start=1):
            x0 = float(box.position[0])
            y0 = float(box.position[1])
            x1 = x0 + float(box.length)
            y1 = y0 + float(box.width)
            footprint_area = max(0.0, (x1 - x0) * (y1 - y0))
            total_area += footprint_area
            support_rectangles = []
            box_supporter_indices = set()

            if on_floor:
                support_rectangles.append((x0, y0, x1, y1))
            else:
                for obstacle_index, obstacle in enumerate(
                        self.display_objects):
                    obs_x0, obs_y0, _, obs_x1, obs_y1, obs_top = obstacle
                    if abs(float(obs_top) - current_z) > z_tolerance:
                        continue
                    overlap = (
                        max(x0, float(obs_x0)),
                        max(y0, float(obs_y0)),
                        min(x1, float(obs_x1)),
                        min(y1, float(obs_y1)),
                    )
                    if overlap[2] <= overlap[0] or overlap[3] <= overlap[1]:
                        continue
                    support_rectangles.append(overlap)
                    supporter_indices.add(obstacle_index)
                    box_supporter_indices.add(obstacle_index)

            supported_area = min(
                footprint_area,
                self._rectangle_union_area(support_rectangles))
            support_ratio = (
                supported_area / footprint_area if footprint_area > 0 else 0.0)
            center_x = (x0 + x1) * 0.5
            center_y = (y0 + y1) * 0.5
            center_supported = any(
                rect[0] <= center_x <= rect[2] and
                rect[1] <= center_y <= rect[3]
                for rect in support_rectangles)
            hull_supported = self._support_hull_contains(
                (center_x, center_y), support_rectangles)
            if on_floor:
                support_mode = 'floor'
            elif center_supported:
                support_mode = 'direct'
            elif hull_supported and len(box_supporter_indices) >= 2:
                support_mode = 'bridge'
            elif supported_area <= 0:
                support_mode = 'floating'
            else:
                support_mode = 'unbalanced'
            stable = (
                on_floor or center_supported or
                (hull_supported and len(box_supporter_indices) >= 2))
            total_supported_area += supported_area
            per_box.append({
                'index': box_index,
                'footprint': [x0, y0, x1, y1],
                'support_ratio': round(support_ratio, 6),
                'center_supported': bool(center_supported),
                'hull_supported': bool(hull_supported),
                'stable': bool(stable),
                'support_mode': support_mode,
                'supporter_count': len(box_supporter_indices),
            })

        total_ratio = (
            total_supported_area / total_area if total_area > 0 else 0.0)
        min_observed_box_ratio = min(
            item['support_ratio'] for item in per_box)
        risk_box_indices = [
            item['index'] for item in per_box
            if not item['stable']
        ]
        has_risk = (
            not on_floor and
            bool(risk_box_indices))
        if total_ratio <= 0.01 and not on_floor:
            risk_level = 'floating'
        elif has_risk:
            risk_level = 'weak_support'
        else:
            risk_level = 'ok'

        return {
            'risk': bool(has_risk),
            'risk_level': risk_level,
            'on_floor': bool(on_floor),
            'support_ratio': round(total_ratio, 6),
            'unsupported_ratio': round(max(0.0, 1.0 - total_ratio), 6),
            'min_box_support_ratio': round(min_observed_box_ratio, 6),
            'risk_box_indices': risk_box_indices,
            'supporter_count': len(supporter_indices),
            'z_tolerance_mm': z_tolerance,
            'min_support_ratio': min_total_ratio,
            'required_min_box_support_ratio': min_box_ratio,
            'stability_rule': 'center_of_mass_in_support_hull',
            'area_thresholds_diagnostic_only': True,
            'per_box': per_box,
        }

    def trajectory_collision_free(self, path, size, sample_step=10.0):
        """连续采样检查整抓包围盒沿路径是否碰撞。

        x0→x1、x1→APP 使用带 reserve_object 的安全障碍物；APP→goal
        使用真实箱体，允许最终正常贴箱和落在支撑面上。返回 (是否安全, 详情)。
        """
        if len(path) < 2:
            return True, None
        if sample_step <= 0:
            raise ValueError("轨迹采样步长必须大于 0")

        for seg_idx, (start, end) in enumerate(zip(path, path[1:])):
            # 最后一段是精确落箱过程，不能使用外扩障碍，否则正常贴箱也会误报。
            obstacles = (self.display_objects
                         if seg_idx == len(path) - 2 else self.objects)
            distance = math.sqrt(sum((end[i] - start[i]) ** 2 for i in range(3)))
            sample_count = max(1, int(math.ceil(distance / sample_step)))
            for sample_idx in range(sample_count + 1):
                ratio = sample_idx / sample_count
                point = tuple(start[i] + (end[i] - start[i]) * ratio
                              for i in range(3))
                carried = (
                    point[0], point[1], point[2],
                    point[0] + size[0],
                    point[1] + size[1],
                    point[2] + size[2],
                )
                for obs_idx, obstacle in enumerate(obstacles):
                    if self._aabb_intersects(carried, obstacle):
                        return False, {
                            'segment': seg_idx,
                            'sample': sample_idx,
                            'ratio': ratio,
                            'point': point,
                            'obstacle_index': obs_idx,
                        }
        return True, None

    @staticmethod
    def to_box(action):
        """根据 action 生成 Box 列表。
        单段（num=[n]）：支持翻转编码（num>=10）；多段（num=[n1,n2,...]）：按 gaps 累积 y 偏移。"""
        boxs = []
        area = action['area']
        num_list = action['num']
        gaps = action.get('gaps', [])
        pos = action['pos']
        l = action['size'][0]
        w = action['size'][1]
        h = action['size'][2]
        if area == 'p1':
            # P1 主区：沿 y 方向排列，多段时按 gaps 累积偏移
            y = pos[1]
            for seg_i, seg_n in enumerate(num_list):
                for i in range(seg_n):
                    box = Box(l, w, h, rotation_type=0, position=[pos[0], y + i * w, pos[2]])
                    boxs.append(box)
                y += seg_n * w
                if seg_i < len(gaps):
                    y += gaps[seg_i]
        elif area == 'p2':
            # P2 侧壁区：箱子竖立，沿 x 方向排列（每抓 grab_num 个）
            num = sum(num_list)
            for i in range(num):
                box = Box(l, w, h, rotation_type=0, position=[pos[0] + i * l, pos[1], pos[2]])
                boxs.append(box)
        elif area == 'p3':
            # P3 侧立区：箱子侧立（rotation_type=2），沿 y 方向按原始高度 h 累积；
            # 经旋转后 box.width=h（y跨度），box.height=w（z跨度），位置步长 i*h 与 box.width 吻合
            num = sum(num_list)
            for i in range(num):
                box = Box(l, w, h, rotation_type=2, position=[pos[0], pos[1] + i * h, pos[2]])
                boxs.append(box)
        else:
            raise Exception('no support area !')
        return boxs

    def step(self, action):
        """更新障碍物列表：普通放置追加 box 包围盒，换面/block结束/全部结束时清空。
        objects 含 reserve_object 外扩；display_objects 保留真实尺寸供可视化使用。"""
        boxs = self.to_box(action)
        if action['action'] == 0:
            if action['dir'] != 2:
                for box in boxs:
                    self.objects.append((box.position[0], box.position[1], box.position[2],
                                         box.position[0] + box.length + self.reserve_object[0],
                                         box.position[1] + box.width + self.reserve_object[1],
                                         box.position[2] + box.height + self.reserve_object[2]))
            else:
                for box in boxs:
                    self.objects.append((box.position[0], box.position[1] - self.reserve_object[1], box.position[2],
                                         box.position[0] + box.length + self.reserve_object[0],
                                         box.position[1] + box.width,
                                         box.position[2] + box.height + self.reserve_object[2]))
            for box in boxs:
                self.display_objects.append((box.position[0], box.position[1], box.position[2],
                                             box.position[0] + box.length,
                                             box.position[1] + box.width,
                                             box.position[2] + box.height))
        elif action['action'] in (1, 2, 3):
            self.objects.clear()
            self.display_objects.clear()

    def render(self):
        """保留的环境渲染接口；当前可视化由 plotting 模块完成。"""
        pass


class Node:
    """保留给二维搜索算法使用的轻量节点结构。"""

    def __init__(self, x, y):
        """创建坐标为 ``(x, y)``、代价为0且无父节点的节点。"""
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None
