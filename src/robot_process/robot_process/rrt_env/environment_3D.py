import copy
import logging
import math
from collections import defaultdict

_logger = logging.getLogger(__name__)


def _mixture_items(mixture_face):
    """返回 Mixture 的放置信息列表。"""
    return mixture_face.get('Items', mixture_face.get('items', []))


def _mixture_box_count(mixture_face):
    return sum(item.get('Num', item.get('num', 0))
               for item in _mixture_items(mixture_face))


class Box:

    def __init__(self, l, w, h, rotation_type, position):
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
        # box
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
        # car
        self.L = round(config_data['car']['size']['L'])
        self.W = round(config_data['car']['size']['W'])
        self.H = round(config_data['car']['size']['H'])
        self.RW = round(config_data['car']['reserve']['W'])

        # block 类型与对应字段（互斥）
        self.regular   = config_data.get('regular',   []) or []
        self.trapezoid = config_data.get('trapezoid', []) or []
        self.mixture   = config_data.get('mixture',   []) or []
        self.block_type = self._detect_block_type()

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

    def _emit(self, area, num, num_F, dir_, pos, is_tail=False,
              box_type=None, box_size=None):
        """向 robot_offsets 和 boxes 同步追加一条动作记录。
        robot_offsets 中 num 存列表形式（方便机器人侧按段读取），
        boxes 中 num 存整数，并保存当前抓的箱型和有效尺寸；箱型信号（+10）：
          - 20x 箱型：任意区域 +10
          - 10x / 30x 箱型：仅 P3 区域 +10
        """
        self._id += 1
        num_int = sum(num) if isinstance(num, list) else num
        num_list = num if isinstance(num, list) else [num]
        action_box_type = self.box_type if box_type is None else box_type
        params = self._box_params(action_box_type)
        action_box_size = list(params['size'] if box_size is None else box_size)
        box_first = str(action_box_type)[0]
        if box_first in ('2', '3') or (box_first == '1' and area == 'p3'):
            num_int += 10
        self.robot_offsets.append({
            'id': self._id, 'area': area, 'num': num_list, 'gaps': [], 'num_F': num_F,
            'action': 0, 'dir': dir_, 'pos': pos, 'size': action_box_size,
            'box_type': action_box_type, 'grab_num_p1': params['grab_p1'],
        })
        self.boxes.append({
            'id': self._id, 'area': area, 'num': num_int, 'num_F': num_F,
            'action': 0, 'area_cfg': 0, 'is_tail': is_tail,
            'box_type': action_box_type, 'size': action_box_size,
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
                        box_type=None, box_size=None):
        """按 Stack/Group 分组方式发出一层 P1 的所有抓取动作。
        n1: 本条 regular/梯形的 N1（P1 每层箱数），用于算右边界（多条 regular 时各条不同）。
        放置顺序：左(0) → 右(last) → 中间(1..last-1)。
        g 以访问顺序给出每抓箱数（g[0]=第1抓，g[1]=第2抓…），
        g_phys[物理槽位] = g[访问步骤] 用于计算实际坐标。
        Stack 长度为 N1+1：s[i] 表示第i箱左侧缝隙占比（s[0]=左墙到第0箱，s[N1]=最后一箱到右墙，不参与位置计算）。
        pos = gap * sum(s[:base+1]) * 0.01 + base * w
        若 Stack 在组内部有非零比例（缝隙落在一次抓取中间），num 按缝隙位置拆成子数组，
        gaps 存每段之间的缝隙 mm，机器人侧按 num/gaps 分段放置。"""
        action_box_size = self.box_size if box_size is None else box_size
        box_w = action_box_size[1]
        n_groups = len(g)
        if n_groups <= 2:
            visit_order = list(range(n_groups))
        else:
            # 先左(0) → 右半段顺序(mid..last) → 左中间(1..mid-1)
            # mid = ceil(n/2)，例：3组→[0,2,1]，4组→[0,2,3,1]，5组→[0,3,4,1,2]
            mid = (n_groups + 1) // 2
            visit_order = [0] + list(range(mid, n_groups)) + list(range(1, mid))
        placed = set()
        for step, n in enumerate(visit_order):
            base = sum(g[:n])
            num_boxes = g[n]
            pos = gap_ * sum(s[:base + 1]) * 0.01 + base * box_w
            # dir_：按放置步骤序号决定
            #   第1抓(step=0) → dir_=1（从右往左放）
            #   第2抓(step=1)且组数>2 → dir_=2（从左往右放）
            #   其余 → 看相邻侧：左邻已放 dir_=2，右邻已放 dir_=1，
            #             两侧均放则比较物理间隙，靠近间隙小的一侧（相等取 dir_=1）
            if step == 0:
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
            # P1 可分配的 y 方向缝隙宽度（车厢宽 - P2占用 - P1箱体总宽）
            gap = self.W - N2 * self.l - N1 * self.w
            p2_filled, grab_p2, p2_f = False, 0, 0
            E_ = Nx
            for f in range(F13):
                num_F_reg = num_F_base + f + 1
                for t in range(T12):
                    # brick 奇偶用全局常规面号 reg_face_base+f，保证跨条连续
                    g, s = self._layer_pattern(reg_face_base + f, t, Stack, Group)
                    # P2：侧壁区，按面推进条件触发
                    if p2_f < F2 and (f + 1) * self.l > p2_f * self.w:
                        p2_filled = True
                        grab_p2 = self.grab_num_p2 if (p2_f + self.grab_num_p2) < F2 else F2 - p2_f
                        for i in range(N2):
                            self._emit('p2', grab_p2, num_F_reg, 2, [0, self.W - (i + 1) * self.l, t * self.h])
                    # P1：主区，按 Stack/Group 分组放置
                    self._emit_p1_groups(g, s, gap, t * self.h, num_F_reg, N1)
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
                    N3_ = int(self.W // self.h) if N3 == 0 else N3
                    Nx_ = min(N3_, E_)
                    self._emit_p3_row(Nx_, _z_tail, num_F_reg)
                    E_ -= Nx_
                # E==3 顶置尾料（每面完成后追加到 P1 顶层，箱子竖放，仅2xx/3xx）
                elif E == 3 and E_ != 0:
                    N1_ = int((self.W - self.RW) // self.w)
                    Nx_ = min(N1_, E_)
                    self._emit_p1_simple(Nx_, self.grab_num_p1, _z_tail, num_F_reg,
                                         tail=True)
                    E_ -= Nx_
                self.robot_offsets[-1]['action'] = 1
                self.boxes[-1]['action'] = 1

            if E == 2 and E_ != 0:
                N3_cap = int(self.W // self.h) if N3 == 0 else N3
                _logger.warning(
                    f"E==2 尾料未完全分配，剩余 {E_} 个（Nx={Nx}, F13={F13}, N3_={N3_cap}）")
            if E == 3 and E_ != 0:
                _logger.warning(
                    f"E==3 尾料未完全分配，剩余 {E_} 个（Nx={Nx}, F13={F13}, N1_={int((self.W - self.RW) // self.w)}）")

            faces_used = F13
            # E==1 前置尾料（本条额外占 1 面）
            if Nx != 0 and E == 1:
                N1_tail = int((self.W - self.RW) // self.w)
                num_F_tail = num_F_base + F13 + 1
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
            # 梯形 P1
            for t in range(trap['T1']):
                if trap['Isdoor']:
                    # 门口区：简单顺序放置，无间隙分组；用 tail 分批规则确保顺序为升序（小→大）
                    self._emit_p1_simple(trap['N1'], self.grab_num_p1, t * self.h, num_F_trap, tail=True)
                else:
                    trap_gap = (self.W - trap['N1'] * self.w -
                                (self.l - self.w) * (trap['Group'][0][0] // 10 + trap['Group'][0][1] // 10))
                    g, s = self._layer_pattern(f, t, trap['Stack'], trap['Group'])
                    self._emit_p1_groups(g, s, trap_gap, t * self.h, num_F_trap, trap['N1'])
            # 梯形 P3
            for t3 in range(trap['T3']):
                self._emit_p3_row(trap['N3'], trap['T1'] * self.h + t3 * self.w, num_F_trap)
            # 梯形尾料：10x→P3顶置（侧立），20x/30x→P1顶置（竖放），仅一层，超出丢弃
            if trap['Nx'] != 0:
                z_tail = trap['T1'] * self.h + trap['T3'] * self.w
                box_first = str(self.box_type)[0]
                if box_first == '1':
                    # P3顶置：规则同 regular E==2
                    N_cap = trap['N3'] if trap['N3'] != 0 else int(self.W // self.h)
                    Nx_ = min(N_cap, trap['Nx'])
                    if trap['Nx'] > N_cap:
                        _logger.warning(
                            f"梯形P3尾料超出单行容量，Nx={trap['Nx']} > N3_={N_cap}"
                            f"（面序={f}），超出 {trap['Nx'] - N_cap} 个丢弃")
                    self._emit_p3_row(Nx_, z_tail, num_F_trap)
                else:
                    # P1顶置：规则同 regular E==3
                    N_cap = int((self.W - self.RW) // self.w)
                    Nx_ = min(N_cap, trap['Nx'])
                    if trap['Nx'] > N_cap:
                        _logger.warning(
                            f"梯形P1尾料超出单行容量，Nx={trap['Nx']} > N1_={N_cap}"
                            f"（面序={f}），超出 {trap['Nx'] - N_cap} 个丢弃")
                    self._emit_p1_simple(Nx_, self.grab_num_p1, z_tail, num_F_trap,
                                         tail=True)
            self.robot_offsets[-1]['action'] = 1
            self.boxes[-1]['action'] = 1

    def _parse_mixture(self):
        """混装面：按 Items 中给出的箱型、数量和 Pos 直接生成放置动作。"""
        for face_idx, mix in enumerate(self.mixture):
            num_F = face_idx + 1
            emitted_before = len(self.robot_offsets)

            items = _mixture_items(mix)
            for item_idx, item in enumerate(items):
                box_type = item.get('Type', item.get('type', ''))
                num = item.get('Num', item.get('num', 0))
                pos = item.get('Pos', item.get('pos', {})) or {}
                try:
                    num = int(num)
                    pos_x = float(pos.get('X', pos.get('x', 0.0)))
                    pos_y = float(pos.get('Y', pos.get('y', 0.0)))
                    pos_z = float(pos.get('Z', pos.get('z', 0.0)))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Mixture 面{num_F}第{item_idx + 1}项数值格式错误") from exc

                if not box_type:
                    raise ValueError(f"Mixture 面{num_F}第{item_idx + 1}项缺少箱型 Type")
                if num <= 0:
                    raise ValueError(
                        f"Mixture 面{num_F}第{item_idx + 1}项 Num 必须大于 0，当前为 {num}")
                if not all(math.isfinite(v) for v in (pos_x, pos_y, pos_z)):
                    raise ValueError(
                        f"Mixture 面{num_F}第{item_idx + 1}项 Pos 含非有限数值")
                if min(pos_x, pos_y, pos_z) < 0:
                    raise ValueError(
                        f"Mixture 面{num_F}第{item_idx + 1}项 Pos 不能为负数")

                params = self._box_params(box_type)
                box_size = params['size']
                box_l, box_w, box_h = box_size
                if num > params['grab_p1']:
                    raise ValueError(
                        f"Mixture 面{num_F}第{item_idx + 1}项 Num={num} 超出"
                        f"箱型 {box_type} 的 P1 单抓能力 {params['grab_p1']}")
                if pos_y + num * box_w > self.W:
                    raise ValueError(
                        f"Mixture 面{num_F}第{item_idx + 1}项沿车宽超界："
                        f"Y({pos_y}) + Num({num})×箱宽({box_w}) > {self.W}")
                if pos_x + box_l > self.L:
                    raise ValueError(
                        f"Mixture 面{num_F}第{item_idx + 1}项沿车深超界："
                        f"X({pos_x}) + 箱长({box_l}) > {self.L}")
                if pos_z + box_h > self.H:
                    raise ValueError(
                        f"Mixture 面{num_F}第{item_idx + 1}项沿车高超界："
                        f"Z({pos_z}) + 箱高({box_h}) > {self.H}")

                # 接口与内部坐标一致：X=车深、Y=车宽、Z=车高。
                internal_pos = [pos_x, pos_y, pos_z]
                center_y = pos_y + num * box_w * 0.5
                dir_ = 1 if center_y <= self.W * 0.5 else 2
                self._emit(
                    'p1', num, num_F, dir_, internal_pos,
                    box_type=box_type, box_size=box_size)

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

        # 常规/梯形 area_cfg：按同面、同区域、同高度的 y 顺序确定左/中/右。
        # 混装面使用三维邻箱关系：4=当前高度的墙边收尾抓，其余为1。
        groups = defaultdict(list)
        for offset in self.robot_offsets:
            if offset == 'done':
                continue
            groups[(offset['num_F'], offset['area'], offset['pos'][2])].append(offset)
        id_to_cfg = {}
        for offsets in groups.values():
            sorted_ids = [o['id'] for o in sorted(offsets, key=lambda o: o['pos'][1])]
            n = len(sorted_ids)
            for rank, oid in enumerate(sorted_ids):
                if n == 1 or rank == 0:
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

    def __init__(self, config_data):
        self.reserve_grip = config_data['reserve_grip']
        self.reserve_object = config_data['reserve_object']
        self.mixture_z_overlap_min = float(
            config_data.get('mixture_z_overlap_min', 20.0))
        self.objects = []          # 含 reserve_object 外扩的包围盒，用于碰撞检测
        self.display_objects = []  # 真实尺寸的箱体，用于可视化

    def reset(self):
        self.objects = []
        self.display_objects = []

    @staticmethod
    def _aabb_intersects(a, b, tol=0.5):
        """判断两个三维 AABB 是否存在实质重叠，边界接触不算碰撞。"""
        return all(a[i] < b[i + 3] - tol and a[i + 3] > b[i] + tol
                   for i in range(3))

    def mixture_side_clearance(self, action, left_wall, right_wall,
                               min_z_overlap=None):
        """按真实已码箱体计算混装抓取目标左右两侧的可用空间。

        与规则垛的“底部 Z 完全相等”不同，这里使用 X/Z 包围盒的有效重叠
        判断某个已码箱体是否会影响当前抓的横向进入。返回的 blocking 表示
        目标位置本身已经与已有箱体重叠，由调用方决定回退或报警。
        """
        if action['area'] != 'p1':
            raise ValueError("混装动态 APP 当前仅支持 P1 区域")
        if min_z_overlap is None:
            min_z_overlap = self.mixture_z_overlap_min

        current_boxes = self.to_box(action)
        if not current_boxes:
            raise ValueError("当前混装动作没有可用箱体")

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
        pass


class Node:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None
