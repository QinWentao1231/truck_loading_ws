"""什邡旧版正在使用的垛序规则快照，不包含RRT搜索或旧碰撞环境。"""

import copy


class OriginalRobotPosition:
    def __init__(self, config_data):
        # box
        self.box_type = config_data['box_list'][0]
        self.l = config_data['box'][self.box_type]['size']['L']
        self.w = config_data['box'][self.box_type]['size']['W']
        self.h = config_data['box'][self.box_type]['size']['H']
        # car
        self.L = round(config_data['car']['size']['L'])
        self.W = round(config_data['car']['size']['W'])
        self.H = round(config_data['car']['size']['H'])
        # palletizing
        self.N1 = config_data['regular'][0]['N1']
        self.N2 = config_data['regular'][0]['N2']
        self.N3 = config_data['regular'][0]['N3']
        self.T12 = config_data['regular'][0]['T12']
        self.T3 = config_data['regular'][0]['T3']
        self.F13 = config_data['regular'][0]['F13']
        self.F2 = config_data['regular'][0]['F2']
        self.E = config_data['regular'][0]['E']
        self.Nx = config_data['regular'][0]['Nx']
        self.Stack = config_data['regular'][0]['Stack']
        self.Group = config_data['regular'][0]['Group']
        self.trapezoid = config_data['trapezoid']
        self.box_count = (self.N1 * self.T12 * self.F13 + self.N2 * self.T12 * self.F2 + self.N3 * self.T3 * self.F13 + self.Nx +
                          sum(x['N1'] * x['T1'] + x['N3'] * x['T3'] + x['Nx'] for x in self.trapezoid))
        # robot
        self.grab_num_p1 = 4
        self.grab_num_p2 = 2
        self.grab_num_p3 = 2
        # initialization
        self.robot_offsets = []
        self.fit_offsets = []
        self.ori_offsets = []
        self.boxes = []
        self.paths = []
        # for test
        # self.L = 2239
        # self.W = 2400 - 50
        # self.H = 2849
        # self.N1 = 9
        # self.N2 = 0
        # self.N3 = 0
        # self.F2 = 0
        # self.cal_robot_offset()
        self.read_robot_offset()

    def read_robot_offset(self):
        id_ = 0
        E_ = copy.deepcopy(self.Nx)
        gap = self.W - self.N1 * self.w
        stack_ = self.Stack[0]
        group_ = self.Group[0]
        rev_stack = self.Stack[1][::-1]
        rev_group = self.Group[1][::-1]
        for f in range(0, self.F13):
            # 常规垛P1
            for t in range(0, self.T12):
                if (f+t) % 2 == 0:
                    for n in range(0, len(group_)):
                        id_ += 1
                        self.robot_offsets.append({
                            'id': id_, 'area': 'p1', 'num': group_[n], 'num_F': f+1, 'action': 0, 'dir': 1,
                            'pos': [0,
                                    gap*sum(stack_[:sum(group_[:n])+1])*0.01 + sum(group_[:n])*self.w,
                                    t * self.h],
                            'size': [self.l, self.w, self.h]
                        })
                        self.boxes.append({'id': id_, 'area': 'p1', 'num': group_[n], 'num_F': f+1, 'action': 0})
                else:
                    for n in range(0, len(rev_group)):
                        id_ += 1
                        self.robot_offsets.append({
                            'id': id_, 'area': 'p1', 'num': rev_group[n], 'num_F': f+1, 'action': 0, 'dir': 2,
                            'pos': [0,
                                    self.W-gap*sum(rev_stack[:sum(rev_group[:n])+1])*0.01-sum(rev_group[:(n+1)])*self.w,
                                    t * self.h],
                            'size': [self.l, self.w, self.h]
                        })
                        self.boxes.append({'id': id_, 'area': 'p1', 'num': rev_group[n], 'num_F': f+1, 'action': 0})
            # 常规垛P3
            if self.N3 != 0:
                for t in range(0, self.T3):
                    first_p3 = self.N3 % self.grab_num_p3 if self.N3 % self.grab_num_p3 != 0 else self.grab_num_p3
                    id_ += 1
                    self.robot_offsets.append({
                        'id': id_, 'area': 'p3', 'num': first_p3, 'num_F': f + 1, 'action': 0, 'dir': 1,
                        'pos': [0, 0 + 50, t * self.w + self.T12 * self.h],
                        'size': [self.l, self.w, self.h]
                    })
                    self.boxes.append({
                        'id': id_, 'area': 'p3', 'num': first_p3, 'num_F': f + 1, 'action': 0
                    })
                    for n in range(0, int((self.N3 - first_p3) / self.grab_num_p3)):
                        id_ += 1
                        self.robot_offsets.append({
                            'id': id_, 'area': 'p3', 'num': self.grab_num_p3, 'num_F': f + 1, 'action': 0, 'dir': 1,
                            'pos': [0, 50 + (first_p3 + n * self.grab_num_p3) * self.h, t * self.w + self.T12 * self.h],
                            'size': [self.l, self.w, self.h]
                        })
                        self.boxes.append({
                            'id': id_, 'area': 'p3', 'num': self.grab_num_p3, 'num_F': f + 1, 'action': 0
                        })
            # 常规垛顶置尾料
            if self.E == 2 and E_ != 0:
                if self.N3 == 0:
                    N3_ = int((self.W-80) // self.h)
                else:
                    N3_ = self.N3
                Nx_ =  N3_ if E_ - N3_ >= 0 else E_
                first_p3 = Nx_ % self.grab_num_p3 if Nx_ % self.grab_num_p3 != 0 else self.grab_num_p3
                id_ += 1
                self.robot_offsets.append({
                    'id': id_, 'area': 'p3', 'num': first_p3, 'num_F': f + 1, 'action': 0, 'dir': 1,
                    'pos': [0, 0 + 50, self.T3 * self.w + self.T12 * self.h],
                    'size': [self.l, self.w, self.h]
                })
                self.boxes.append({
                    'id': id_, 'area': 'p3', 'num': first_p3, 'num_F': f + 1, 'action': 0
                })
                for n in range(0, int((Nx_ - first_p3) / self.grab_num_p3)):
                    id_ += 1
                    self.robot_offsets.append({
                        'id': id_, 'area': 'p3', 'num': self.grab_num_p3, 'num_F': f + 1, 'action': 0, 'dir': 1,
                        'pos': [0, 50+(first_p3 + n * self.grab_num_p3)*self.h, self.T3*self.w + self.T12*self.h],
                        'size': [self.l, self.w, self.h]
                    })
                    self.boxes.append({
                        'id': id_, 'area': 'p3', 'num': self.grab_num_p3, 'num_F': f + 1, 'action': 0
                    })
                E_ -= Nx_
            self.robot_offsets[-1]['action'] = 1
            self.boxes[-1]['action'] = 1
        # 常规垛前置尾料
        if (self.Nx != 0) and (len(self.trapezoid) == 0) and (self.E == 1):
            Nx_t = self.Nx // self.N1
            Nx_ = self.Nx % self.N1
            for t in range(0, Nx_t):
                first_p1 = (self.N1 % self.grab_num_p1) if self.N1 % self.grab_num_p1 != 0 else self.grab_num_p1
                if first_p1 == 1:
                    first_p1 = 2
                id_ += 1
                self.robot_offsets.append({
                    'id': id_, 'area': 'p1', 'num': first_p1, 'num_F': self.F13 + 1, 'action': 0, 'dir': 1,
                    'pos': [0, 0, t * self.h], 'size': [self.l, self.w, self.h]
                })
                self.boxes.append({'id': id_, 'area': 'p1', 'num': first_p1,
                                   'num_F': self.F13 + 1, 'action': 0})
                second_p1 = (self.N1 - first_p1) % self.grab_num_p1 \
                    if (self.N1 - first_p1) % self.grab_num_p1 != 0 else self.grab_num_p1
                id_ += 1
                self.robot_offsets.append({
                    'id': id_, 'area': 'p1', 'num': second_p1, 'num_F': self.F13 + 1, 'action': 0, 'dir': 1,
                    'pos': [0, first_p1 * self.w, t * self.h], 'size': [self.l, self.w, self.h]
                })
                self.boxes.append({'id': id_, 'area': 'p1', 'num': second_p1, 'num_F': self.F13 + 1, 'action': 0})
                for i in range(0, int((self.N1 - first_p1 - second_p1) / self.grab_num_p1)):
                    id_ += 1
                    self.robot_offsets.append({
                        'id': id_, 'area': 'p1', 'num': self.grab_num_p1, 'num_F': self.F13 + 1, 'action': 0,
                        'dir': 1,
                        'pos': [0, (first_p1 + second_p1 + i * self.grab_num_p1) * self.w, t * self.h],
                        'size': [self.l, self.w, self.h]
                    })
                    self.boxes.append({
                        'id': id_, 'area': 'p1', 'num': self.grab_num_p1, 'num_F': self.F13 + 1, 'action': 0
                    })
            if Nx_ != 0:
                first_p1 = Nx_ % self.grab_num_p1 if Nx_ % self.grab_num_p1 != 0 else self.grab_num_p1
                if first_p1 == 1:
                    first_p1 = 2
                if (Nx_ - first_p1) >= 0:
                    id_ += 1
                    self.robot_offsets.append({
                        'id': id_, 'area': 'p1', 'num': first_p1, 'num_F': self.F13 + 1, 'action': 0, 'dir': 1,
                        'pos': [0, 0, Nx_t * self.h], 'size': [self.l, self.w, self.h]
                    })
                    self.boxes.append({'id': id_, 'area': 'p1', 'num': first_p1, 'num_F': self.F13+1, 'action': 0})
                else:
                    first_p1 = 0
                if first_p1 < Nx_:
                    second_p1 = (Nx_ - first_p1) % self.grab_num_p1 \
                        if (Nx_ - first_p1) % self.grab_num_p1 != 0 else self.grab_num_p1
                    id_ += 1
                    self.robot_offsets.append({
                        'id': id_, 'area': 'p1', 'num': second_p1, 'num_F': self.F13 + 1, 'action': 0, 'dir': 1,
                        'pos': [0, first_p1 * self.w, Nx_t * self.h], 'size': [self.l, self.w, self.h]
                    })
                    self.boxes.append({'id': id_, 'area': 'p1', 'num': second_p1, 'num_F': self.F13+1, 'action': 0})
                    for i in range(0, int((Nx_ - first_p1 - second_p1) / self.grab_num_p1)):
                        id_ += 1
                        self.robot_offsets.append({
                            'id': id_, 'area': 'p1', 'num': self.grab_num_p1, 'num_F': self.F13 + 1, 'action': 0,
                            'dir': 1,
                            'pos': [0, (first_p1 + second_p1 + i * self.grab_num_p1) * self.w, Nx_t * self.h],
                            'size': [self.l, self.w, self.h]
                        })
                        self.boxes.append({
                            'id': id_, 'area': 'p1', 'num': self.grab_num_p1, 'num_F': self.F13 + 1, 'action': 0
                        })
        # 梯形垛
        for f in range(0, len(self.trapezoid)):
            gap = self.W - self.trapezoid[f]['N1'] * self.w
            stack_ = self.trapezoid[f]['Stack'][0]
            group_ = self.trapezoid[f]['Group'][0]
            rev_stack = self.trapezoid[f]['Stack'][1][::-1]
            rev_group = self.trapezoid[f]['Group'][1][::-1]
            # 梯形P1
            for t in range(0, self.trapezoid[f]['T1']):
                if self.trapezoid[f]['Isdoor']:
                    nx_p1 = self.trapezoid[f]['N1'] % self.grab_num_p1 \
                        if self.trapezoid[f]['N1'] % self.grab_num_p1 != 0 else self.grab_num_p1
                    if nx_p1 == 1:
                        nx_p1 = 2
                    id_ += 1
                    self.robot_offsets.append({'id': id_, 'area': 'p1', 'num': nx_p1, 'num_F': self.F13 + f + 1,
                                               'action': 0, 'dir': 1,
                                               'pos': [0, 0, t * self.h],
                                               'size': [self.l, self.w, self.h]})
                    self.boxes.append({
                        'id': id_, 'area': 'p1', 'num': nx_p1, 'num_F': self.F13 + f + 1, 'action': 0
                    })
                    if nx_p1 >= self.trapezoid[f]['N1']:
                        continue
                    nx_p1_sec = (self.trapezoid[f]['N1'] - nx_p1) % self.grab_num_p1 \
                        if (self.trapezoid[f]['N1'] - nx_p1) % self.grab_num_p1 != 0 else self.grab_num_p1
                    id_ += 1
                    self.robot_offsets.append(
                        ({'id': id_, 'area': 'p1', 'num': nx_p1_sec, 'num_F': self.F13 + f + 1,
                          'action': 0, 'dir': 1,
                          'pos': [0, nx_p1 * self.w, t * self.h],
                          'size': [self.l, self.w, self.h]}))
                    self.boxes.append({
                        'id': id_, 'area': 'p1', 'num': nx_p1_sec, 'num_F': self.F13 + f + 1, 'action': 0
                    })
                    if (nx_p1 + nx_p1_sec) >= self.trapezoid[f]['N1']:
                        continue
                    for n in range(0, int((self.trapezoid[f]['N1'] - nx_p1 - nx_p1_sec) / self.grab_num_p1)):
                        id_ += 1
                        self.boxes.append({
                            'id': id_, 'area': 'p1', 'num': self.grab_num_p1, 'num_F': self.F13 + f + 1,
                            'action': 0
                        })
                        self.robot_offsets.append({'id': id_, 'area': 'p1', 'num': self.grab_num_p1,
                                                   'num_F': self.F13 + f + 1, 'action': 0, 'dir': 1,
                                                   'pos': [0, (nx_p1 + nx_p1_sec + n * self.grab_num_p1) * self.w,
                                                           t * self.h],
                                                   'size': [self.l, self.w, self.h]})
                else:
                    if (self.F13 + f + t) % 2 == 0:
                        for n in range(0, len(group_)):
                            id_ += 1
                            self.robot_offsets.append({
                                'id': id_, 'area': 'p1', 'num': group_[n], 'num_F': self.F13+f+1,
                                'action': 0, 'dir': 1,
                                'pos': [0,
                                        gap*sum(stack_[:sum(group_[:n])+1])*0.01 + sum(group_[:n])*self.w,
                                        t * self.h],
                                'size': [self.l, self.w, self.h]
                            })
                            self.boxes.append({'id': id_, 'area': 'p1', 'num': group_[n],
                                               'num_F': self.F13+f+1, 'action': 0})
                    else:
                        for n in range(0, len(rev_group)):
                            id_ += 1
                            self.robot_offsets.append({
                                'id': id_, 'area': 'p1', 'num': rev_group[n], 'num_F': self.F13+f+1,
                                'action': 0, 'dir': 2,
                                'pos': [0,
                                        self.W - gap*sum(rev_stack[:sum(rev_group[:n])+1])*0.01 -
                                        sum(rev_group[:(n+1)])*self.w,
                                        t * self.h],
                                'size': [self.l, self.w, self.h]
                            })
                            self.boxes.append({'id': id_, 'area': 'p1', 'num': rev_group[n],
                                               'num_F': self.F13+f+1, 'action': 0})
            # 梯形 P3
            for t in range(0, self.trapezoid[f]['T3']):
                nx_p3 = self.trapezoid[f]['N3'] % self.grab_num_p3 \
                    if self.trapezoid[f]['N3'] % self.grab_num_p3 != 0 else self.grab_num_p3
                id_ += 1
                self.robot_offsets.append({'id': id_, 'area': 'p3', 'num': nx_p3, 'num_F': self.F13 + f + 1,
                                           'action': 0, 'dir': 1,
                                           'pos': [0, 0 + 50, t * self.w + self.trapezoid[f]['T1'] * self.h],
                                           'size': [self.l, self.w, self.h]})
                self.boxes.append({
                    'id': id_, 'area': 'p3', 'num': nx_p3, 'num_F': self.F13 + f + 1, 'action': 0
                })
                for n in range(0, int((self.trapezoid[f]['N3'] - nx_p3) / self.grab_num_p3)):
                    id_ += 1
                    self.robot_offsets.append({'id': id_, 'area': 'p3', 'num': self.grab_num_p3,
                                               'num_F': self.F13 + f + 1, 'action': 0, 'dir': 1,
                                               'pos': [0, 50+(nx_p3 + n * self.grab_num_p3) * self.h,
                                                       t * self.w + self.trapezoid[f]['T1'] * self.h],
                                               'size': [self.l, self.w, self.h]})
                    self.boxes.append({
                        'id': id_, 'area': 'p3', 'num': self.grab_num_p3, 'num_F': self.F13 + f + 1, 'action': 0
                    })
            # 梯形尾料
            if self.trapezoid[f]['Nx'] != 0:
                nx_end_1 = self.trapezoid[f]['Nx'] % self.grab_num_p3 \
                    if self.trapezoid[f]['Nx'] % self.grab_num_p3 != 0 else self.grab_num_p3
                id_ += 1
                self.robot_offsets.append({
                    'id': id_, 'area': 'p3', 'num': nx_end_1, 'num_F': self.F13 + f + 1, 'action': 0, 'dir': 1,
                    'pos': [0, 50 + 0, self.trapezoid[f]['T1'] * self.h + self.trapezoid[f]['T3'] * self.w],
                    'size': [self.l, self.w, self.h]})
                self.boxes.append({
                    'id': id_, 'area': 'p3', 'num': nx_end_1, 'num_F': self.F13 + f + 1, 'action': 0
                })
                for n in range(0, int((self.trapezoid[f]['Nx'] - nx_end_1) / self.grab_num_p3)):
                    id_ += 1
                    self.robot_offsets.append({
                        'id': id_, 'area': 'p3', 'num': self.grab_num_p3, 'num_F': self.F13 + f + 1,
                        'action': 0, 'dir': 1,
                        'pos': [0,
                                50 + (nx_end_1 + n * self.grab_num_p3) * self.h,
                                self.trapezoid[f]['T1'] * self.h + self.trapezoid[f]['T3'] * self.w],
                        'size': [self.l, self.w, self.h]})
                    self.boxes.append({
                        'id': id_, 'area': 'p3', 'num': self.grab_num_p3, 'num_F': self.F13 + f + 1,
                        'action': 0
                    })
            self.robot_offsets[-1]['action'] = 1
            self.boxes[-1]['action'] = 1
        self.robot_offsets[-1]['action'] = 2
        self.boxes[-1]['action'] = 2
        self.robot_offsets.append('done')
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
