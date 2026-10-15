class Box:

    def __init__(self, l, w, h, rotation_type, position):
        if rotation_type == 0:
            self.length = l
            self.width = w
            self.height = h
            self.position = position
        if rotation_type == 1:
            self.length = w
            self.width = l
            self.height = h
            self.position = position
        if rotation_type == 2:
            self.length = l
            self.width = h
            self.height = w
            self.position = position


class RobotPosition:

    def __init__(self, config_data):
        # box
        box_type = config_data['box_list'][0]
        self.l = config_data['box'][box_type]['size']['L']
        self.w = config_data['box'][box_type]['size']['W']
        self.h = config_data['box'][box_type]['size']['H']
        # car
        self.L = int(config_data['car']['size']['L'])
        self.W = int(config_data['car']['size']['W'])
        self.H = int(config_data['car']['size']['H'])
        # palletizing
        self.N1 = config_data['regular'][0]['N1']
        self.N2 = config_data['regular'][0]['N2']
        self.N3 = config_data['regular'][0]['N3']
        self.T12 = config_data['regular'][0]['T12']
        self.T3 = config_data['regular'][0]['T3']
        self.F13 = config_data['regular'][0]['F13']
        self.F2 = config_data['regular'][0]['F2']
        self.Nx = config_data['regular'][0]['Nx']
        self.trapezoid = config_data['trapezoid']
        self.box_count = ((self.N1 * self.T12 * self.F13) + (self.N2 * self.T12 * self.F2) +
                          (self.N3 * self.T3 * self.F13)) + config_data['regular'][0]['Nx']
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
        self.cal_robot_offset()

    def cal_robot_offset(self):
        p2_filled, grab_p2, p2_f, index = False, 0, 0, 0
        for f in range(0, self.F13):
            for t in range(0, self.T12):
                first_p1 = (self.N1 % self.grab_num_p1) if self.N1 % self.grab_num_p1 != 0 else self.grab_num_p1
                if first_p1 == 1:
                    first_p1 = 2
                if (self.N1 - first_p1) != 0:
                    index += 1
                    self.robot_offsets.append({
                        'id': index, 'area': 'p1', 'num': first_p1, 'num_F': f + 1, 'action': 0,
                        'pos': [0, 0, t * self.h], 'size': [self.l, self.w, self.h]
                    })
                    self.boxes.append({'id': index, 'area': 'p1', 'num': first_p1, 'num_F': f + 1, 'action': 0})
                else:
                    first_p1 = 0
                if self.N2 == 0:
                    index += 1
                    second_p1 = (self.N1 - first_p1) % self.grab_num_p1 \
                        if (self.N1 - first_p1) % self.grab_num_p1 != 0 else self.grab_num_p1
                    self.robot_offsets.append({
                        'id': index, 'area': 'p1', 'num': second_p1, 'num_F': f + 1, 'action': 0,
                        'pos': [0, first_p1 * self.w, t * self.h], 'size': [self.l, self.w, self.h]
                    })
                    self.boxes.append({'id': index, 'area': 'p1', 'num': second_p1, 'num_F': f + 1, 'action': 0})
                    third_p1 = self.grab_num_p1 if self.N1 - first_p1 - second_p1 != 0 else 0
                    if third_p1 != 0:
                        index += 1
                        self.robot_offsets.append({
                            'id': index, 'area': 'p1', 'num': third_p1, 'num_F': f + 1, 'action': 0,
                            'pos': [0, self.W - 30 - third_p1*self.w, t*self.h],
                            'size': [self.l, self.w, self.h]
                        })
                        self.boxes.append({
                            'id': index, 'area': 'p1', 'num': third_p1, 'num_F': f + 1, 'action': 0
                        })
                    for i in range(0, int((self.N1 - first_p1 - second_p1 - third_p1) / self.grab_num_p1)):
                        index += 1
                        self.robot_offsets.append({
                            'id': index, 'area': 'p1', 'num': self.grab_num_p1, 'num_F': f + 1, 'action': 0,
                            'pos': [0, (first_p1 + second_p1 + i * self.grab_num_p1) * self.w, t * self.h],
                            'size': [self.l, self.w, self.h]
                        })
                        self.boxes.append({
                            'id': index, 'area': 'p1', 'num': self.grab_num_p1, 'num_F': f + 1, 'action': 0
                        })
                else:
                    if p2_f < self.F2 and ((f + 1) * self.l) > (p2_f * self.w):
                        p2_filled = True
                        for i in range(0, self.N2):
                            grab_p2 = self.grab_num_p2 if (p2_f + self.grab_num_p2) < self.F2 else self.F2 - p2_f
                            index += 1
                            self.robot_offsets.append({
                                'id': index, 'area': 'p2', 'num': grab_p2, 'num_F': f + 1, 'action': 0,
                                'pos': [-f * self.l + p2_f * self.w, self.W - (i + 1) * self.l, t * self.h],
                                'size': [self.l, self.w, self.h]
                            })
                            self.boxes.append({
                                'id': index, 'area': 'p2', 'num': grab_p2, 'num_F': f + 1, 'action': 0
                            })
                    index += 1
                    second_p1 = (self.N1 - first_p1) % self.grab_num_p1 \
                        if (self.N1 - first_p1) % self.grab_num_p1 != 0 else self.grab_num_p1
                    self.robot_offsets.append({
                        'id': index, 'area': 'p1', 'num': second_p1, 'num_F': f + 1, 'action': 0,
                        'pos': [0, first_p1 * self.w, t * self.h], 'size': [self.l, self.w, self.h]
                    })
                    self.boxes.append({'id': index, 'area': 'p1', 'num': second_p1, 'num_F': f + 1, 'action': 0})
                    for i in range(0, int((self.N1 - first_p1 - second_p1 - third_p1) / self.grab_num_p1)):
                        index += 1
                        self.robot_offsets.append({
                            'id': index, 'area': 'p1', 'num': self.grab_num_p1, 'num_F': f + 1, 'action': 0,
                            'pos': [0, (first_p1 + second_p1 + i * self.grab_num_p1) * self.w, t * self.h],
                            'size': [self.l, self.w, self.h]
                        })
                        self.boxes.append({
                            'id': index, 'area': 'p1', 'num': self.grab_num_p1, 'num_F': f + 1, 'action': 0
                        })
            if self.N3 != 0:
                for j in range(0, self.T3):
                    first_p3 = self.N3 % self.grab_num_p3 if self.N3 % self.grab_num_p3 != 0 else self.grab_num_p3
                    index += 1
                    self.robot_offsets.append({
                        'id': index, 'area': 'p3', 'num': first_p3, 'num_F': f + 1, 'action': 0,
                        'pos': [0, 0, j * self.w + self.T12 * self.h],
                        'size': [self.l, self.w, self.h]
                    })
                    self.boxes.append({
                        'id': index, 'area': 'p3', 'num': first_p3, 'num_F': f + 1, 'action': 0
                    })
                    for i in range(0, int((self.N3 - first_p3) / self.grab_num_p3)):
                        index += 1
                        self.robot_offsets.append({
                            'id': index, 'area': 'p3', 'num': self.grab_num_p3, 'num_F': f + 1, 'action': 0,
                            'pos': [0, (first_p3 + i * self.grab_num_p3) * self.h, j * self.w + self.T12 * self.h],
                            'size': [self.l, self.w, self.h]
                        })
                        self.boxes.append({
                            'id': index, 'area': 'p3', 'num': self.grab_num_p3, 'num_F': f + 1, 'action': 0
                        })
            if p2_filled:
                p2_f += grab_p2
                p2_filled = False
            self.robot_offsets[-1]['action'] = 1
            self.boxes[-1]['action'] = 1
        if (self.Nx != 0) and (len(self.trapezoid) == 0):
            Nx_t = self.Nx // self.N1
            Nx_ = self.Nx % self.N1
            if Nx_t != 0:
                for t in range(0, Nx_t):
                    first_p1 = (self.N1 % self.grab_num_p1) if self.N1 % self.grab_num_p1 != 0 else self.grab_num_p1
                    if first_p1 == 1:
                        first_p1 = 2
                    if (self.N1 - first_p1) != 0:
                        index += 1
                        self.robot_offsets.append({
                            'id': index, 'area': 'p1', 'num': first_p1, 'num_F': self.F13 + 1, 'action': 0,
                            'pos': [0, 0, t * self.h], 'size': [self.l, self.w, self.h]
                        })
                        self.boxes.append({'id': index, 'area': 'p1', 'num': first_p1,
                                           'num_F': self.F13 + 1, 'action': 0})
                    else:
                        first_p1 = 0
                    index += 1
                    second_p1 = (self.N1 - first_p1) % self.grab_num_p1 \
                        if (self.N1 - first_p1) % self.grab_num_p1 != 0 else self.grab_num_p1
                    self.robot_offsets.append({
                        'id': index, 'area': 'p1', 'num': second_p1, 'num_F': self.F13 + 1, 'action': 0,
                        'pos': [0, first_p1 * self.w, t * self.h], 'size': [self.l, self.w, self.h]
                    })
                    self.boxes.append({'id': index, 'area': 'p1', 'num': second_p1, 'num_F': self.F13 + 1, 'action': 0})
                    third_p1 = self.grab_num_p1 if self.N1 - first_p1 - second_p1 != 0 else 0
                    if third_p1 != 0:
                        index += 1
                        self.robot_offsets.append({
                            'id': index, 'area': 'p1', 'num': third_p1, 'num_F': self.F13 + 1, 'action': 0,
                            'pos': [0, self.W - 30 - third_p1*self.w, t*self.h],
                            'size': [self.l, self.w, self.h]
                        })
                        self.boxes.append({
                            'id': index, 'area': 'p1', 'num': third_p1, 'num_F': self.F13 + 1, 'action': 0
                        })
                    for i in range(0, int((self.N1 - first_p1 - second_p1 - third_p1) / self.grab_num_p1)):
                        index += 1
                        self.robot_offsets.append({
                            'id': index, 'area': 'p1', 'num': self.grab_num_p1, 'num_F': self.F13 + 1, 'action': 0,
                            'pos': [0, (first_p1 + second_p1 + i * self.grab_num_p1) * self.w, t * self.h],
                            'size': [self.l, self.w, self.h]
                        })
                        self.boxes.append({
                            'id': index, 'area': 'p1', 'num': self.grab_num_p1, 'num_F': self.F13 + 1, 'action': 0
                        })
            if Nx_ != 0:
                first_p1 = Nx_ % self.grab_num_p1 if Nx_ % self.grab_num_p1 != 0 else self.grab_num_p1
                if first_p1 == 1:
                    first_p1 = 2
                if (Nx_ - first_p1) > 0:
                    index += 1
                    self.robot_offsets.append({
                        'id': index, 'area': 'p1', 'num': first_p1, 'num_F': self.F13 + 1, 'action': 0,
                        'pos': [0, 0, Nx_t * self.h], 'size': [self.l, self.w, self.h]
                    })
                    self.boxes.append({'id': index, 'area': 'p1', 'num': first_p1, 'num_F': self.F13 + 1, 'action': 0})
                else:
                    first_p1 = 0
                second_p1 = (Nx_- first_p1) % self.grab_num_p1 \
                    if (Nx_ - first_p1) % self.grab_num_p1 != 0 else self.grab_num_p1
                index += 1
                self.robot_offsets.append({
                    'id': index, 'area': 'p1', 'num': second_p1, 'num_F': self.F13 + 1, 'action': 0,
                    'pos': [0, first_p1 * self.w, Nx_t * self.h], 'size': [self.l, self.w, self.h]
                })
                self.boxes.append({'id': index, 'area': 'p1', 'num': second_p1, 'num_F': self.F13 + 1, 'action': 0})
                if (Nx_ - first_p1 - second_p1) != 0:
                    for i in range(0, int((Nx_ - first_p1 - second_p1) / self.grab_num_p1)):
                        index += 1
                        self.robot_offsets.append({
                            'id': index, 'area': 'p1', 'num': self.grab_num_p1, 'num_F': self.F13 + 1, 'action': 0,
                            'pos': [0, (first_p1 + second_p1 + i * self.grab_num_p1) * self.w, Nx_t * self.h],
                            'size': [self.l, self.w, self.h]
                        })
                        self.boxes.append({
                            'id': index, 'area': 'p1', 'num': self.grab_num_p1, 'num_F': self.F13 + 1, 'action': 0
                        })
        # 'trapezoid': [{'N1': 11, 'N3': 5, 'T1': 1, 'T3': 7, 'Nx': 0}, {'N1': 11, 'N3': 5, 'T1': 1, 'T3': 7, 'Nx': 0},
        #               {'N1': 11, 'N3': 5, 'T1': 1, 'T3': 7, 'Nx': 0}, {'N1': 11, 'N3': 5, 'T1': 1, 'T3': 6, 'Nx': 2}]
        if len(self.trapezoid) != 0:
            for t in range(0, len(self.trapezoid)):
                # 梯形 P1
                for i in range(0, self.trapezoid[t]['T1']):
                    nx_p1 = self.trapezoid[t]['N1'] % self.grab_num_p1 \
                        if self.trapezoid[t]['N1'] % self.grab_num_p1 != 0 else self.grab_num_p1
                    if nx_p1 == 1:
                        nx_p1 = 2
                    index += 1
                    self.robot_offsets.append({'id': index, 'area': 'p1', 'num': nx_p1, 'num_F': self.F13 + t + 1,
                                               'action': 0, 'pos': [0, 0, i * self.h],
                                               'size': [self.l, self.w, self.h]})
                    self.boxes.append({
                        'id': index, 'area': 'p1', 'num': nx_p1, 'num_F': self.F13 + t + 1, 'action': 0
                    })
                    if nx_p1 >= self.trapezoid[t]['N1']:
                        self.robot_offsets[-1]['action'] = 1
                        self.boxes[-1]['action'] = 1
                        break
                    nx_p1_sec = (self.trapezoid[t]['N1'] - nx_p1) % self.grab_num_p1 \
                        if (self.trapezoid[t]['N1'] - nx_p1) % self.grab_num_p1 != 0 else self.grab_num_p1
                    index += 1
                    self.robot_offsets.append(({'id': index, 'area': 'p1', 'num': nx_p1_sec, 'num_F': self.F13 + t + 1,
                                                'action': 0,
                                                'pos': [0, nx_p1 * self.w, i * self.h],
                                                'size': [self.l, self.w, self.h]}))
                    self.boxes.append({
                        'id': index, 'area': 'p1', 'num': nx_p1_sec, 'num_F': self.F13 + t + 1, 'action': 0
                    })
                    if (nx_p1 + nx_p1_sec) >= self.trapezoid[t]['N1']:
                        self.robot_offsets[-1]['action'] = 1
                        self.boxes[-1]['action'] = 1
                        break
                    nx_p1_trd = self.grab_num_p1 if self.N1 - first_p1 - second_p1 != 0 else 0
                    if nx_p1_trd != 0:
                        index += 1
                        self.robot_offsets.append({
                            'id': index, 'area': 'p1', 'num': nx_p1_trd, 'num_F': self.F13 + t + 1, 'action': 0,
                            'pos': [0, self.W-30-(self.N1-self.trapezoid[t]['N1'])*self.w-nx_p1_trd*self.w, i * self.h],
                            'size': [self.l, self.w, self.h]
                        })
                        self.boxes.append({
                            'id': index, 'area': 'p1', 'num': nx_p1_trd, 'num_F': self.F13 + t + 1, 'action': 0
                        })
                    for j in range(0, int((self.trapezoid[t]['N1'] - nx_p1 - nx_p1_sec - nx_p1_trd) / self.grab_num_p1)):
                        index += 1
                        self.robot_offsets.append({'id': index, 'area': 'p1', 'num': self.grab_num_p1,
                                                   'num_F': self.F13 + t + 1, 'action': 0,
                                                   'pos': [0, (nx_p1 + nx_p1_sec + j * self.grab_num_p1) * self.w,
                                                           i * self.h],
                                                   'size': [self.l, self.w, self.h]})
                        self.boxes.append({
                            'id': index, 'area': 'p1', 'num': self.grab_num_p1, 'num_F': self.F13 + t + 1, 'action': 0
                        })
                # 梯形 P3
                for i in range(0, self.trapezoid[t]['T3']):
                    nx_p3 = self.trapezoid[t]['N3'] % self.grab_num_p3 \
                        if self.trapezoid[t]['N3'] % self.grab_num_p3 != 0 else self.grab_num_p3
                    index += 1
                    self.robot_offsets.append({'id': index, 'area': 'p3', 'num': nx_p3, 'num_F': self.F13 + t + 1,
                                               'action': 0, 'pos': [0, 0, 0], 'size': [self.l, self.w, self.h]})
                    self.boxes.append({
                        'id': index, 'area': 'p3', 'num': nx_p3, 'num_F': self.F13 + t + 1, 'action': 0
                    })
                    if nx_p3 >= self.trapezoid[t]['N3']:
                        self.robot_offsets[-1]['action'] = 1
                        self.boxes[-1]['action'] = 1
                        break
                    for j in range(0, int((self.trapezoid[t]['N3'] - nx_p3) / self.grab_num_p3)):
                        index += 1
                        self.robot_offsets.append({'id': index, 'area': 'p3', 'num': self.grab_num_p3,
                                                   'num_F': self.F13 + t + 1, 'action': 0,
                                                   'pos': [0, (nx_p3 + j * self.grab_num_p3) * self.h,
                                                           i * self.w + self.trapezoid[t]['T1'] * self.h],
                                                   'size': [self.l, self.w, self.h]})
                        self.boxes.append({
                            'id': index, 'area': 'p3', 'num': self.grab_num_p3, 'num_F': self.F13 + t + 1, 'action': 0
                        })
                # 梯形尾料
                if self.trapezoid[t]['Nx'] != 0:
                    nx_end_1 = self.trapezoid[t]['N3'] % self.grab_num_p3 \
                        if self.trapezoid[t]['Nx'] % self.grab_num_p3 != 0 else self.grab_num_p3
                    index += 1
                    self.robot_offsets.append({
                        'id': index, 'area': 'p3', 'num': nx_end_1, 'num_F': self.F13 + t + 1, 'action': 0,
                        'pos': [0, 0, self.trapezoid[t]['T1']*self.h+self.trapezoid[t]['T3']*self.w],
                        'size': [self.l, self.w, self.h]})
                    self.boxes.append({
                        'id': index, 'area': 'p3', 'num': nx_end_1, 'num_F': self.F13 + t + 1, 'action': 0
                    })
                    if nx_end_1 < self.trapezoid[t]['Nx']:
                        for j in range(0, int((self.trapezoid[t]['Nx'] - nx_end_1) / self.grab_num_p3)):
                            index += 1
                            self.robot_offsets.append({
                                'id': index, 'area': 'p3', 'num': self.grab_num_p3, 'num_F': self.F13 + t + 1,
                                'action': 0,
                                'pos': [0,
                                        (nx_end_1 + j * self.grab_num_p3) * self.h,
                                        self.trapezoid[t]['T1'] * self.h + self.trapezoid[t]['T3'] * self.w],
                                'size': [self.l, self.w, self.h]})
                            self.boxes.append({
                                'id': index, 'area': 'p3', 'num': self.grab_num_p3, 'num_F': self.F13 + t + 1,
                                'action': 0
                            })
                self.robot_offsets[-1]['action'] = 1
                self.boxes[-1]['action'] = 1
        self.robot_offsets[-1]['action'] = 2
        self.boxes[-1]['action'] = 2
        self.robot_offsets.append('done')
        self.ori_offsets = self.robot_offsets.copy()
        # 测试用
        # for ofs in self.robot_offsets:
        #     if ofs['id'] <= 12:
        #         self.fit_offsets.append(ofs)

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


class BinEnv:

    def __init__(self, config_data):
        self.reserve_grip = config_data['reserve_grip']
        self.reserve_object = config_data['reserve_object']
        self.objects = []

    def reset(self):
        self.objects = []

    def step(self, action):
        boxs = []
        area = action['area']
        num = action['num']
        pos = action['pos']
        l = action['size'][0]
        w = action['size'][1]
        h = action['size'][2]
        temp_list = []
        if area == 'p1':
            for i in range(num):
                box = Box(l, w, h, rotation_type=0, position=[pos[0], pos[1] + i * w, pos[2]])
                boxs.append(box)
        if area == 'p2':
            for i in range(num):
                box = Box(l, w, h, rotation_type=1, position=[pos[0] + i * w, pos[1], pos[2]])
                boxs.append(box)
        if area == 'p3':
            for i in range(num):
                box = Box(l, w, h, rotation_type=2, position=[pos[0], pos[1] + i * h, pos[2]])
                boxs.append(box)
        if action['action'] == 0:
            if action['area'] != 'p2':
                for box in boxs:
                    self.objects.append(
                        (box.position[0], box.position[1], box.position[2],
                         box.position[0] + box.length + self.reserve_object[0],
                         box.position[1] + box.width + self.reserve_object[1],
                         box.position[2] + box.height + self.reserve_object[2]))
            else:
                for box in boxs:
                    self.objects.append(
                        (box.position[0], box.position[1] - self.reserve_object[1], box.position[2],
                         box.position[0] + box.length + self.reserve_object[0],
                         box.position[1] + box.width,
                         box.position[2] + box.height + self.reserve_object[2]))
        elif action['action'] == 1:
            if action['area'] != 'p2':
                for box in boxs:
                    self.objects.append(
                        (box.position[0], box.position[1], box.position[2],
                         box.position[0] + box.length + self.reserve_object[0],
                         box.position[1] + box.width + self.reserve_object[1],
                         box.position[2] + box.height + self.reserve_object[2]))
            else:
                for box in boxs:
                    self.objects.append(
                        (box.position[0], box.position[1] - self.reserve_object[1], box.position[2],
                         box.position[0] + box.length + self.reserve_object[0],
                         box.position[1] + box.width,
                         box.position[2] + box.height + self.reserve_object[2]))
            for ob in self.objects:
                if ob[3] > (l + self.reserve_object[0]):
                    temp_list.append(
                        (ob[0] - l if ob[0] - l > 0 else 0, ob[1], ob[2], ob[3] - l - self.reserve_object[0],
                         ob[4] - self.reserve_object[1], ob[5] - self.reserve_object[2]))
            self.objects.clear()
            self.objects = temp_list.copy()
        elif action['action'] == 2:
            self.objects.clear()

    @staticmethod
    def to_box(action):
        boxs = []
        area = action['area']
        num = action['num']
        pos = action['pos']
        l = action['size'][0]
        w = action['size'][1]
        h = action['size'][2]
        if area == 'p1':
            for i in range(num):
                box = Box(l, w, h, rotation_type=0, position=[pos[0], pos[1] + i * w, pos[2]])
                boxs.append(box)
        if area == 'p2':
            for i in range(num):
                box = Box(l, w, h, rotation_type=1, position=[pos[0] + i * w, pos[1], pos[2]])
                boxs.append(box)
        if area == 'p3':
            for i in range(num):
                box = Box(l, w, h, rotation_type=2, position=[pos[0], pos[1] + i * h, pos[2]])
                boxs.append(box)
        return boxs

    def render(self):
        pass


class Node:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None
