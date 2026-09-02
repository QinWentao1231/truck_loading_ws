import os
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robot_process'))

from rrt_env.environment_3D import BinEnv, RobotPosition


def make_mixture_config():
    return {
        'box_list': ['104', '201'],
        'box': {
            '104': {
                'size': {'L': 460, 'W': 250, 'H': 580},
                'reserve': {},
                'grip': {'P1': [4], 'P2': [2], 'P3': [2]},
            },
            '201': {
                'size': {'L': 525, 'W': 275, 'H': 292},
                'reserve': {'H': 8},
                'grip': {'P1': [4], 'P2': [2], 'P3': [2]},
            },
        },
        'car': {
            'size': {'L': 9000, 'W': 2460, 'H': 2600},
            'reserve': {'W': 100},
        },
        'regular': [],
        'trapezoid': [],
        'mixture': [{
            'Items': [
                {
                    'Type': '104',
                    'Num': 4,
                    'Pos': {'X': 20, 'Y': 0, 'Z': 0},
                },
                {
                    'Type': '201',
                    'Num': 4,
                    'Pos': {'X': 30, 'Y': 1360, 'Z': 580},
                },
            ],
        }],
    }


def make_complex_mixture_config():
    """现场混装测试面；105 高度按 580 mm。"""
    positions = [
        ('105', 2, 0, 0),
        ('105', 3, 515, 0),
        ('105', 4, 1280, 0),
        ('105', 4, 0, 580),
        ('203', 4, 1177, 580),
        ('203', 4, 1177, 873),
        ('203', 4, 0, 1160),
        ('203', 4, 1177, 1166),
        ('203', 4, 0, 1453),
        ('203', 4, 1177, 1465),
    ]
    return {
        'box_list': ['105', '203'],
        'box': {
            '105': {
                'size': {'L': 455, 'W': 245, 'H': 580},
                'reserve': {},
                'grip': {'P1': [4], 'P2': [3], 'P3': [2]},
            },
            '203': {
                'size': {'L': 520, 'W': 285, 'H': 285},
                'reserve': {},
                'grip': {'P1': [4], 'P2': [3], 'P3': [2]},
            },
        },
        'car': {
            'size': {'L': 6100, 'W': 2900, 'H': 2900},
            'reserve': {'L': 50, 'W': 80, 'H': 165},
        },
        'regular': [],
        'trapezoid': [],
        'mixture': [{
            'Items': [
                {
                    'Type': box_type,
                    'Num': num,
                    'Pos': {'X': 0, 'Y': y, 'Z': z},
                }
                for box_type, num, y, z in positions
            ],
        }],
    }


def make_bin_env():
    return BinEnv({
        'reserve_grip': [0, 0, 50],
        'reserve_object': [49, 49, 49],
    })


def make_three_grab_regular_config(box_type='101', block_type='regular'):
    """构造一层三抓[3,3,4]垛型，用于验证最右抓数量编码。"""
    base = {
        'box_list': [box_type],
        'box': {
            box_type: {
                'size': {'L': 455, 'W': 245, 'H': 580},
                'reserve': {},
                'grip': {'P1': [4], 'P2': [2], 'P3': [2]},
            },
        },
        'car': {
            'size': {'L': 9000, 'W': 2900, 'H': 2800},
            'reserve': {'L': 0, 'W': 0, 'H': 0},
        },
        'regular': [],
        'trapezoid': [],
        'mixture': [],
    }
    stack = [[0] * 11, [0] * 11]
    group = [[3, 3, 4], [3, 3, 4]]
    if block_type == 'regular':
        base['regular'] = [{
            'N1': 10, 'N2': 0, 'N3': 0,
            'T12': 1, 'T3': 0, 'F13': 1, 'F2': 0,
            'E': 0, 'Nx': 0, 'Stack': stack, 'Group': group,
            'Type': box_type, 'Ishead': False,
        }]
    else:
        base['trapezoid'] = [{
            'N1': 10, 'N3': 0, 'T1': 1, 'T3': 0, 'Nx': 0,
            'Stack': stack, 'Group': group,
            'Isdoor': False, 'Type': box_type, 'Ishead': False,
        }]
    return base


class MixturePlacementTest(unittest.TestCase):

    def test_regular_three_grab_rightmost_uses_plus_twenty_signal(self):
        rp = RobotPosition(make_three_grab_regular_config('101'))

        # 三抓执行顺序为左、右、中；右抓实际4箱，因此发送24。
        self.assertEqual([box['num'] for box in rp.boxes], [3, 24, 3])
        self.assertEqual(
            [action['is_p1_three_grab_right_aligned']
             for action in rp.ori_offsets if action != 'done'],
            [False, True, False],
        )

    def test_non_1xx_three_grab_keeps_original_plus_ten_signal(self):
        rp = RobotPosition(make_three_grab_regular_config('203'))

        self.assertEqual([box['num'] for box in rp.boxes], [13, 14, 13])

    def test_trapezoid_1xx_three_grab_uses_plus_twenty_signal(self):
        rp = RobotPosition(make_three_grab_regular_config(
            '101', block_type='trapezoid'))

        self.assertEqual([box['num'] for box in rp.boxes], [3, 24, 3])

    def test_door_trapezoid_1xx_p1_three_grab_uses_plus_twenty_signal(self):
        cfg = make_three_grab_regular_config(
            '101', block_type='trapezoid')
        cfg['trapezoid'][0]['Isdoor'] = True
        rp = RobotPosition(cfg)

        # 门口梯形走简单分抓[2,4,4]，物理最右抓仍使用右对齐编码。
        self.assertEqual([box['num'] for box in rp.boxes], [2, 4, 24])

    def test_1xx_p3_three_grab_does_not_use_plus_twenty_signal(self):
        cfg = make_three_grab_regular_config('101')
        regular = cfg['regular'][0]
        regular.update({'N1': 0, 'T12': 0, 'N3': 6, 'T3': 1})
        rp = RobotPosition(cfg)

        # 1XX 的 P3 原规则为实际数量+10，不得套用 P1 的+20编码。
        self.assertEqual([box['num'] for box in rp.boxes], [12, 12, 12])
        self.assertFalse(any(
            action['is_p1_three_grab_right_aligned']
            for action in rp.ori_offsets if action != 'done'
        ))

    def test_mixture_1xx_three_grab_does_not_use_plus_twenty_signal(self):
        cfg = make_mixture_config()
        cfg['box_list'] = ['104']
        cfg['box'] = {'104': cfg['box']['104']}
        cfg['mixture'][0]['Items'] = [
            {
                'Type': '104', 'Num': 3,
                'Pos': {'X': 20, 'Y': y, 'Z': 0},
            }
            for y in (0, 750, 1500)
        ]
        rp = RobotPosition(cfg)

        self.assertEqual([box['num'] for box in rp.boxes], [3, 3, 3])
        self.assertFalse(any(
            action['is_p1_three_grab_right_aligned']
            for action in rp.ori_offsets if action != 'done'
        ))

    def test_position_items_are_parsed_as_direct_placements(self):
        rp = RobotPosition(make_mixture_config())
        actions = [a for a in rp.ori_offsets if a != 'done']

        self.assertEqual(rp.block_type, 'mixture')
        self.assertFalse(rp.is_head)
        self.assertEqual(rp.frame, {'L': 0.0, 'W': 0.0, 'H': 0.0})
        self.assertEqual(rp.head, {'L': 0.0, 'W': 0.0, 'H': 0.0})
        self.assertEqual(rp.box_count, 8)
        self.assertEqual(len(actions), 2)
        self.assertEqual([a['box_type'] for a in actions], ['104', '201'])
        self.assertEqual([a['num'] for a in actions], [[4], [4]])
        self.assertEqual(actions[0]['pos'], [20.0, 0.0, 0.0])
        self.assertEqual(actions[1]['pos'], [30.0, 1360.0, 580.0])
        self.assertEqual([a['dir'] for a in actions], [1, 2])
        self.assertTrue(all(a['area'] == 'p1' for a in actions))
        self.assertEqual([b['box_type'] for b in rp.boxes], ['104', '201'])
        self.assertEqual([b['num'] for b in rp.boxes], [4, 14])
        self.assertEqual(
            [b['size'] for b in rp.boxes],
            [[460, 250, 580], [525, 275, 300]],
        )

    def test_ishead_marks_the_whole_mixture_block(self):
        cfg = make_mixture_config()
        cfg['car']['head'] = {'L': 1200, 'W': 2460, 'H': 900}
        cfg['mixture'][0]['Items'][1]['Ishead'] = True

        rp = RobotPosition(cfg)

        self.assertTrue(rp.is_head)
        self.assertEqual(rp.head, {'L': 1200.0, 'W': 2460.0, 'H': 900.0})

    def test_tail_frame_dimensions_are_preserved(self):
        cfg = make_mixture_config()
        cfg['car']['frame'] = {'L': 100, 'W': 2380, 'H': 2550}

        rp = RobotPosition(cfg)

        self.assertEqual(
            rp.frame, {'L': 100.0, 'W': 2380.0, 'H': 2550.0})

    def test_position_item_rejects_num_above_grip_capacity(self):
        cfg = make_mixture_config()
        cfg['mixture'][0]['Items'][0]['Num'] = 5
        with self.assertRaisesRegex(ValueError, 'P1 单抓能力'):
            RobotPosition(cfg)

    def test_mixture_wall_side_finishing_grab_uses_area_cfg_four(self):
        cfg = make_mixture_config()
        # 两抓位于同一高度：第二抓靠近右侧边界，且左侧已有高于当前底面
        # 50 mm 的箱体，因此它是当前高度的墙边收尾抓。
        cfg['mixture'][0]['Items'][1]['Pos']['Z'] = 0
        rp = RobotPosition(cfg)
        self.assertEqual([box['area_cfg'] for box in rp.boxes], [1, 4])

    def test_empty_items_are_rejected(self):
        cfg = make_mixture_config()
        cfg['mixture'][0]['Items'] = []
        with self.assertRaisesRegex(ValueError, 'Items 为空'):
            RobotPosition(cfg)

    def test_sixth_grab_detects_fourth_grab_by_z_volume_overlap(self):
        rp = RobotPosition(make_complex_mixture_config())
        actions = [a for a in rp.ori_offsets if a != 'done']
        env = make_bin_env()
        for action in actions[:5]:
            env.step(action)

        clearance = env.mixture_side_clearance(
            actions[5], left_wall=0, right_wall=rp.W,
            min_z_overlap=20)

        self.assertEqual(clearance['relevant_count'], 4)
        self.assertAlmostEqual(clearance['left_edge'], 980)
        self.assertAlmostEqual(clearance['left_gap'], 197)
        self.assertAlmostEqual(clearance['right_gap'], 583)
        self.assertFalse(clearance['blocking'])
        self.assertEqual(1 if clearance['left_gap'] <= clearance['right_gap'] else 2, 1)

    def test_eighth_grab_right_app_avoids_left_stack(self):
        rp = RobotPosition(make_complex_mixture_config())
        actions = [a for a in rp.ori_offsets if a != 'done']
        env = make_bin_env()
        for action in actions[:7]:
            env.step(action)

        action = actions[7]
        clearance = env.mixture_side_clearance(
            action, left_wall=0, right_wall=rp.W,
            min_z_overlap=20)
        self.assertAlmostEqual(clearance['left_gap'], 37)
        self.assertAlmostEqual(clearance['right_gap'], 583)

        size = (520, 1140, 335)
        unsafe_path = [
            (650, 1077, 1715),
            (650, 1077, 1715),
            (50, 1077, 1216),
            (0, 1177, 1166),
        ]
        safe_path = [
            (650, 1277, 1715),
            (650, 1277, 1715),
            (50, 1277, 1216),
            (0, 1177, 1166),
        ]
        unsafe, detail = env.trajectory_collision_free(
            unsafe_path, size, sample_step=10)
        safe, _ = env.trajectory_collision_free(
            safe_path, size, sample_step=10)

        self.assertFalse(unsafe)
        self.assertIsNotNone(detail)
        self.assertTrue(safe)


if __name__ == '__main__':
    unittest.main()
