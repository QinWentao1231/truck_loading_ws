import os
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robot_process'))

from rrt_env.environment_3D import RobotPosition


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


class MixturePlacementTest(unittest.TestCase):

    def test_position_items_are_parsed_as_direct_placements(self):
        rp = RobotPosition(make_mixture_config())
        actions = [a for a in rp.ori_offsets if a != 'done']

        self.assertEqual(rp.block_type, 'mixture')
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

    def test_position_item_rejects_num_above_grip_capacity(self):
        cfg = make_mixture_config()
        cfg['mixture'][0]['Items'][0]['Num'] = 5
        with self.assertRaisesRegex(ValueError, 'P1 单抓能力'):
            RobotPosition(cfg)

    def test_all_mixture_area_cfg_are_fixed_to_one(self):
        cfg = make_mixture_config()
        # 两抓放在同一层时，通用左右排序原本会得到 1/3；混装面必须全部固定为 1。
        cfg['mixture'][0]['Items'][1]['Pos']['Z'] = 0
        rp = RobotPosition(cfg)
        self.assertEqual([box['area_cfg'] for box in rp.boxes], [1, 1])

    def test_empty_items_are_rejected(self):
        cfg = make_mixture_config()
        cfg['mixture'][0]['Items'] = []
        with self.assertRaisesRegex(ValueError, 'Items 为空'):
            RobotPosition(cfg)


if __name__ == '__main__':
    unittest.main()
