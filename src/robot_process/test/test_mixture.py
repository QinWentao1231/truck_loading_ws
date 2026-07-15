import copy
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
            'NA': 9,
            'TA': 2,
            'StackA': [
                [25, 0, 25, 0, 0, 50, 0, 0, 0, 0],
                [0, 0, 25, 0, 0, 50, 0, 0, 0, 25],
            ],
            'GroupA': [[2, 3, 4], [2, 3, 4]],
            'TypeA': '104',
            'NB': 8,
            'Tb': 2,
            'StackB': [
                [33, 0, 0, 0, 67, 0, 0, 0, 0],
                [0, 0, 0, 0, 67, 0, 0, 0, 33],
            ],
            'GroupB': [[4, 4], [4, 4]],
            'TypeB': '201',
        }],
    }


class MixturePlacementTest(unittest.TestCase):

    def test_two_box_types_share_one_p1_face(self):
        rp = RobotPosition(make_mixture_config())
        actions = [a for a in rp.ori_offsets if a != 'done']

        self.assertEqual(rp.block_type, 'mixture')
        self.assertEqual(rp.box_count, 34)
        self.assertEqual(sum(sum(a['num']) for a in actions), 34)
        self.assertEqual(len(actions), len(rp.boxes))
        self.assertEqual({a['num_F'] for a in actions}, {1})
        self.assertTrue(all(a['area'] == 'p1' for a in actions))

        a_actions = [a for a in actions if a['box_type'] == '104']
        b_actions = [a for a in actions if a['box_type'] == '201']
        self.assertEqual(sum(sum(a['num']) for a in a_actions), 18)
        self.assertEqual(sum(sum(a['num']) for a in b_actions), 16)
        self.assertTrue(all(a['size'] == [460, 250, 580] for a in a_actions))
        self.assertTrue(all(a['size'] == [525, 275, 300] for a in b_actions))
        self.assertEqual(min(a['pos'][2] for a in b_actions), 1160)

        self.assertTrue(all(b['box_type'] == a['box_type']
                            for a, b in zip(actions, rp.boxes)))
        self.assertTrue(all(b['num'] < 10 for b in rp.boxes
                            if b['box_type'] == '104'))
        self.assertTrue(all(b['num'] >= 10 for b in rp.boxes
                            if b['box_type'] == '201'))
        self.assertEqual(actions[-1]['action'], 2)
        self.assertEqual(rp.ori_offsets[-1], 'done')

        placed = []
        for action in actions:
            for box in BinEnv.to_box(action):
                current = (
                    box.position[1], box.position[1] + box.width,
                    box.position[2], box.position[2] + box.height,
                )
                self.assertGreaterEqual(current[0], 0)
                self.assertLessEqual(current[1], rp.W)
                self.assertGreaterEqual(current[2], 0)
                self.assertLessEqual(current[3], rp.H)
                for previous in placed:
                    overlap = (
                        current[0] < previous[1] - 0.5
                        and current[1] > previous[0] + 0.5
                        and current[2] < previous[3] - 0.5
                        and current[3] > previous[2] + 0.5
                    )
                    self.assertFalse(overlap)
                placed.append(current)

    def test_invalid_b_group_is_rejected(self):
        cfg = copy.deepcopy(make_mixture_config())
        cfg['mixture'][0]['GroupB'][0] = [3, 4]
        with self.assertRaisesRegex(ValueError, 'GroupB'):
            RobotPosition(cfg)


if __name__ == '__main__':
    unittest.main()
