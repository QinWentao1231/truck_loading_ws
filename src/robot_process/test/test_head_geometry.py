import os
import sys
import unittest


sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..', 'robot_process'))

from rrt_env.environment_3D import build_robot_positions


CAR = {
    'size': {'L': 9000, 'W': 2440, 'H': 2700},
    'reserve': {'L': 0, 'W': 0, 'H': 0},
    'head': {'L': 1500, 'W': 1800, 'H': 2700},
}


def _box(box_type, length, width):
    return {
        'box_list': [box_type],
        'box': {
            box_type: {
                'size': {'L': length, 'W': width, 'H': 300},
                # reserve.L不参与异形车头的实际纵深递推。
                'reserve': {'L': 100, 'W': 0, 'H': 0},
                'grip': {'P1': [4], 'P2': [2], 'P3': [2]},
            },
        },
        'car': CAR,
        'regular': [],
        'trapezoid': [],
        'mixture': [],
    }


def _regular_block(face_count=1, is_head=False):
    block = _box('101', 455, 200)
    block['regular'] = [{
        'N1': 8, 'N2': 0, 'N3': 0,
        'T12': 1, 'T3': 0,
        'F13': face_count, 'F2': 0,
        'E': 0, 'Nx': 0,
        'Stack': [[0] * 9, [0] * 9],
        'Group': [[4, 4], [4, 4]],
        'Type': '101',
        'Ishead': is_head,
    }]
    return block


def _mixture_block():
    block = _box('203', 520, 285)
    block['mixture'] = [{
        'Items': [{
            'Type': '203', 'Num': 4,
            'Pos': {'X': 0, 'Y': 0, 'Z': 0},
            'Ishead': False,
        }],
    }]
    return block


def _trapezoid_block():
    block = _box('101', 455, 200)
    block['trapezoid'] = [{
        'N1': 8, 'N3': 0, 'T1': 1, 'T3': 0, 'Nx': 0,
        'Stack': [], 'Group': [],
        'Isdoor': True, 'Type': '101', 'Ishead': False,
    }]
    return block


class HeadGeometryTest(unittest.TestCase):

    def test_width_uses_previous_face_physical_depth_across_blocks(self):
        blocks = [
            _regular_block(face_count=2, is_head=True),
            _mixture_block(),
            _trapezoid_block(),
            _regular_block(face_count=1, is_head=False),
        ]

        rp_list = build_robot_positions(blocks)

        # 面起点依次为0、455、910、1430、1885mm；reserve.L=100不参与。
        expected = [
            (0.0, 1800.0, True),
            (455.0, 1994.133333, True),
            (910.0, 2188.266667, True),
            (1430.0, 2410.133333, True),
            (1885.0, 2440.0, False),
        ]
        actual = []
        for rp in rp_list:
            for geometry in rp.head_face_geometry.values():
                actual.append((
                    geometry['depth_x'], geometry['car_width'],
                    geometry['is_head']))

        self.assertEqual(len(actual), len(expected))
        for observed, wanted in zip(actual, expected):
            self.assertAlmostEqual(observed[0], wanted[0], places=5)
            self.assertAlmostEqual(observed[1], wanted[1], places=5)
            self.assertEqual(observed[2], wanted[2])

        # 动作和来料队列携带同一面级宽度，供路径、校验和垛面检测使用。
        first_action = rp_list[1].ori_offsets[0]
        first_box = rp_list[1].boxes[0]
        self.assertAlmostEqual(first_action['car_width'], expected[2][1], places=5)
        self.assertAlmostEqual(first_box['car_width'], expected[2][1], places=5)
        self.assertEqual([rp.is_head for rp in rp_list], [True, True, True, False])

    def test_explicit_ishead_requires_head_dimensions(self):
        block = _regular_block(is_head=True)
        block['car'] = {
            'size': dict(CAR['size']),
            'reserve': dict(CAR['reserve']),
        }
        with self.assertRaisesRegex(ValueError, '异形车头尺寸无效'):
            build_robot_positions([block])

    def test_normal_order_keeps_original_width(self):
        block = _regular_block(face_count=2, is_head=False)
        block['car'] = {
            'size': dict(CAR['size']),
            'reserve': dict(CAR['reserve']),
        }
        rp = build_robot_positions([block])[0]
        widths = {
            action['car_width']
            for action in rp.ori_offsets if action != 'done'
        }
        self.assertEqual(widths, {2440.0})
        self.assertFalse(rp.is_head)


if __name__ == '__main__':
    unittest.main()
