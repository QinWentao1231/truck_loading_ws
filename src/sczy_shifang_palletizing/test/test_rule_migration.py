"""离线回归：不启动ROS、gRPC服务、机器人连接或可视化窗口。"""

import ast
import copy
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
SCRIPTS = PACKAGE / 'sczy_shifang_palletizing'
sys.path.insert(0, str(PACKAGE))

from sczy_shifang_palletizing import runtime as main
from sczy_shifang_palletizing.grpc_pkg.interface import issuer_pb2
from sczy_shifang_palletizing.rule_planning.environment import BinEnv
from sczy_shifang_palletizing.rule_planning.legacy_order import RobotPosition, parse_legacy_request
from sczy_shifang_palletizing.rule_planning.legacy_protocol import (
    box_message, pallet_message, path_message, per_count_message,
)
from sczy_shifang_palletizing.rule_planning.path_planner import (
    calculate_path, row_clearance, record_sent_placement,
)
from sczy_shifang_palletizing.socket_pkg.socket_server import SocketApp


def config():
    with (PACKAGE / 'config/planner.json').open() as stream:
        result = json.load(stream)
    result.update(show_env=False, use_corner=False)
    return result


def order():
    stack = [0, 0, 50, 0, 0, 50, 0, 0, 0, 0]
    return {
        'box_list': ['A'],
        'box': {'A': {'size': {'L': 460, 'W': 250, 'H': 580}}},
        'car': {'size': {'L': 10000, 'W': 2460, 'H': 2800},
                'reserve': {'L': 0, 'W': 80, 'H': 100}},
        'regular': [dict(N1=9, N2=0, N3=0, T12=2, T3=0,
                         F13=2, F2=0, E=0, Nx=0,
                         Stack=[stack, stack.copy()], Group=[[2, 3, 4], [2, 3, 4]])],
        'trapezoid': [],
    }


def trap(door=False, tail=0):
    src = order()['regular'][0]
    return dict(N1=9, N3=0, T1=1, T3=0, Nx=tail, Isdoor=door,
                Stack=src['Stack'], Group=src['Group'])


def request_from_order(data):
    request = issuer_pb2.IssueRequest()
    box = request.condition.boxes.add(type='A')
    for key, value in data['box']['A']['size'].items():
        setattr(box.original, key, value)
    for key, value in data['car']['size'].items():
        setattr(request.condition.car.original, key, value)
    for key, value in data['car']['reserve'].items():
        setattr(request.condition.car.reserve, key, value)
    for source in data['regular']:
        entry = request.answer.regular.add()
        for key, value in source.items():
            if key == 'Group':
                for values in value:
                    entry.Group.add().Unit.extend(values)
            elif key == 'Stack':
                for values in value:
                    entry.Stack.add().Width.extend(values)
            else:
                setattr(entry, key, value)
    for source in data['trapezoid']:
        entry = request.answer.trapezoid.add()
        for key, value in source.items():
            if key == 'Group':
                for values in value:
                    entry.Group.add().Unit.extend(values)
            elif key == 'Stack':
                for values in value:
                    entry.Stack.add().Width.extend(values)
            else:
                setattr(entry, key, value)
    return request


def decode(message):
    assert message[:7] == b'\xfe\xfe\x01\x00\x10\x23\x01'
    assert message[8:14] == b'\x00\x00\x00\x00\x00\x91'
    count = message[7]
    assert len(message) == 16 + 41 * count
    assert message[-2:] == b'\x00\x00'
    return [struct.unpack('!BI9f', message[14 + i*41:55 + i*41])
            for i in range(count)]


class FakeServer:
    def __init__(self, commands, fail_send=False):
        self.commands = iter(commands)
        self.messages = []
        self.fail_send = fail_send

    def receive_message(self, byte_size):
        assert byte_size == 16
        return next(self.commands, '')

    def send_message(self, message):
        if self.fail_send:
            raise ConnectionError('test send failure')
        self.messages.append(message)


class RuleMigrationTests(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger('migration-tests')
        self.logger.handlers = [logging.NullHandler()]
        self.logger.propagate = False

    def test_protobuf_files_unchanged(self):
        hashes = {
            'proto/all.proto': '8b7e4cfac53c9c0cd4710a4df8076d411ac311625e194d57f2226d6cab763072',
            'proto/answer.proto': '68e5cb52b8a4fdd1a9681eeaef9c6ca3e4d2d596a8f54e0497af0911a414ad8e',
            'proto/condition.proto': 'b315d0335a335d474b5a6079f2993ed8b97a1c81494d0b255147d637586bc038',
            'proto/issuer.proto': '296141dd826f30faeca59df2f5258fd70af42ea5e9916b8fdce154fdbfb30263',
            'answer_pb2.py': 'dd438f6b9f8c21ff3a74679b9fda1b33717db772b94c795bc4d6e9d023268b95',
            'condition_pb2.py': 'eba52eb29ddf27cbd48b6294e7f8132734f393874cbb70407302f648a91ca280',
            'issuer_pb2.py': 'e36f9fe15e2df9a40cddeda98ea88dd288d51121b93af4682295a72e8974be36',
            'answer_pb2_grpc.py': 'd686e804f171693117b7d030ec4023f205c70c234c8590f6557aa8702f65fe09',
            'condition_pb2_grpc.py': 'd686e804f171693117b7d030ec4023f205c70c234c8590f6557aa8702f65fe09',
            'issuer_pb2_grpc.py': 'f913dca9f5bb7812c923001cfbf8aef54f5b6644767be7e5d004b8d21bae3b79',
        }
        for relative, expected in hashes.items():
            with self.subTest(file=relative):
                content = (SCRIPTS / 'grpc_pkg/interface' / relative).read_bytes()
                # ROS2包命名空间改变了生成代码的import路径，描述符和proto不得改变。
                content = content.replace(b'sczy_shifang_palletizing.grpc_pkg', b'grpc_pkg')
                actual = hashlib.sha256(content).hexdigest()
                self.assertEqual(actual, expected)

    def test_same_core_as_robot_process(self):
        source = PACKAGE.parent / 'robot_process/robot_process/rrt_env/environment_3D.py'
        if not source.exists():
            self.skipTest('独立部署时无需robot_process源码')
        self.assertEqual(ast.dump(ast.parse(source.read_text())), ast.dump(ast.parse(
            (SCRIPTS / 'rule_planning/environment.py').read_text())))
        node = PACKAGE.parent / 'robot_process/robot_process/robot_process_node.py'
        original = {n.name: n for n in ast.parse(node.read_text()).body
                    if isinstance(n, ast.FunctionDef)}
        for helper in ast.parse((SCRIPTS / 'rule_planning/path_helpers.py').read_text()).body:
            if isinstance(helper, ast.FunctionDef):
                self.assertEqual(ast.dump(helper), ast.dump(original[helper.name]), helper.name)

    def test_old_proto_ignores_reserve_grip_frame_like_original(self):
        request = request_from_order(order())
        request.condition.boxes[0].reserve.H = 5
        request.condition.boxes[0].grip.P1.extend([3, 8])
        request.condition.car.frame.W = 2400
        parsed = parse_legacy_request(issuer_pb2.IssueRequest.FromString(request.SerializeToString()))
        rp = RobotPosition(parsed)
        self.assertEqual(rp.h, 580)
        self.assertEqual(rp.grab_num_p1, 4)
        self.assertNotIn('frame', parsed['car'])
        self.assertEqual(rp.box_count, 36)

    def test_three_grab_order_and_face_flags(self):
        rp = RobotPosition(order())
        self.assertEqual([sum(a['num']) for a in rp.robot_offsets[:6]], [2, 3, 4, 4, 3, 2])
        self.assertEqual([a['dir'] for a in rp.robot_offsets[:6]], [1, 1, 1, 2, 2, 2])
        self.assertEqual(rp.robot_offsets[5]['action'], 1)
        self.assertEqual(rp.robot_offsets[-2]['action'], 2)
        self.assertEqual(rp.face_count, 2)

    def test_two_grab_alternation(self):
        data = order()
        data['regular'][0].update(N1=8, Group=[[4, 4], [4, 4]],
                                 Stack=[[0, 0, 0, 0, 100, 0, 0, 0, 0]] * 2)
        a = RobotPosition(data).robot_offsets
        self.assertLess(a[0]['pos'][1], a[1]['pos'][1])
        self.assertGreater(a[2]['pos'][1], a[3]['pos'][1])
        self.assertGreater(a[4]['pos'][1], a[5]['pos'][1])

    def test_merge_regular_and_trapezoid(self):
        data = order()
        data['trapezoid'] = [trap(), trap(door=True)]
        rp = RobotPosition(data)
        actions = rp.robot_offsets[:-1]
        self.assertEqual(sorted({a['num_F'] for a in actions}), [1, 2, 3, 4])
        self.assertEqual([a['id'] for a in actions], list(range(1, len(actions) + 1)))
        self.assertEqual(sum(a['action'] == 2 for a in actions), 1)
        self.assertEqual(sum(a['action'] == 1 for a in actions), 3)
        self.assertEqual(rp.box_count, 54)

    def test_original_trapezoid_tail_is_p3(self):
        data = order()
        data['regular'][0].update(F13=0)
        data['trapezoid'] = [trap(tail=3)]
        rp = RobotPosition(data)
        self.assertEqual([a['area'] for a in rp.robot_offsets[:-1]], ['p1'] * 3 + ['p3'] * 2)
        self.assertEqual(rp.robot_offsets[-3]['num'], [1])
        self.assertEqual(rp.robot_offsets[-2]['num'], [2])
        self.assertEqual(rp.box_count, 12)

    def test_door_restores_fixed_grip_four_and_no_centering(self):
        data = order()
        data['regular'][0].update(F13=0)
        data['trapezoid'] = [trap(door=True)]
        data['box']['A']['grip'] = {'P1': [3]}
        rp = RobotPosition(data)
        self.assertEqual([a['num'] for a in rp.robot_offsets[:-1]], [[2], [3], [4]])
        self.assertEqual([a['pos'][1] for a in rp.robot_offsets[:-1]], [0, 500, 1250])

    def test_original_p2_omission_is_reported_not_silently_sent(self):
        data = order()
        data['regular'][0].update(N1=6, N2=1, F2=1, T12=1, F13=1,
                                 Group=[[3, 3]] * 2,
                                 Stack=[[0, 0, 0, 100, 0, 0, 0]] * 2)
        with self.assertRaisesRegex(ValueError, '原版垛序箱数不一致'):
            RobotPosition(data)

    def test_all_tail_modes(self):
        for mode, area in ((1, 'p1'), (2, 'p3')):
            with self.subTest(mode=mode):
                data = order()
                data['regular'][0].update(F13=1, E=mode, Nx=3)
                rp = RobotPosition(data)
                self.assertEqual(rp.box_count, 21)
                self.assertEqual(rp.face_count, 2 if mode == 1 else 1)
                self.assertEqual(rp.robot_offsets[-2]['area'], area)
                self.assertEqual(sum(a['action'] == 2 for a in rp.robot_offsets[:-1]), 1)

    def test_only_first_regular_entry_like_original(self):
        data = order()
        data['regular'][0].update(F13=1)
        data['regular'].append(copy.deepcopy(data['regular'][0]))
        rp = RobotPosition(data)
        self.assertEqual(rp.face_count, 1)
        self.assertEqual(rp.box_count, 18)
        self.assertEqual(rp.robot_offsets[5]['action'], 2)

    def test_p3_position_and_old_recipe(self):
        data = order()
        data['regular'][0].update(N3=3, T3=1, F13=1)
        rp = RobotPosition(data)
        p3 = [(a, b) for a, b in zip(rp.robot_offsets, rp.boxes) if a['area'] == 'p3']
        self.assertEqual([a['pos'][1] for a, _ in p3], [50, 630])
        self.assertEqual([decode(box_message(b, 'A'))[0][2] for _, b in p3], [5, 6])

    def test_internal_gap_keeps_original_single_placement(self):
        data = order()
        data['regular'][0]['Stack'][0][6] = 10
        rp = RobotPosition(data)
        self.assertEqual(rp.robot_offsets[2]['num'], [4])
        self.assertEqual(rp.robot_offsets[2]['gaps'], [])
        with patch.object(main, 'logs', self.logger):
            ok, message = main.callback(request_from_order(data))
        self.assertTrue(ok, message)

    def test_invalid_or_unrepresentable_orders(self):
        cases = []
        empty = order(); empty['regular'] = []; cases.append(empty)
        numeric = order(); numeric['box_list'] = ['105']; cases.append(numeric)
        no_e = order(); del no_e['regular'][0]['E']; cases.append(no_e)
        incomplete = order(); incomplete['regular'][0]['Group'] = [[2, 3, 4]]; cases.append(incomplete)
        dropped = order(); dropped['regular'][0].update(E=3, Nx=3); cases.append(dropped)
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                RobotPosition(data)
        request = request_from_order(order())
        request.answer.mixture.add(Na=4)
        with self.assertRaisesRegex(ValueError, '混装面'):
            parse_legacy_request(request)

    def test_all_actions_exactly_match_original_sequence(self):
        original_file = PACKAGE / 'scripts/rrt_env/environment_3D.py'
        spec = importlib.util.spec_from_file_location('shifang_original_for_test', original_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cases = [order()]
        for mode in (1, 2):
            data = order()
            data['regular'][0].update(E=mode, Nx=5)
            cases.append(data)
        for door in (False, True):
            for nx in (0, 3):
                data = order()
                data['regular'][0].update(N3=3, T3=1)
                data['trapezoid'] = [trap(door=door, tail=nx)]
                cases.append(data)
        for i, data in enumerate(cases):
            with self.subTest(case=i):
                expected = module.RobotPosition(copy.deepcopy(data))
                actual = RobotPosition(data)
                reconstructed = []
                for action in actual.robot_offsets[:-1]:
                    row = {key: copy.deepcopy(action[key]) for key in
                           ('id', 'area', 'num', 'num_F', 'action', 'dir', 'pos', 'size')}
                    row['num'] = sum(row['num'])
                    reconstructed.append(row)
                self.assertEqual(reconstructed, expected.robot_offsets[:-1])
                self.assertEqual(actual.box_count, expected.box_count)
                for actual_box, expected_box in zip(actual.boxes, expected.boxes):
                    self.assertEqual({k: actual_box[k] for k in expected_box}, expected_box)

    def test_protocol_layout_and_flags(self):
        rp = RobotPosition(order())
        self.assertEqual(decode(pallet_message(rp))[0][2:5], (36, 2, 2460))
        self.assertEqual(decode(per_count_message(rp))[0][2:5], (6, 65, 0))
        self.assertEqual(decode(box_message(rp.boxes[0], 'A'))[0][2:5], (2, 65, 0))
        for area, region in [('p1', 1), ('p2', 2), ('p3', 3)]:
            for flag in (0, 1, 2):
                blocks = decode(path_message([(1, 2, 3)] * 4, {'area': area, 'action': flag}))
                self.assertEqual([b[1] for b in blocks], [region, region, region, flag + 1])
                self.assertTrue(all(b[5:] == (0,) * 6 for b in blocks))

    def test_app_centering_capped_and_corner_only_changes_goal(self):
        rp = RobotPosition(order())
        cfg = config()
        cfg['use_corner'] = True
        environment = BinEnv(cfg)
        a = rp.robot_offsets[3]  # 第二层最右抓，原版dir=2
        plain = calculate_path(a, rp, cfg, environment)
        compensated = calculate_path(a, rp, cfg, environment, rp.W + 80)
        self.assertEqual(plain.y_offset_app, -100)
        self.assertEqual(plain.path[:3], compensated.path[:3])
        self.assertEqual(compensated.path[-1][1] - plain.path[-1][1], 40)
        middle = rp.robot_offsets[1]
        left, right, _ = row_clearance(middle, rp)
        result = calculate_path(middle, rp, cfg, environment, rp.W + 80)
        self.assertEqual(result.y_offset_app, max(-100, min(100, (right - left) / 2)))
        self.assertEqual(result.path[-1], tuple(middle['pos']))

    def test_paths_match_source_functions_across_heights(self):
        source = PACKAGE.parent / 'robot_process/robot_process/robot_process_node.py'
        if not source.exists():
            self.skipTest('独立部署时无需robot_process源码')
        import numpy as np
        names = {'_action_car_width', '_action_p1_right_wall', '_action_right_edge_y',
                 '_is_row_rightmost_grab', '_corner_compensation_applies',
                 '_build_p1_p3_approach', '_candidate_goal'}
        functions = [n for n in ast.parse(source.read_text()).body
                     if isinstance(n, ast.FunctionDef) and n.name in names]
        namespace = {'np': np}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
        rp, cfg = RobotPosition(order()), config()
        for area in ('p1', 'p3'):
            for z in (0, 580, 1160, 1820, 1821, 2150, 2151):
                for direction in (1, 2):
                    action = copy.deepcopy(rp.robot_offsets[1])
                    action.update(area=area, dir=direction)
                    action['pos'][2] = z
                    result = calculate_path(action, rp, cfg, BinEnv(cfg))
                    expected = namespace['_build_p1_p3_approach'](
                        action, rp, cfg, cfg['reserve_grip'], result.y_offset_app)
                    self.assertEqual(result.path[:3], [tuple(p) for p in expected[:3]])
                    self.assertEqual(result.size, expected[3])

    def test_sent_goal_updates_both_obstacle_models_and_row_clearance(self):
        for enabled, delta, expected in ((True, 30, 30), (True, 80, 40),
                                         (True, -30, 0), (True, 0, 0),
                                         (False, 30, 0)):
            with self.subTest(enabled=enabled, delta=delta):
                rp, cfg = RobotPosition(order()), config()
                cfg['use_corner'] = enabled
                originals = copy.deepcopy(rp.ori_offsets)
                boxes = copy.deepcopy(rp.boxes)
                target, following = originals[3:5]
                original_gaps = row_clearance(following, rp)
                env = BinEnv(cfg)
                server = FakeServer([main.cmd_get_per_count] + [main.cmd_get_path] * 4)
                with patch.object(main, 'BinEnv', return_value=env):
                    main.serve_commands(server, rp, cfg, self.logger,
                                        corner_reader=lambda *args: rp.W + delta)
                sent = decode(server.messages[-1])[-1][2:5]
                self.assertEqual(sent, rp.sent_positions[target['id']])
                self.assertEqual(sent[1], target['pos'][1] + expected)
                first_box = -sum(target['num'])
                self.assertEqual(env.display_objects[first_box][1], sent[1])
                self.assertEqual(env.objects[first_box][1],
                                 sent[1] - cfg['reserve_object'][1])
                gaps = row_clearance(following, rp)
                self.assertEqual(gaps[0], original_gaps[0])
                self.assertEqual(gaps[1], original_gaps[1] + expected)
                self.assertEqual(rp.ori_offsets, originals)
                self.assertEqual(rp.boxes, boxes)
                self.assertEqual(rp.robot_offsets, originals[4:])

    def test_failed_path_send_does_not_record_compensated_position(self):
        rp, cfg = RobotPosition(order()), config()
        cfg['use_corner'] = True
        rp.robot_offsets = rp.robot_offsets[3:]
        original = copy.deepcopy(rp.robot_offsets)
        env = BinEnv(cfg)
        class FailPathServer(FakeServer):
            def send_message(self, message):
                if self.messages:  # 面信息成功，路径发送失败。
                    raise ConnectionError('path send failed')
                super().send_message(message)
        server = FailPathServer([main.cmd_get_per_count, main.cmd_get_path])
        with patch.object(main, 'BinEnv', return_value=env):
            with self.assertRaises(ConnectionError):
                main.serve_commands(server, rp, cfg, self.logger,
                                    corner_reader=lambda *args: rp.W + 30)
        self.assertEqual(rp.sent_positions, {})
        self.assertEqual(env.objects, [])
        self.assertEqual(env.display_objects, [])
        self.assertEqual(rp.robot_offsets, original)

    def test_following_app_uses_compensated_neighbor(self):
        data, cfg = order(), config()
        cfg['use_corner'] = True
        data['car']['size']['W'] = 2200
        data['regular'][0].update(
            N1=8, F13=1, Group=[[4, 4], [4, 4]],
            Stack=[[25, 0, 0, 0, 50, 0, 0, 0, 25]] * 2)
        rp = RobotPosition(data)
        # 第二层先右后左；左抓理论左右间隙50/100，原APP偏移25。
        following = copy.deepcopy(rp.ori_offsets[3])
        self.assertEqual(row_clearance(following, rp)[:2], (50, 100))
        server = FakeServer([main.cmd_get_per_count] + [main.cmd_get_path] * 4)
        main.serve_commands(server, rp, cfg, self.logger,
                            corner_reader=lambda *args: rp.W + 30)
        # 右抓补偿后间隙变50/130，下一抓APP偏移应为40，而非25。
        self.assertEqual(decode(server.messages[-1])[2][3], following['pos'][1] + 40)
        self.assertEqual(decode(server.messages[-1])[-1][2:5], tuple(following['pos']))
        self.assertEqual(rp.sent_positions, {})  # 最后一抓结束整单。

    def test_validation_failure_does_not_record_position(self):
        data, cfg = order(), config()
        data['car']['size']['H'] = 1500
        rp, env = RobotPosition(data), BinEnv(cfg)
        with patch.object(main, 'BinEnv', return_value=env):
            with self.assertRaises(main.PathValidationError):
                main.serve_commands(FakeServer([main.cmd_get_path]), rp, cfg, self.logger)
        self.assertEqual(rp.sent_positions, {})
        self.assertEqual(env.display_objects, [])

    def test_face_end_clears_sent_positions_without_changing_originals(self):
        for terminal in (1, 2):
            with self.subTest(terminal=terminal):
                rp, cfg = RobotPosition(order()), config()
                cfg['use_corner'] = True
                env = BinEnv(cfg)
                action = copy.deepcopy(rp.ori_offsets[3])
                result = calculate_path(action, rp, cfg, env, rp.W + 30)
                record_sent_placement(action, result, rp, env)
                self.assertTrue(rp.sent_positions)
                self.assertTrue(env.objects)
                # 原动作位置不变，再次计算不会把30mm叠加成60mm。
                self.assertEqual(calculate_path(action, rp, cfg, env, rp.W + 30).path[-1],
                                 result.path[-1])
                action['action'] = terminal
                record_sent_placement(action, result, rp, env)
                self.assertEqual(rp.sent_positions, {})
                self.assertEqual(env.objects, [])
                self.assertEqual(env.display_objects, [])
                self.assertEqual(RobotPosition(order()).sent_positions, {})

    def test_full_order_commands_and_repeated_count(self):
        rp = RobotPosition(order())
        commands = [main.cmd_get_pallet, main.cmd_get_per_count, main.cmd_get_per_count]
        for _ in rp.boxes:
            commands += [main.cmd_get_box, main.cmd_get_path]
        commands += [main.cmd_get_pallet, main.cmd_get_path]
        server = FakeServer(commands)
        main.serve_commands(server, rp, config(), self.logger)
        self.assertEqual(server.messages[1], server.messages[2])
        self.assertEqual(rp.boxes, [])
        self.assertEqual(rp.robot_offsets, ['done'])
        self.assertEqual(decode(server.messages[-1])[0][2:5], (36, 2, 2460))
        self.assertEqual(decode(server.messages[-2])[-1][1], 3)

    def test_validation_failure_does_not_send_or_pop(self):
        data = order()
        data['car']['size']['H'] = 1500
        rp = RobotPosition(data)
        before = copy.deepcopy(rp.robot_offsets)
        server = FakeServer([main.cmd_get_path])
        with self.assertRaisesRegex(main.PathValidationError, '超高'):
            main.serve_commands(server, rp, config(), self.logger)
        self.assertEqual(rp.robot_offsets, before)
        self.assertEqual(server.messages, [])

    def test_send_failure_does_not_pop(self):
        for command in (main.cmd_get_box, main.cmd_get_path):
            rp = RobotPosition(order())
            before = copy.deepcopy((rp.boxes, rp.robot_offsets))
            with self.assertRaises(ConnectionError):
                main.serve_commands(FakeServer([command], fail_send=True), rp, config(), self.logger)
            self.assertEqual((rp.boxes, rp.robot_offsets), before)

    def test_callback_accepts_complete_order_only(self):
        with patch.object(main, 'logs', self.logger), \
                patch.object(main, '_accepting_orders', True):
            ok, message = main.callback(request_from_order(order()))
            self.assertTrue(ok, message)
            accepted = main.glob_position
            bad = request_from_order(order())
            bad.condition.boxes.add(type='B')
            ok, _ = main.callback(bad)
            self.assertFalse(ok)
            self.assertIs(main.glob_position, accepted)

    def test_locked_callback_does_not_replace_order(self):
        accepted = main.glob_position
        with patch.object(main, '_accepting_orders', False):
            ok, _ = main.callback(request_from_order(order()))
        self.assertFalse(ok)
        self.assertIs(main.glob_position, accepted)

    def test_p2_path_is_four_points(self):
        rp, cfg = RobotPosition(order()), config()
        action = copy.deepcopy(rp.robot_offsets[0])
        action['area'] = 'p2'
        result = calculate_path(action, rp, cfg, BinEnv(cfg))
        self.assertEqual(len(result.path), 4)
        self.assertEqual(result.path[0], result.path[1])
        self.assertEqual(result.path[-1], tuple(action['pos']))

    def test_validation_can_be_disabled_like_new_main(self):
        data = order()
        data['car']['size']['H'] = 1500
        rp, cfg = RobotPosition(data), config()
        cfg['chk_enable'] = False
        server = FakeServer([main.cmd_get_path])
        main.serve_commands(server, rp, cfg, self.logger)
        self.assertEqual(len(server.messages), 1)
        self.assertEqual(len(decode(server.messages[0])), 4)

    def test_no_active_rrt_dependency(self):
        sources = [SCRIPTS / 'runtime.py', SCRIPTS / 'node.py'] + list((SCRIPTS / 'rule_planning').glob('*.py'))
        for path in sources:
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith('rrt_planning'))
                elif isinstance(node, ast.Import):
                    self.assertTrue(all(not item.name.startswith('rrt_planning') for item in node.names))

    def test_fragmented_commands_do_not_merge(self):
        first = bytes.fromhex(main.cmd_get_box)
        second = bytes.fromhex(main.cmd_get_path)
        class FragmentSocket:
            def __init__(self):
                self.data = first + second
                self.calls = 0
            def recv(self, size):
                self.calls += 1
                count = min(size, 5) if self.calls == 1 else size
                result, self.data = self.data[:count], self.data[count:]
                return result
        app = SocketApp()
        app.sock = FragmentSocket()
        self.assertEqual(app.receive_message(16), main.cmd_get_box)
        self.assertEqual(app.receive_message(16), main.cmd_get_path)

    def test_socket_sends_complete_message(self):
        class CompleteSocket:
            def send(self, data):
                raise AssertionError('不能使用不保证完整发送的send')
            def sendall(self, data):
                self.sent = data
        app = SocketApp()
        app.sock = CompleteSocket()
        message = pallet_message(RobotPosition(order()))
        app.send_message(message)
        self.assertEqual(app.sock.sent, message)


if __name__ == '__main__':
    unittest.main()
