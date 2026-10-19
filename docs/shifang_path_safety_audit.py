"""什邡四点路径的离线核查；只用内存中的模拟通信，不连接硬件。

执行：python3 -B docs/shifang_path_safety_audit.py
结果是复现当前检查缺口的诊断，不是现场安全认证。依赖 numpy。
"""
import ast
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import sys
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1] / 'src/sczy_shifang_palletizing'
sys.path.insert(0, str(PACKAGE))
from sczy_shifang_palletizing import runtime
from sczy_shifang_palletizing.rule_planning.environment import BinEnv
from sczy_shifang_palletizing.rule_planning.legacy_order import RobotPosition
from sczy_shifang_palletizing.rule_planning.path_planner import (
    _obstacle_free, calculate_path, record_sent_placement,
)


def fixture():
    # Read only the existing pure fixture, avoiding protobuf/ROS test imports.
    nodes = ast.parse((PACKAGE / 'test/test_rule_migration.py').read_text()).body
    function = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == 'order')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'fixture', 'exec'), namespace)
    return namespace['order']()


def swept_overlap(start, end, size, obstacle):
    """Exact open-interval overlap for a fixed-orientation AABB translating linearly.

    This does not model robot links, rotation, blended motion or TCP conversion.
    Face-only contact is allowed; positive overlap on all axes is penetration.
    """
    low, high = 0.0, 1.0
    for axis in range(3):
        lower = obstacle[axis] - size[axis]
        upper = obstacle[axis + 3]
        delta = end[axis] - start[axis]
        if abs(delta) < 1e-12:
            if not lower + 1e-8 < start[axis] < upper - 1e-8:
                return None
        else:
            a, b = sorted(((lower - start[axis]) / delta,
                           (upper - start[axis]) / delta))
            low, high = max(low, a), min(high, b)
    if high - low <= 1e-9:
        return None
    ratio = (low + high) / 2
    return {'interval': [low, high], 'example_point': [
        start[i] + ratio * (end[i] - start[i]) for i in range(3)]}


def old_edge_check(start, end, size, obstacles):
    # Execute the unchanged legacy methods; replace only Rtree's point lookup.
    namespace = {}
    exec((PACKAGE / 'scripts/legacy_auxiliary_methods/geometry.py').read_text(), namespace)
    tree = ast.parse((PACKAGE / 'scripts/legacy_auxiliary_methods/search_space.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ('obstacle_free', 'collision_free')]
    exec(compile(ast.Module(body=methods, type_ignores=[]), 'legacy_methods', 'exec'), namespace)

    class DirectIndex:
        def count(self, point):
            return sum(all(o[i] <= point[i] <= o[i + 3] for i in range(3))
                       for o in obstacles)

    cls = type('LegacySearch', (), {name: namespace[name]
               for name in ('obstacle_free', 'collision_free')})
    space = cls()
    space.obs = DirectIndex()
    return space.collision_free(start, end, 1, size)


def evaluate_order(data, config, legacy):
    rp, environment = RobotPosition(data), BinEnv(config)
    old = legacy.RobotPosition(copy.deepcopy(data))
    reconstructed = []
    for a in rp.robot_offsets[:-1]:
        item = {k: copy.deepcopy(a[k]) for k in
                ('id', 'area', 'num', 'num_F', 'action', 'dir', 'pos', 'size')}
        item['num'] = sum(item['num'])
        reconstructed.append(item)
    assert reconstructed == old.robot_offsets[:-1]
    result = {'accepted': 0, 'rejected': 0, 'physical_overlap_actions': 0,
              'nominal_lateral_overrun_actions': 0}
    for action in rp.robot_offsets[:-1]:
        path = calculate_path(action, rp, config, environment)
        if path.issues:
            result['rejected'] += 1
            break  # Match runtime: do not continue past a rejected path.
        result['accepted'] += 1
        if any(swept_overlap(start, end, path.size, obstacle)
               for start, end in zip(path.path, path.path[1:])
               for obstacle in environment.display_objects):
            result['physical_overlap_actions'] += 1
        if any(p[1] < 0 or p[1] + path.size[1] > rp.W for p in path.path):
            result['nominal_lateral_overrun_actions'] += 1
        record_sent_placement(action, path, rp, environment)
    return result


def main():
    config = json.loads((PACKAGE / 'config/planner.json').read_text())
    config['show_env'] = False
    spec = importlib.util.spec_from_file_location(
        'legacy_audit', PACKAGE / 'scripts/rrt_env/environment_3D.py')
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    baseline = evaluate_order(fixture(), config, legacy)
    totals = dict(orders=0, **{k: 0 for k in baseline})
    for width, height, n, group, layers, n3, t3 in itertools.product(
            (200, 250, 300, 350), (350, 500, 580, 650), (7, 8, 9),
            ((4, 3), (4, 4), (2, 3, 4)), (2, 3, 4), (0, 3), (0, 1)):
        if (sum(group) != n or (n3 == 0) != (t3 == 0) or n * width > 2460
                or layers * height + t3 * width + 100 > 2800):
            continue
        data = fixture()
        data['box']['A']['size'].update(W=width, H=height)
        stack = [0] * (n + 1)
        for index in list(itertools.accumulate(group))[:-1]:
            stack[index] = 100 / (len(group) - 1)
        data['regular'][0].update(N1=n, Group=[list(group), list(group)],
            Stack=[stack, stack.copy()], T12=layers, N3=n3, T3=t3, F13=1)
        result = evaluate_order(data, config, legacy)
        totals['orders'] += 1
        for key, value in result.items():
            totals[key] += value

    # Synthetic obstacle state: proves a checking gap, not a recorded site order.
    rp, environment = RobotPosition(fixture()), BinEnv(config)
    action = rp.robot_offsets[0]
    environment.step(dict(action, pos=[0, 200, 750], num=[1], action=0))
    path = calculate_path(action, rp, config, environment)
    obstacle = environment.display_objects[0]
    witness = swept_overlap(path.path[1], path.path[2], path.size, obstacle)
    assert not path.issues and witness
    assert all(swept_overlap(p, p, path.size, obstacle) is None for p in path.path)
    old_passes = old_edge_check(path.path[1], path.path[2], path.size, environment.objects)
    assert old_passes is False

    class MemoryServer:
        def __init__(self):
            self.commands = iter([runtime.cmd_get_path, ''])
            self.messages = []

        def receive_message(self, byte_size):
            return next(self.commands)

        def send_message(self, message):
            self.messages.append(message)

    server = MemoryServer()
    with patch.object(runtime, 'BinEnv', return_value=environment):
        runtime.serve_commands(server, rp, config)
    assert len(server.messages) == 1

    narrow = fixture()
    narrow['box']['A']['size']['W'] = 300
    narrow['car']['size']['W'] = 2300
    stack = [0] * 8
    stack[4] = 100
    narrow['regular'][0].update(N1=7, Group=[[4, 3], [4, 3]], Stack=[stack, stack.copy()])
    rp = RobotPosition(narrow)
    narrow_path = calculate_path(rp.robot_offsets[0], rp, config, BinEnv(config))
    overrun = narrow_path.path[0][1] + narrow_path.size[1] - rp.W
    assert not narrow_path.issues and overrun == 100
    assert _obstacle_free((0, 0, 0), (460, 1000, 680),
                          [(100, 100, 100, 200, 200, 200)])

    print(json.dumps({
        'baseline': baseline, 'constructed_regular_orders': totals,
        'sequence_comparison': 'all evaluated orders match the archived legacy sequence',
        'synthetic_middle_segment': {'path': path.path, 'size': path.size,
            'obstacle': obstacle, 'current_issues': path.issues,
            'swept_overlap': witness, 'legacy_edge_check_passes': old_passes,
            'mock_runtime_sent_bytes': [len(m) for m in server.messages]},
        'constructed_narrow_car': {'width': rp.W, 'path': narrow_path.path,
            'size': narrow_path.size, 'current_issues': narrow_path.issues,
            'x0_right_overrun_mm': overrun},
        'vertex_only_check_misses_containment': True,
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
