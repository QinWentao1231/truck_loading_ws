"""ROS2包元数据、依赖及可中断网络等待检查；不连接机器人。"""

import ast
import importlib.metadata
from pathlib import Path
import socket
import sys
import threading
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from packaging.requirements import Requirement

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from sczy_shifang_palletizing.socket_pkg.socket_server import SocketApp
from sczy_shifang_palletizing.runtime import RuntimeService


class Ros2PackagingTests(unittest.TestCase):
    def test_package_and_entrypoint(self):
        root = ET.parse(PACKAGE / 'package.xml').getroot()
        self.assertEqual(root.findtext('name'), 'sczy_shifang_palletizing')
        self.assertEqual(root.findtext('export/build_type'), 'ament_python')
        self.assertEqual(root.findtext('buildtool_depend'), 'ament_python')
        self.assertFalse((PACKAGE / 'CMakeLists.txt').exists())
        self.assertTrue((PACKAGE / 'resource/sczy_shifang_palletizing').exists())
        self.assertTrue((PACKAGE / 'launch/sczy_shifang_palletizing.launch.py').exists())
        source = (PACKAGE / 'setup.py').read_text()
        self.assertIn('sczy_shifang_palletizing.node:main', source)
        self.assertIn('lib/sczy_shifang_palletizing', (PACKAGE / 'setup.cfg').read_text())

    def test_requirements_match_setup_and_current_environment(self):
        requirements = [line.strip() for line in (PACKAGE / 'requirements.txt').read_text().splitlines()
                        if line.strip() and not line.startswith('#')]
        setup = next(n for n in ast.walk(ast.parse((PACKAGE / 'setup.py').read_text()))
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'setup')
        declared = ast.literal_eval(next(k.value for k in setup.keywords if k.arg == 'install_requires'))
        self.assertEqual(set(declared), set(requirements) | {'setuptools'})
        for text in requirements:
            requirement = Requirement(text)
            self.assertIn(importlib.metadata.version(requirement.name), requirement.specifier)
        self.assertFalse(any('rtree' in s.lower() or 'colorlog' in s.lower() for s in requirements))

    def test_accept_wait_is_interruptible(self):
        entered = threading.Event()
        released = threading.Event()
        class FakeListener:
            def setsockopt(self, *args): pass
            def settimeout(self, value): pass
            def bind(self, address): pass
            def listen(self, count): pass
            def accept(self):
                entered.set()
                released.wait(2)
                raise OSError('closed by shutdown')
            def shutdown(self, how): released.set()
            def close(self): released.set()
        app = SocketApp()
        results = []
        with patch('sczy_shifang_palletizing.socket_pkg.socket_server.socket.socket', return_value=FakeListener()):
            worker = threading.Thread(target=lambda: results.append(app.start('127.0.0.1', 8001)))
            worker.start()
            try:
                self.assertTrue(entered.wait(1))
                app.close()
                worker.join(1)
                self.assertFalse(worker.is_alive())
                self.assertEqual(results, [False])
            finally:
                app.close()
                worker.join(2)

    def test_receive_wait_is_interruptible(self):
        entered = threading.Event()
        released = threading.Event()
        class FakeConnection:
            def recv(self, count):
                entered.set()
                released.wait(2)
                return b''
            def shutdown(self, how): released.set()
            def close(self): released.set()
        app = SocketApp()
        app.sock = FakeConnection()
        results = []
        worker = threading.Thread(target=lambda: results.append(app.receive_message(16)))
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            app.close()
            worker.join(1)
            self.assertFalse(worker.is_alive())
            self.assertEqual(results, [''])
            app.close()  # 重复退出是幂等的。
        finally:
            app.close()
            worker.join(2)

    def test_stopped_runtime_does_not_open_ports(self):
        import logging
        runtime = RuntimeService({}, logging.getLogger('test'))
        with patch('sczy_shifang_palletizing.runtime._accepting_orders', True):
            runtime.stop()
            with patch('sczy_shifang_palletizing.grpc_pkg.grpc_server.GrpcServer') as grpc_server:
                runtime.run()
                grpc_server.assert_not_called()


if __name__ == '__main__':
    unittest.main()
