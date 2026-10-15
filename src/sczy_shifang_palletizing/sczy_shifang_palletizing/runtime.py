"""ROS2节点的独立工作循环；内部使用规则垛序和四点路径。

不启动 RRT，不依赖 robot_process 的 ROS2/点云运行环境。
路径校验失败时记录并结束当前连接，不以正常路径报文冒充失败响应。
"""

import logging
import os
import pickle
import threading
import time
from types import SimpleNamespace

from .rule_planning.environment import BinEnv
from .rule_planning.legacy_order import RobotPosition, parse_legacy_request
from .rule_planning.legacy_protocol import (
    CMD_GET_PALLET as cmd_get_pallet, CMD_GET_PER_COUNT as cmd_get_per_count,
    CMD_GET_BOX as cmd_get_box, CMD_GET_PATH as cmd_get_path,
    pallet_message, per_count_message, box_message, path_message,
)
from .rule_planning.path_planner import calculate_path, record_sent_placement

logs = logging.getLogger(__name__)
glob_data = None
glob_position = None
_order_lock = threading.Lock()
_accepting_orders = True
_order_generation = 0


def callback(data):
    """完整解析、生成并验证可用性后再发布订单；失败不发布半成品队列。"""
    global glob_data, glob_position
    try:
        with _order_lock:
            generation = _order_generation
            if not _accepting_orders:
                return False, '当前订单已锁定，请在下一轮重新下发'
        config = parse_legacy_request(data)
        position = RobotPosition(config)
        with _order_lock:
            if not _accepting_orders or generation != _order_generation:
                return False, '机器人已经连接，当前订单已锁定，请在下一轮重新下发'
            glob_data, glob_position = config, position
        logs.info('订单解析成功：%s箱、%s面、%s抓；使用规则路径，无RRT',
                  position.box_count, position.face_count, len(position.boxes))
        return True, 'receive succeed!'
    except Exception as error:
        logs.exception('订单解析失败: %s', error)
        return False, str(error)


class PathValidationError(ValueError):
    """不能通过旧协议下发的路径风险，退出本连接并保留未发送的队列首项。"""


def _save_visualization(action, result, environment, log_dir='logs'):
    """保留路径HTML显示，不再绘制搜索树；显示失败不能撤销已下发动作。"""
    from .auxiliary_methods.plotting import Plot
    output_dir = os.path.join(log_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    plot = Plot(f"rule_path_{stamp}_face{action['num_F']:02d}_grab{action['id']:04d}")
    plot.filename = os.path.join(output_dir, os.path.basename(plot.filename))
    space = SimpleNamespace(dimensions=3)
    plot.plot_path(space, result.path, result.size)
    plot.plot_obstacles(space, environment.display_objects)
    plot.plot_start(space, result.path[0])
    plot.plot_goal(space, result.path[-1])
    plot.draw(auto_open=False)


def serve_commands(server, rp, config, logger=None, corner_reader=None, stop_event=None):
    """处理原有四条命令。生成/编码/发送失败不提前弹出待执行动作。"""
    logger = logger or logs
    environment = BinEnv(config)
    distance = 0.0
    while stop_event is None or not stop_event.is_set():
        command = server.receive_message(byte_size=16)
        if not command:
            return
        if command == cmd_get_pallet:
            # 旧格式2块：总箱数/面数/车宽；箱长/宽/高。
            server.send_message(pallet_message(rp))
        elif command == cmd_get_per_count:
            # 旧格式1块：当前面总抓数、字符箱型编码；请求不推进队列。
            if config['use_corner']:
                if corner_reader is None:
                    from .socket_pkg.socket_client import create_tcp_client
                    corner_reader = create_tcp_client
                try:
                    distance = corner_reader(
                        config.get('corner_ip', '192.168.1.100'),
                        config.get('corner_port', 9002),
                        b'\xff\xfe\x01\x00\x01\xfd')
                    logger.info('角点距离=%.2fmm，车宽差=%.2fmm', distance, distance - rp.W)
                except Exception as error:
                    distance = 0.0
                    logger.warning('本次角点读取失败，不沿用上次补偿: %s', error)
            server.send_message(per_count_message(rp))
        elif command == cmd_get_box:
            # 旧格式1块：P1/P2实际箱数或P3配方5/6、字符箱型编码。
            if not rp.boxes:
                logger.warning('来料队列已经结束，不返回重复配方')
                return
            box = rp.boxes[0]
            server.send_message(box_message(box, rp.box_type))
            rp.boxes.pop(0)
            logger.info('第%s面 第%s抓：%s，箱数=%s',
                        box['num_F'], box['id'], box['area'], box['num'])
        elif command == cmd_get_path:
            # 旧格式每路径点1块，前三点带区域码，末点带继续/换面/完成码。
            if not rp.robot_offsets or rp.robot_offsets[0] == 'done':
                logger.info('路径队列已经结束，不返回重复路径')
                return
            action = rp.robot_offsets[0]
            start = time.perf_counter()
            result = calculate_path(action, rp, config, environment, distance)
            if result.issues:
                message = (f"第{action['num_F']}面 第{action['id']}抓路径检查失败: "
                           + '；'.join(result.issues))
                logger.error('%s；旧报文无校验状态，本次不下发路径、不消耗队列', message)
                raise PathValidationError(message)
            server.send_message(path_message(result.path, action))
            rp.robot_offsets.pop(0)
            logger.info('第%s面 第%s抓 路径=%s APP-Y偏移=%.1fmm，耗时=%.4fs',
                        action['num_F'], action['id'], result.path,
                        result.y_offset_app, time.perf_counter() - start)
            if config.get('show_env', False):
                try:
                    _save_visualization(action, result, environment, config.get('log_dir', 'logs'))
                except Exception as error:
                    logger.warning('路径已下发，可视化保存失败: %s', error)
            record_sent_placement(action, result, rp, environment)
            if action['action'] == 1:
                logger.info('第%s面结束', action['num_F'])
            elif action['action'] == 2:
                logger.info('全部路径规划结束')
        else:
            logger.warning('收到未知命令或报文不完整: %s，结束当前连接', command)
            return


class RuntimeService:
    """阻塞网络操作放在工作线程；ROS2主线程保持spin并可主动关闭连接。"""

    def __init__(self, config, logger, offline_order_file=''):
        self.config = dict(config)
        self.logger = logger
        self.offline_order_file = offline_order_file
        self.stop_event = threading.Event()
        self._resources_lock = threading.Lock()
        self._socket_server = None
        self._grpc_server = None

    def stop(self):
        """关闭accept/recv等待和gRPC服务，让ROS2退出不被网络阻塞。"""
        global _accepting_orders
        self.stop_event.set()
        with _order_lock:
            _accepting_orders = False
        with self._resources_lock:
            server, grpc_server = self._socket_server, self._grpc_server
        if grpc_server is not None:
            grpc_server.stop()
        if server is not None:
            server.close()

    def run(self):
        """每次连接处理一个完整旧协议订单；失败后不自动重放旧订单。"""
        global logs, glob_data, glob_position, _accepting_orders, _order_generation
        from .grpc_pkg.grpc_server import GrpcServer
        from .socket_pkg.socket_server import SocketApp

        logs = self.logger
        offline_used = False
        while not self.stop_event.is_set():
            grpc_server = None
            server = SocketApp()
            with self._resources_lock:
                self._socket_server = server
            try:
                if self.config['off_line_mode']:
                    if offline_used:
                        self.logger.warning('离线订单本轮已结束；重新执行须核对现场并重启节点，不自动整单重放')
                        return
                    # ROS2启动不使用input()，离线文件由参数明确指定。
                    with open(self.offline_order_file, 'rb') as stream:
                        rp = RobotPosition(pickle.load(stream))
                    offline_used = True
                else:
                    with _order_lock:
                        glob_data, glob_position = None, None
                        _order_generation += 1
                        _accepting_orders = True
                    # 与stop()共享锁，避免停止后仍在竞态中启动gRPC服务。
                    with self._resources_lock:
                        if self.stop_event.is_set():
                            return
                        grpc_server = GrpcServer(self.config.get('grpc_port', 5007), callback)
                        self._grpc_server = grpc_server
                        grpc_server.run()
                    self.logger.info('等待旧协议订单与机器人连接...')
                if self.stop_event.is_set() or not server.start(self.config['ip'], self.config['port']):
                    return
                if grpc_server is not None:
                    with _order_lock:
                        _accepting_orders = False
                        rp = glob_position
                    grpc_server.stop()
                    if rp is None:
                        raise ValueError('机器人连接前尚无成功解析的订单，请重新下发订单并重连')
                serve_commands(server, rp, self.config, self.logger, stop_event=self.stop_event)
            except Exception as error:
                if not self.stop_event.is_set():
                    self.logger.exception('当前会话结束: %s；不自动重发已下发动作', error)
                    self.stop_event.wait(1.0)
            finally:
                with _order_lock:
                    _accepting_orders = False
                if grpc_server is not None:
                    grpc_server.stop()
                server.close()
                with self._resources_lock:
                    self._socket_server = None
                    self._grpc_server = None
