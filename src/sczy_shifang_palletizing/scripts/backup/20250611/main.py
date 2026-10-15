import os
import math
import pickle
import sys

current_file = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file)
sys.path.insert(0, current_directory)
import time
import struct
import json
import numpy as np
from rrt_env.environment_3D import BinEnv, RobotPosition
from rrt_planning.rrt_star import RRTStar
from auxiliary_methods.search_space import SearchSpace
from socket_pkg.socket_server import SocketApp
from socket_pkg import socket_client
from grpc_pkg.grpc_server import GrpcServer
from auxiliary_methods.plotting import Plot
from auxiliary_methods.logging import Logger

cmd_get_pallet = 'fefe0000010100000000000000000000'
cmd_get_per_count = 'fefe0000010200000000000000000000'
cmd_get_box = 'fefe0000010300000000000000000000'
cmd_get_path = 'fefe0000010400000000000000000000'


def callback(data):
    # 创建结构体
    global glob_data
    glob_data = {'box_list': [], 'box': [], 'car': None, 'regular': None, 'trapezoid': None}
    glob_data['car'] = {'size': None, 'reserve': None}
    glob_data['car']['size'] = {'L': None, 'W': None, 'H': None}
    glob_data['car']['reserve'] = {'L': None, 'W': None, 'H': None}
    glob_data['regular'] = [
        {'N1': None, 'N2': None, 'N3': None, 'T12': None, 'T3': None, 'F13': None, 'F2': None, 'E': None, 'Nx': None,
         'Stack': [], 'Group': []}]
    glob_data['trapezoid'] = []
    # 解析gRPC数据并赋值给结构体
    glob_data['box_list'].append(data.condition.boxes[0].type)
    glob_data['box_list'].append(data.condition.boxes[1].type)
    glob_data['box'].append({'type': data.condition.boxes[0].type, 'size': [data.condition.boxes[0].original.L,
                                                                            data.condition.boxes[0].original.W,
                                                                            data.condition.boxes[0].original.H]})
    glob_data['box'].append({'type': data.condition.boxes[1].type, 'size': [data.condition.boxes[1].original.L,
                                                                            data.condition.boxes[1].original.W,
                                                                            data.condition.boxes[1].original.H]})
    glob_data['car']['size']['L'] = data.condition.car.original.L
    glob_data['car']['size']['W'] = data.condition.car.original.W
    glob_data['car']['size']['H'] = data.condition.car.original.H
    glob_data['car']['reserve']['L'] = data.condition.car.reserve.L
    glob_data['car']['reserve']['W'] = data.condition.car.reserve.W
    glob_data['car']['reserve']['H'] = data.condition.car.reserve.H
    glob_data['regular'][0]['N1'] = data.answer.regular[0].N1
    glob_data['regular'][0]['N2'] = data.answer.regular[0].N2
    glob_data['regular'][0]['N3'] = data.answer.regular[0].N3
    glob_data['regular'][0]['T12'] = data.answer.regular[0].T12
    glob_data['regular'][0]['T3'] = data.answer.regular[0].T3
    glob_data['regular'][0]['F13'] = data.answer.regular[0].F13
    glob_data['regular'][0]['F2'] = data.answer.regular[0].F2
    glob_data['regular'][0]['E'] = data.answer.regular[0].E
    glob_data['regular'][0]['Nx'] = data.answer.regular[0].Nx
    glob_data['regular'][0]['Type'] = data.answer.regular[0].Type
    for x in data.answer.regular[0].Stack:
        temp_l = []
        for y in x.Width:
            temp_l.append(y)
        glob_data['regular'][0]['Stack'].append(temp_l)
    for x in data.answer.regular[0].Group:
        temp_l = []
        for y in x.Unit:
            temp_l.append(y)
        glob_data['regular'][0]['Group'].append(temp_l)
    for i in range(0, len(data.answer.trapezoid)):
        if data.answer.trapezoid[i].N1 != 0:
            glob_data['trapezoid'].append({'N1': data.answer.trapezoid[i].N1, 'N3': data.answer.trapezoid[i].N3,
                                           'T1': data.answer.trapezoid[i].T1, 'T3': data.answer.trapezoid[i].T3,
                                           'Nx': data.answer.trapezoid[i].Nx, 'Stack': [], 'Group': [],
                                           'Isdoor': data.answer.trapezoid[i].Isdoor,
                                           'Type': data.answer.trapezoid[i].Type})
            for x in data.answer.trapezoid[i].Stack:
                temp_l = []
                for y in x.Width:
                    temp_l.append(y)
                glob_data['trapezoid'][i]['Stack'].append(temp_l)
            for x in data.answer.trapezoid[i].Group:
                temp_l = []
                for y in x.Unit:
                    temp_l.append(y)
                glob_data['trapezoid'][i]['Group'].append(temp_l)
    logs.info("已接受垛型数据，请将机器人进行连接或重新下发垛型数据")
    return True, "receive succeed!"


if __name__ == "__main__":
    # 日志配置
    logs = Logger(level="debug").get_log()
    logs.info("***垛序及机器人路径规划程序启动 ver.0.5.0-alpha***")
    try:
        with open(current_directory + "/config.json", "r", encoding="utf-8") as f:
            config_be = json.load(f)
        # 配置文件参数获取
        ip = config_be['ip']
        port = config_be['port']
        level = config_be['level']
        reserve_grip = config_be['reserve_grip']
        off_line_mode = config_be['off_line_mode']
        show_env = config_be['show_env']
    except Exception as e:
        logs.error(e)
        logs.error("读取参数文件失败,请检查项目根目录config.json文件")
        sys.exit(1)
    logs.info("读取参数文件成功！")
    # 主循环
    while True:
        # 选择读取垛型参数的方式
        if off_line_mode:
            try:
                # 离线读取文件模式
                logs.warning("已启用离线模式，垛型参数文件夹为 {}".format(os.getcwd() + '/pkl_data/'))
                time.sleep(1)
                try:
                    file
                except NameError:
                    file = input("请输入垛型文件名称：")
                config_rp = pickle.load(file=open('{0}/pkl_data/{1}'.format(current_directory, file), 'rb'))
                logs.info("读取垛型配置文件成功！配置文件路径为 {}".format(os.getcwd() + '/pkl_data/' + file))
            except Exception as e:
                logs.error(e)
                logs.error("{} 垛型配置文件读取失败".format(os.getcwd() + '/pkl_data/' + file))
                sys.exit(1)
        else:
            # 在线读取垛型模式
            glob_data = None
            if 'svr' not in vars():
                svr = GrpcServer(5007, callback)
                svr.run()
            logs.warning("已启用在线模式，服务器已启动，等待接收垛型数据...")
        try:
            # logs.info("等待机器人连接中...")
            if 'server' not in vars():
                server = SocketApp()
            server.start(ip, port)
        except Exception as e:
            logs.error(e)
            logs.error("创建机器人服务器失败，请检查项目根目录config.json文件")
            sys.exit(1)
        logs.info("连接机器人成功！")
        # 停止接受垛型数据
        if not off_line_mode:
            config_rp = glob_data
            svr.stop()
            del svr
            logs.info("停止接受垛型")
        # 创建环境
        be = BinEnv(config_be)
        logs.info("码垛环境初始化成功！")
        # 计算垛序
        try:
            rp = RobotPosition(config_rp)
            actions = rp.robot_offsets
            logs.info("计算垛序成功！")
        except Exception as e:
            logs.error(e)
            logs.error("垛序计算失败，请检查垛型数据")
            sys.exit(1)
        # 环境初始化
        be.reset()
        dis = 0
        # 指令集循环
        # try:
        while True:
            mes_hex = server.receive_message(byte_size=16)
            # mes_hex = cmd_get_path
            if mes_hex == cmd_get_pallet:
                send_data = []
                send_data.append(b'\xfe\xfe')  # start_word
                send_data.append(b'\x01\x00')  # VERSION
                send_data.append(b'\x10')  # trans_to
                send_data.append(b'\x23')  # action_id
                send_data.append(b'\x01')  # block_type
                send_data.append(b'\x02')  # num_blocks
                send_data.append(b'\x00\x00')  # err_code
                send_data.append(b'\x00\x00\x00\x91')  # mes_data_num
                send_data.append(b'\x00')  # index
                send_data.append(b'\x00\x00\x00\x00')  # class_id
                send_data.append(struct.pack('!f', rp.box_count))  # 总箱数
                send_data.append(struct.pack('!f', rp.robot_offsets[-2]['num_F']))  # 码垛面数
                send_data.append(struct.pack('!f', rp.W))  # 车厢宽
                send_data.append(b'\x00\x00\x00\x00')  # null
                send_data.append(b'\x00\x00\x00\x00')  # null
                send_data.append(b'\x00\x00\x00\x00')  # null
                send_data.append(b'\x00\x00\x00\x00')  # length
                send_data.append(b'\x00\x00\x00\x00')  # width
                send_data.append(b'\x00\x00\x00\x00')  # height
                send_data.append(b'\x00')  # index
                send_data.append(b'\x00\x00\x00\x00')  # class_id
                send_data.append(struct.pack('!f', rp.l))  # box_l
                send_data.append(struct.pack('!f', rp.w))  # box_w
                send_data.append(struct.pack('!f', rp.h))  # box_h
                send_data.append(b'\x00\x00\x00\x00')  # null
                send_data.append(b'\x00\x00\x00\x00')  # null
                send_data.append(b'\x00\x00\x00\x00')  # null
                send_data.append(b'\x00\x00\x00\x00')  # length
                send_data.append(b'\x00\x00\x00\x00')  # width
                send_data.append(b'\x00\x00\x00\x00')  # height
                send_data.append(b'\x00\x00')  # reserve
                send_datas = b''
                for i in send_data:
                    send_datas += i
                server.send_message(send_datas)
                if rp.Nx == 0 or rp.E == 2:
                    logs.debug('总箱数: {}, 码垛面数：{}, 车厢宽度：{}, 箱子尺寸：长{} 宽{} 高{}'.format(
                        rp.box_count, rp.F13 + len(rp.trapezoid), round(rp.W, 2), rp.l, rp.w, rp.h))
                else:
                    logs.debug('总箱数: {}, 码垛面数：{}, 车厢宽度：{}, 箱子尺寸：长{} 宽{} 高{}'.format(
                        rp.box_count, rp.F13 + 1 + len(rp.trapezoid), round(rp.W, 2), rp.l, rp.w, rp.h))
            elif mes_hex == cmd_get_per_count:
                pallet_cnt = rp.cal_floor_count()
                if config_be['use_corner']:
                    dis = socket_client.create_tcp_client("192.168.1.100", 9002,
                                                          b'\xff\xfe\x01\x00\x01\xfd')
                    logs.warning("value between corners and car width: {:.2f}".format(dis - rp.W))
                send_data = []
                send_data.append(b'\xfe\xfe')  # start_word
                send_data.append(b'\x01\x00')  # version
                send_data.append(b'\x10')  # trans_to
                send_data.append(b'\x23')  # action_id
                send_data.append(b'\x01')  # block_type
                send_data.append(b'\x01')  # num_blocks
                send_data.append(b'\x00\x00')  # err_code
                send_data.append(b'\x00\x00\x00\x91')  # mes_data_num
                send_data.append(b'\x00')  # index
                send_data.append(b'\x00\x00\x00\x00')  # class_id
                send_data.append(struct.pack('!f', pallet_cnt))  # 单面码垛放置次数
                send_data.append(struct.pack('!f', ord(rp.box_type)))  # 当前箱子类型
                send_data.append(b'\x00\x00\x00\x00')  # null
                send_data.append(b'\x00\x00\x00\x00')  # null
                send_data.append(b'\x00\x00\x00\x00')  # null
                send_data.append(b'\x00\x00\x00\x00')  # null
                send_data.append(b'\x00\x00\x00\x00')  # length
                send_data.append(b'\x00\x00\x00\x00')  # width
                send_data.append(b'\x00\x00\x00\x00')  # height
                send_data.append(b'\x00\x00')  # reserve
                send_datas = b''
                for i in send_data:
                    send_datas += i
                server.send_message(send_datas)
                logs.debug('单面码垛放置次数：{}'.format(pallet_cnt))
            elif mes_hex == cmd_get_box:
                box = rp.boxes.pop(0)
                if box['area'] != 'p3':
                    box_cfg = box['num']
                elif box['num'] == 1:
                    box_cfg = 5
                elif box['num'] == 2:
                    box_cfg = 6
                send_data = []
                send_data.append(b'\xfe\xfe')  # start_word
                send_data.append(b'\x01\x00')  # version
                send_data.append(b'\x10')  # trans_to
                send_data.append(b'\x23')  # action_id
                send_data.append(b'\x01')  # block_type
                send_data.append(b'\x01')  # num_blocks
                send_data.append(b'\x00\x00')  # err_code
                send_data.append(b'\x00\x00\x00\x91')  # mes_data_num
                send_data.append(b'\x00')  # index
                send_data.append(b'\x00\x00\x00\x00')  # class_id
                send_data.append(struct.pack('!f', box_cfg))  # 来料箱子配方号
                send_data.append(struct.pack('!f', ord(rp.box_type)))  # 当前箱子类型
                send_data.append(b'\x00\x00\x00\x00')  # null
                send_data.append(b'\x00\x00\x00\x00')  # null
                send_data.append(b'\x00\x00\x00\x00')  # null
                send_data.append(b'\x00\x00\x00\x00')  # null
                send_data.append(b'\x00\x00\x00\x00')  # length
                send_data.append(b'\x00\x00\x00\x00')  # width
                send_data.append(b'\x00\x00\x00\x00')  # height
                send_data.append(b'\x00\x00')  # reserve
                send_datas = b''
                for i in send_data:
                    send_datas += i
                server.send_message(send_datas)
                logs.debug('来料箱子配方号：{}'.format(box_cfg))
                if box['action'] == 2:
                    logs.info("垛序规划结束 ！ ")
            elif mes_hex == cmd_get_path:
                action = rp.robot_offsets.pop(0)
                if action['id'] == 1:
                    logs.info("*** NO.{} 码垛面 开始！***".format(1))
                logs.debug("Round.{}/{} start planning...".format(action['id'], len(rp.ori_offsets) - 1))
                if action == 'done':
                    logs.info("码垛路径规划结束 ！ ")
                    break
                rp.l = action['size'][0]
                rp.w = action['size'][1]
                rp.h = action['size'][2]
                start_time = time.time()
                obstacles = be.objects
                boxs = be.to_box(action)
                if action['area'] == 'p1':
                    if action['dir'] == 2:
                        x_app = tuple([x + y for x, y in zip(boxs[0].position, config_be['p2_app_offset_list'][0])])
                    else:
                        x_app = tuple([x + y for x, y in zip(boxs[0].position, config_be['p1_app_offset_list'][0])])
                    x_init = [rp.l + 70,
                              x_app[1],
                              1350 if action['pos'][2] <= 1350 else x_app[2]]
                    if action['pos'][2] >= 3 * rp.h:
                        x_init[0] = config_be['p3_init_pos'][0]
                    if (action['num_F'] <= rp.F13 + len(rp.trapezoid) - sum(x['Isdoor'] for x in rp.trapezoid)
                            and action['dir'] == 1 and action['pos'][1] < (rp.W - 5 * rp.w)):
                        x_init[1] += 1 * rp.w
                    if (action['num_F'] > rp.F13 + len(rp.trapezoid) - sum(x['Isdoor'] for x in rp.trapezoid)
                            and action['dir'] == 1 and action['pos'][1] < (rp.W - 6 * rp.w)):
                        x_init[1] += 1 * rp.w
                    if action['dir'] == 2 and action['pos'][1] > rp.w:
                        x_init[1] -= 1*rp.w
                    x0 = [x_init[0], 1250, x_init[2]]
                    size = (boxs[0].length, boxs[0].width * action['num'], boxs[0].height + reserve_grip[2])
                elif action['area'] == 'p2':
                    x_init = tuple(config_be['p2_init_pos'])
                    x_app = tuple([x + y for x, y in zip(boxs[0].position, config_be['p2_app_offset_list'][0])])
                    size = (boxs[0].length * 4, boxs[0].width, boxs[0].height + reserve_grip[2])
                elif action['area'] == 'p3':
                    x_app = tuple([x + y for x, y in zip(boxs[0].position, config_be['p3_app_offset_list'][0])])
                    if action['pos'][2] < 3 * rp.h:
                        x_init = [rp.l + 70,
                                  x_app[1] if action['pos'][1] > rp.h else 1.5 * rp.h,
                                  3 * rp.h]
                    elif 3 * rp.h <= action['pos'][2] < 4 * rp.h:
                        x_init = [rp.l + 70,
                                  x_app[1] if action['pos'][1] > rp.h else 1.5 * rp.h,
                                  x_app[2]]
                    else:
                        x_init = [config_be['p3_init_pos'][0],
                                  x_app[1],
                                  x_app[2]]
                    # if action['pos'][2] < 4 * rp.h:
                    #     x_init = [rp.l + 70,
                    #               x_app[1] if action['pos'][1] > rp.h else 1.5 * rp.h,
                    #               3*rp.h + config_be['p3_app_offset_list'][0][2]]
                    # else:
                    #     x_init = [config_be['p3_init_pos'][0],
                    #               x_app[1],
                    #               x_app[2]]
                    size = (boxs[0].length, boxs[0].width * 2, boxs[0].height + reserve_grip[2])
                else:
                    raise Exception('unknown area ! ')
                if (config_be['use_corner']
                        and action['dir'] == 2 and abs(dis - rp.W) < config_be['p1_app_offset_list'][0][1]):
                    x_goal = tuple([boxs[0].position[0],
                                    boxs[0].position[1] + (dis - rp.W),
                                    boxs[0].position[2]])
                    logs.warning('goal has been corrected, value:{:.2f}'.format(dis - rp.W))
                else:
                    x_goal = tuple(boxs[0].position)

                #  车内总体搜索区域
                x_dimensions = np.array([(0, rp.l + 100),
                                         (0, rp.W - size[1]),
                                         (0, rp.H - size[2])])

                #  单条路径搜索区域切片
                x_slices = np.linspace(x_init[0], x_app[0], level)
                y_slices = np.linspace(x_init[1], x_app[1], level)
                z_slices = np.linspace(x_init[2], x_app[2], level)

                #  去除不合理的路径点
                X = SearchSpace(np.array([(0, 100), (0, 100), (0, 100)]), obstacles)
                j = []
                if not X.obstacle_free((x_slices[-1], y_slices[-1], z_slices[-1]), size):
                    raise Exception('接近点不合理！')
                for i in range(0, len(x_slices)):
                    if not X.obstacle_free((x_slices[i], y_slices[i], z_slices[i]), size) \
                            or y_slices[i] > x_dimensions[1][1]:
                        j.append(i)
                x_slices = np.delete(x_slices, j)
                y_slices = np.delete(y_slices, j)
                z_slices = np.delete(z_slices, j)
                step = int(math.sqrt((x_slices[0] - x_slices[1]) ** 2 +
                                     (y_slices[0] - y_slices[1]) ** 2 +
                                     (z_slices[0] - z_slices[1]) ** 2) / 10)

                #  开始规划
                if action['area'] != 'p3':
                    path = [x0, x_init]
                else:
                    path = [x_init]
                for i in range(len(y_slices) - 1):
                    #  设置规划参数
                    Q = np.array([(step, step + 1)])  # length of tree edges
                    r = 1  # length of smallest edge to check for intersection with obstacles
                    max_samples = 1024  # max number of samples to take before timing out
                    rewire_count = 32  # optional, number of nearby branches to rewire
                    prc = 0.3  # probability of checking for a connection to goal

                    #  创建子搜索区域
                    search_x = (x_slices[i+1], x_slices[i])
                    if y_slices[i] < y_slices[i + 1]:
                        search_y = (y_slices[i], y_slices[i + 1]+config_be['search_range']) \
                            if y_slices[i + 1]+config_be['search_range'] > x_dimensions[1][1] \
                            else (y_slices[i], x_dimensions[1][1])
                    else:
                        search_y = (y_slices[i+1]-config_be['search_range'], y_slices[i]) \
                            if y_slices[i+1]-config_be['search_range'] > 0 \
                            else (0, y_slices[i])

                    # if search_y[1] > x_dimensions[1][1] + 20:
                    #     logs.error("Unreasonable car y reserve")
                    #     temp_list = list(search_y)
                    #     temp_list[1] = x_dimensions[1][1]
                    #     search_y = tuple(temp_list)
                    search_z = (z_slices[i], z_slices[i+1]) \
                        if z_slices[i] < z_slices[i + 1] else (z_slices[i + 1], z_slices[i])
                    if search_z[1] > x_dimensions[2][1]:
                        logs.error("Unreasonable car z reserve")
                        temp_list = list(search_z)
                        temp_list[1] = x_dimensions[2][1]
                        search_z = tuple(temp_list)
                    X = SearchSpace(np.array([search_x, search_y, search_z]), obstacles)

                    # rrt对象创建
                    child_init = (x_slices[i], y_slices[i], z_slices[i])
                    child_goal = (x_slices[i + 1], y_slices[i + 1], z_slices[i + 1])
                    rrt = RRTStar(X, Q, child_init, child_goal, size, max_samples, r, prc, rewire_count)
                    child_path = rrt.rrt_star()

                    # 子路径获取
                    path += child_path
                path.append(x_goal)
                end_time = time.time()
                logs.debug("it cost {:.2f} s".format(end_time - start_time))

                # 发送数据
                send_data = []
                send_data.append(b'\xfe\xfe')  # start_word
                send_data.append(b'\x01\x00')  # version
                send_data.append(b'\x10')  # trans_to
                send_data.append(b'\x23')  # action_id
                send_data.append(b'\x01')  # block_type
                send_data.append(len(path).to_bytes(1, 'big'))  # num_blocks
                send_data.append(b'\x00\x00')  # err_code
                send_data.append(b'\x00\x00\x00\x91')
                for i in range(0, len(path)):
                    send_data.append(b'\x01')
                    if action['area'] == 'p1' and i != len(path) - 1:
                        send_data.append(b'\x00\x00\x00\x01')
                    elif action['area'] == 'p2' and i != len(path) - 1:
                        send_data.append(b'\x00\x00\x00\x02')
                    elif action['area'] == 'p3' and i != len(path) - 1:
                        send_data.append(b'\x00\x00\x00\x03')
                    elif i == len(path) - 1 and action['action'] == 0:
                        send_data.append(b'\x00\x00\x00\x01')
                    elif i == len(path) - 1 and action['action'] == 1:
                        send_data.append(b'\x00\x00\x00\x02')
                    elif i == len(path) - 1 and action['action'] == 2:
                        send_data.append(b'\x00\x00\x00\x03')
                    else:
                        raise Exception('wrong area !')
                    x, y, z = path[i]
                    send_data.append(struct.pack('!f', x))
                    send_data.append(struct.pack('!f', y))
                    send_data.append(struct.pack('!f', z))
                    send_data.append(b'\x00\x00\x00\x00')
                    send_data.append(b'\x00\x00\x00\x00')
                    send_data.append(b'\x00\x00\x00\x00')
                    send_data.append(b'\x00\x00\x00\x00')
                    send_data.append(b'\x00\x00\x00\x00')
                    send_data.append(b'\x00\x00\x00\x00')
                send_data.append(b'\x00\x00')
                send_datas = b''
                for i in send_data:
                    send_datas += i
                server.send_message(send_datas)

                # 可视化
                if show_env:
                    plot = Plot("rrt_star_3d")
                    plot.plot_tree(X, rrt.trees)
                    if path is not None:
                        plot.plot_path(X, path, size)
                    plot.plot_obstacles(X, obstacles)
                    plot.plot_start(X, x_init)
                    plot.plot_goal(X, x_goal)
                    plot.draw(auto_open=False)

                # 更新环境
                be.step(action)
                if action['action'] == 1:
                    logs.info("*** NO.{} 码垛面 结束！***".format(action['num_F']))
                    logs.info("*** NO.{} 码垛面 开始！***".format(action['num_F'] + 1))
                elif action['action'] == 2:
                    logs.info("*** NO.{} 码垛面 结束！***".format(action['num_F']))
                    logs.info("路径规划结束 !")
                    # break
            else:
                logs.warning("机器人已断开连接 ！")
                break
        # except ConnectionError:
        #     logs.error("网络连接异常中断")
        #     sys.exit(1)
        # except Exception as e:
        #     logs.error(e)
        #     sys.exit(1)
