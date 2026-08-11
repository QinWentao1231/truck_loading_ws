"""旧版角点 TCP 客户端及固定响应帧解析工具。"""

import socket
import struct
import math


def create_tcp_client(server_ip, server_port, message_bytes):
    """发送一次请求并解析角点响应；失败时返回零偏移和零角点数。"""
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    dis_y, dis_x, count = 0.0, 0.0, 0
    try:
        # 连接服务器
        client_socket.connect((server_ip, server_port))
        # 发送字节流
        client_socket.sendall(message_bytes)
        # 接收服务器的响应
        response = client_socket.recv(1024)
        # 解析接收到的数据
        dis_y, dis_x, count = parse_data(response)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 关闭连接
        client_socket.close()
    return dis_y, dis_x, count


def parse_data(data):
    """按旧版固定偏移解析最多四个角点，返回 ``(dis_y, dis_x, count)``。

    角点坐标使用网络字节序 float32；四角点时返回 corner1 与 corner4 的
    X/Y 轴差，两角点时 ``dis_y`` 返回 corner1 到 corner2 的三维距离。
    """
    count = data[40]
    corner1x = struct.unpack('!f', data[41:45])
    corner1y = struct.unpack('!f', data[45:49])
    corner1z = struct.unpack('!f', data[49:53])
    corner2x = struct.unpack('!f', data[65:69])
    corner2y = struct.unpack('!f', data[69:73])
    corner2z = struct.unpack('!f', data[73:77])
    corner3x = struct.unpack('!f', data[89:93])
    corner3y = struct.unpack('!f', data[93:97])
    corner3z = struct.unpack('!f', data[97:101])
    corner4x = struct.unpack('!f', data[113:117])
    corner4y = struct.unpack('!f', data[117:121])
    corner4z = struct.unpack('!f', data[121:125])
    if count == 4:
        dis_y = abs(corner1y[0] - corner4y[0])
        dis_x = abs(corner1x[0] - corner4x[0])
    else:
        dis_y = math.sqrt((corner1x[0]-corner2x[0])**2 + (corner1y[0]-corner2y[0])**2 + (corner1z[0]-corner2z[0])**2)
        dis_x = 0

    return dis_y, dis_x, count
