"""原有四条机器人命令的报文编码；不添加新版数据块或改变字段含义。"""

import struct


CMD_GET_PALLET = 'fefe0000010100000000000000000000'
CMD_GET_PER_COUNT = 'fefe0000010200000000000000000000'
CMD_GET_BOX = 'fefe0000010300000000000000000000'
CMD_GET_PATH = 'fefe0000010400000000000000000000'


def _block(values, index=0, class_id=0):
    """每块41字节：index(1)、class_id(4)、9个网络序float(36)。"""
    values = list(values)
    if len(values) > 9:
        raise ValueError('一个旧协议数据块最多包含9个float')
    return struct.pack('!BI9f', index, class_id, *(values + [0.0] * (9 - len(values))))


def _message(blocks):
    # 保留旧版头部中的0x91常量；不是重新定义为动态报文长度。
    return (b'\xfe\xfe\x01\x00\x10\x23\x01' + bytes([len(blocks)])
            + b'\x00\x00\x00\x00\x00\x91' + b''.join(blocks) + b'\x00\x00')


def pallet_message(rp):
    """2块：总箱数/总面数/车宽；有效箱长/宽/高，其余槽位仍为0。"""
    return _message([_block((rp.box_count, rp.face_count, rp.W)),
                     _block((rp.l, rp.w, rp.h))])


def per_count_message(rp):
    """1块：当前面总抓数/字符箱型ASCII编码，不增加单行抓数或角点数据。"""
    return _message([_block((rp.cal_floor_count(), ord(rp.box_type)))])


def box_message(box, box_type):
    """1块：原始配方号/字符箱型；P3单箱=5、双箱=6，其余为实际数量。"""
    number = box['num']
    if box['area'] == 'p3':
        if number not in (1, 2):
            raise ValueError('旧P3配方只定义了1箱=5、2箱=6')
        number += 4
    return _message([_block((number, ord(box_type)))])


def path_message(path, action):
    """每点1块：前三点class_id=P1/P2/P3；末点class_id=继续1/换面2/结束3。"""
    if len(path) != 4:
        raise ValueError('规则路径必须是x0/x1/APP/goal四点，旧协议无分段箱数')
    region = {'p1': 1, 'p2': 2, 'p3': 3}[action['area']]
    terminal = {0: 1, 1: 2, 2: 3}[action['action']]
    return _message([_block(point, index=1,
                            class_id=terminal if i == len(path) - 1 else region)
                     for i, point in enumerate(path)])
