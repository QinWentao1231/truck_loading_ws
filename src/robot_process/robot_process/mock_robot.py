"""
mock_robot.py — 交互式模拟机器人 TCP socket 客户端。

用法：
    python mock_robot.py [--host 127.0.0.1] [--port 8001]

启动后先选择连接，再执行各指令，支持断开重连。
"""

import socket
import struct
import argparse

CMD_GET_PALLET    = bytes.fromhex('fefe0000010100000000000000000000')
CMD_GET_PER_COUNT = bytes.fromhex('fefe0000010200000000000000000000')
CMD_GET_BOX       = bytes.fromhex('fefe0000010300000000000000000000')
CMD_GET_PATH      = bytes.fromhex('fefe0000010400000000000000000000')
CMD_CHK_PATH      = bytes.fromhex('fefe0000010500000000000000000000')
CMD_STACKING      = bytes.fromhex('fefe0000010600000000000000000000')

_HEADER_SIZE = 14   # 7(_MSG_HEADER) + 1(num_blocks) + 6(err+reserved+0x91)
_BLOCK_SIZE  = 41
_ACT_LABEL   = {1: 'normal', 2: 'face_end', 3: 'block_end', 4: 'all_end'}
_AREA_LABEL  = {1: 'p1', 2: 'p2', 3: 'p3'}

_STACK_LABEL = {1: '有结果', 2: '异常'}

MENU = """
┌──────────────────────────────────────────┐
│  0. 连接服务端                            │
│  1. get_pallet    请求垛型总体信息         │
│  2. get_per_count 请求本面箱数             │
│  3. get_box       请求来料配方             │
│  4. get_path      请求放置路径             │
│  5. chk_path      触发路径预取             │
│  6. stacking      触发堆叠检测             │
│  9. 断开连接                              │
│  q. 退出                                 │
└──────────────────────────────────────────┘
"""


def recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("服务端断开连接")
        buf += chunk
    return buf


def recv_response(sock):
    header = recv_exact(sock, _HEADER_SIZE)
    num_blocks = header[7]
    payload = recv_exact(sock, num_blocks * _BLOCK_SIZE + 2)
    blocks = [payload[i * _BLOCK_SIZE:(i + 1) * _BLOCK_SIZE] for i in range(num_blocks)]
    return num_blocks, blocks


def parse_floats(block, count=9):
    return struct.unpack_from('!' + 'f' * count, block, 5)


def parse_path_block(block):
    flag = struct.unpack_from('!I', block, 1)[0]
    x, y, z = struct.unpack_from('!fff', block, 5)
    return flag, x, y, z


def handle_get_pallet(sock):
    sock.sendall(CMD_GET_PALLET)
    _, blocks = recv_response(sock)
    f0 = parse_floats(blocks[0], 3)
    f1 = parse_floats(blocks[1], 3)
    print(f"  总箱数={int(f0[0])}  码垛面数={int(f0[1])}  车宽={f0[2]:.0f}mm")
    print(f"  箱尺寸 L={f1[0]:.0f} W={f1[1]:.0f} H={f1[2]:.0f}")


def handle_get_per_count(sock):
    sock.sendall(CMD_GET_PER_COUNT)
    _, blocks = recv_response(sock)
    f = parse_floats(blocks[0], 3)
    print(f"  本面箱数={int(f[0])}  箱型={int(f[1])}  每行抓数={int(f[2])}")


def handle_get_box(sock):
    sock.sendall(CMD_GET_BOX)
    _, blocks = recv_response(sock)
    f = parse_floats(blocks[0], 3)
    print(f"  配方号={int(f[0])}  箱型={int(f[1])}  区域={int(f[2])}")


def handle_get_path(sock):
    sock.sendall(CMD_GET_PATH)
    n, blocks = recv_response(sock)
    path_blocks = blocks[:-1]
    info_block  = blocks[-1]
    print(f"  共 {n-1} 个路径点：")
    for i, blk in enumerate(path_blocks):
        _, x, y, z = parse_path_block(blk)
        print(f"    [{i}] x={x:8.1f}  y={y:8.1f}  z={z:8.1f}")
    area_f, act_f, num0_f = parse_floats(info_block, 3)
    area_s = _AREA_LABEL.get(int(area_f), int(area_f))
    act_s  = _ACT_LABEL.get(int(act_f),  int(act_f))
    print(f"  info: area={area_s}  action={act_s}  num[0]={int(num0_f)}")


def handle_chk_path(sock):
    sock.sendall(CMD_CHK_PATH)
    print("  chk_path 已发送（服务端无响应体，后续路径由服务端批量推送）")


def handle_stacking(sock):
    sock.sendall(CMD_STACKING)
    _, blocks = recv_response(sock)
    f = parse_floats(blocks[0], 3)
    status = int(f[0])
    label = _STACK_LABEL.get(status, status)
    print(f"  status={status}（{label}）  理论宽度={f[1]:.1f}mm  测量宽度={f[2]:.1f}mm")


HANDLERS = {
    '1': ('get_pallet',    handle_get_pallet),
    '2': ('get_per_count', handle_get_per_count),
    '3': ('get_box',       handle_get_box),
    '4': ('get_path',      handle_get_path),
    '5': ('chk_path',      handle_chk_path),
    '6': ('stacking',      handle_stacking),
}


def run(host, port):
    sock = None

    while True:
        status = '已连接' if sock else '未连接'
        print(f"\n[状态: {status}]", end='')
        print(MENU, end='')
        try:
            choice = input("选择指令 > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break

        if choice == 'q':
            if sock:
                sock.close()
                print("  已断开连接")
            break

        elif choice == '0':
            if sock:
                print("  已有连接，请先断开（选9）")
                continue
            try:
                print(f"  正在连接 {host}:{port} ...")
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((host, port))
                sock = s
                print(f"  连接成功")
            except Exception as e:
                print(f"  [错误] 连接失败: {e}")

        elif choice == '9':
            if not sock:
                print("  当前未连接")
                continue
            sock.close()
            sock = None
            print("  已断开连接")

        elif choice in HANDLERS:
            if not sock:
                print("  请先连接（选0）")
                continue
            name, handler = HANDLERS[choice]
            print(f"→ 发送 {name}")
            try:
                handler(sock)
            except ConnectionError as e:
                print(f"  [错误] {e}")
                sock = None
            except Exception as e:
                print(f"  [错误] {e}")

        else:
            print("  无效选项，请重新输入")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='交互式模拟机器人 socket 客户端')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8001)
    args = ap.parse_args()
    run(args.host, args.port)
