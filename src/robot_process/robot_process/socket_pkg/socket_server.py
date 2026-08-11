"""机器人 TCP 服务端的单连接、定长收发封装。"""

import socket


class SocketApp:
    """维护一个监听 socket 和一个已接受的机器人连接。"""

    def __init__(self):
        """创建尚未监听的服务端状态；``is_sim`` 可禁用实际网络操作。"""
        self.ip = None
        self.port = None
        self.sock = None
        self.is_sim = False

    def start(self, ip, port):
        """监听指定地址并阻塞等待一个客户端连接。"""
        if self.is_sim:
            return
        self.ip = ip
        self.port = port
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 10)
        s.bind((self.ip, self.port))
        s.listen(5)
        self.sock, addr = s.accept()

    def close(self):
        """关闭当前客户端连接；模拟模式下不执行操作。"""
        if self.is_sim:
            return
        self.sock.close()

    def receive_message(self, byte_size):
        """最多读取 ``byte_size`` 字节，并以十六进制字符串返回。"""
        if self.is_sim:
            return
        received_data = b''
        buffer_size = byte_size
        while len(received_data) < buffer_size:
            data = self.sock.recv(buffer_size - len(received_data))
            if not data:
                break
            received_data += data
        return received_data.hex()

    def send_message(self, data):
        """向当前客户端发送已编码的响应字节。"""
        if self.is_sim:
            return
        self.sock.send(data)

    def command(self):
        """保留的扩展接口，当前命令分派由主节点完成。"""
        pass
