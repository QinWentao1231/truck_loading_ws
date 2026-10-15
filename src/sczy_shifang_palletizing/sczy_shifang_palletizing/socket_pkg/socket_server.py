"""机器人TCP传输层；保持旧报文，增加可中断等待供ROS2正常退出。"""

import socket
import threading


class SocketApp:
    def __init__(self):
        self.ip = None
        self.port = None
        self.sock = None
        self.listener = None
        self.is_sim = False
        self._closed = threading.Event()
        self._lock = threading.Lock()

    def start(self, ip, port):
        if self.is_sim:
            return True
        self.ip, self.port = ip, port
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with self._lock:
            if self._closed.is_set():
                listener.close()
                return False
            self.listener = listener
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.settimeout(0.25)
        listener.bind((ip, port))
        listener.listen(5)
        while not self._closed.is_set():
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._closed.is_set():
                    return False
                raise
            with self._lock:
                if self._closed.is_set():
                    connection.close()
                    return False
                self.sock = connection
                connection.settimeout(0.25)
            return True
        return False

    def close(self):
        self._closed.set()
        with self._lock:
            sockets = (self.sock, self.listener)
            self.sock = self.listener = None
        for connection in sockets:
            if connection is not None:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                connection.close()

    def receive_message(self, byte_size):
        if self.is_sim:
            return ''
        connection = self.sock
        received = b''
        while len(received) < byte_size and not self._closed.is_set():
            try:
                data = connection.recv(byte_size - len(received))
            except socket.timeout:
                continue
            except OSError:
                if self._closed.is_set():
                    return ''
                raise
            if not data:
                break
            received += data
        return received.hex() if not self._closed.is_set() else ''

    def send_message(self, data):
        if not self.is_sim:
            if self._closed.is_set():
                raise ConnectionError('机器人连接已关闭')
            self.sock.sendall(data)
