import socket


class SocketApp:

    def __init__(self):
        self.ip = None
        self.port = None
        self.sock = None
        self.is_sim = False

    def start(self, ip, port):
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
        if self.is_sim:
            return
        self.sock.close()

    def receive_message(self, byte_size):
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
        if self.is_sim:
            return
        self.sock.send(data)

    def command(self):
        pass
