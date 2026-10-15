import socket
import struct

def create_tcp_client(server_ip, server_port, message_bytes):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # 连接服务器
        client_socket.connect((server_ip, server_port))
        # 发送字节流
        client_socket.sendall(message_bytes)
        # 接收服务器的响应
        response = client_socket.recv(1024)
        # 解析接收到的数据
        var1= parse_data(response)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 关闭连接
        client_socket.close()
    return var1

def parse_data(data):
    # 假设接收到的数据是一串16进制字节流，例如：b'\x01\x02\x03\x04\x05\x06\x07\x08'
    # 并且每个变量占用固定的字节数，例如每个变量占用2个字节
    corner = data[4:8]  # 前两个字节

    # 将字节转换为整数或其他需要的类型
    var1 = struct.unpack('!f', data[40:44])

    return var1