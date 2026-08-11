"""接收规划器 Issue 请求的轻量 gRPC 服务封装。"""

import logging
from concurrent import futures
import grpc

from grpc_pkg.interface import issuer_pb2 as issuer
from grpc_pkg.interface import issuer_pb2_grpc as issuer_grpc


class Issuer(issuer_grpc.IssuerServicer):
    """把 protobuf 请求交给主节点回调，并返回其成功标志和消息。"""

    def __init__(self, callback):
        """保存订单解析回调。"""
        super().__init__()
        self.__callback = callback
        self.data = None

    def Issue(self, request, context):
        """处理一次规划器下发；回调签名为 ``callback(request)``。"""
        data = request
        if self.__callback is not None:
            result, message = self.__callback(data)
        reply = issuer.IssueReply(result=result, message=message)
        return reply


class GrpcServer(object):
    """管理 gRPC 线程池、端口绑定和服务生命周期。"""

    def __init__(self, port, callback):
        """在所有网卡的指定端口注册 Issuer 服务。"""
        self.__server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        issuer_grpc.add_IssuerServicer_to_server(Issuer(callback), self.__server)
        self.__server.add_insecure_port('0.0.0.0:{}'.format(port))

    def run(self):
        """非阻塞启动 gRPC 服务。"""
        self.__server.start()

    def stop(self):
        """立即停止 gRPC 服务，不设置优雅退出等待时间。"""
        self.__server.stop(None)

