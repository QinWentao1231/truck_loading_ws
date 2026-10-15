import logging
from concurrent import futures
import grpc

from .interface import issuer_pb2 as issuer
from .interface import issuer_pb2_grpc as issuer_grpc


class Issuer(issuer_grpc.IssuerServicer):

    def __init__(self, callback):
        super().__init__()
        self.__callback = callback
        self.data = None

    def Issue(self, request, context):
        # request为IssueRequest类型
        data = request  # TODO:获取后处理为需要的格式
        if self.__callback is not None:
            result, message = self.__callback(data)
        reply = issuer.IssueReply(result=result, message=message)
        return reply


class GrpcServer(object):

    def __init__(self, port, callback):
        self.__server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        issuer_grpc.add_IssuerServicer_to_server(Issuer(callback), self.__server)
        if not self.__server.add_insecure_port('0.0.0.0:{}'.format(port)):
            raise RuntimeError('无法绑定规划器gRPC端口 {}'.format(port))

    def run(self):
        self.__server.start()

    def stop(self):
        self.__server.stop(None)

