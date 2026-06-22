python -m grpc_tools.protoc -I. --python_out=.. --grpc_python_out=.. answer.proto
python -m grpc_tools.protoc -I. --python_out=.. --grpc_python_out=.. condition.proto
python -m grpc_tools.protoc -I. --python_out=.. --grpc_python_out=.. issuer.proto