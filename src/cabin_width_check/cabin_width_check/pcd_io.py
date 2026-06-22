"""轻量 PCD 读取（numpy，无 open3d 依赖），支持 ascii / binary（未压缩）。

离线回放用，避免对 open3d 的运行时硬依赖（open3d 仅在可选 ICP 时才需要）。
"""
import numpy as np

_NP_TYPE = {
    ('F', 4): 'f4', ('F', 8): 'f8',
    ('U', 1): 'u1', ('U', 2): 'u2', ('U', 4): 'u4', ('U', 8): 'u8',
    ('I', 1): 'i1', ('I', 2): 'i2', ('I', 4): 'i4', ('I', 8): 'i8',
}


def read_pcd_xyz(path):
    """读 PCD 返回 Nx3 float ndarray（仅取 x/y/z）。"""
    with open(path, 'rb') as f:
        fields, sizes, types, counts, n, data_fmt = [], [], [], [], 0, 'ascii'
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f'PCD 头部异常（未见 DATA）：{path}')
            line = raw.decode('ascii', 'replace').strip()
            if not line or line.startswith('#'):
                continue
            key = line.split()[0].upper()
            if key == 'FIELDS':
                fields = line.split()[1:]
            elif key == 'SIZE':
                sizes = [int(x) for x in line.split()[1:]]
            elif key == 'TYPE':
                types = line.split()[1:]
            elif key == 'COUNT':
                counts = [int(x) for x in line.split()[1:]]
            elif key == 'POINTS':
                n = int(line.split()[1])
            elif key == 'DATA':
                data_fmt = line.split()[1].lower()
                break
        if not counts:
            counts = [1] * len(fields)

        dt = []
        for fn, sz, tp, ct in zip(fields, sizes, types, counts):
            base = _NP_TYPE.get((tp.upper(), sz), 'f4')
            dt.append((fn, base) if ct == 1 else (fn, base, (ct,)))
        dtype = np.dtype(dt)

        if data_fmt == 'binary':
            arr = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype, count=n)
        elif data_fmt == 'ascii':
            arr = np.loadtxt(f, dtype=dtype, max_rows=n)
        else:
            raise ValueError(f'不支持的 PCD DATA 格式：{data_fmt}（仅 ascii/binary）')

    xyz = np.column_stack([arr['x'], arr['y'], arr['z']]).astype(float)
    mask = np.isfinite(xyz).all(axis=1)
    return xyz[mask]
