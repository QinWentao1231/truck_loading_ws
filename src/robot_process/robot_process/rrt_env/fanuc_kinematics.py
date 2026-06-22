"""
Fanuc R-1000iA/120F 运动学辅助模块
=====================================
用途：
  在路径点发送给机器人控制器之前，在 PC 侧做：
    1. 逆运动学可达性验证
    2. 关节限位检查
    3. 奇异性检测（腕部/肘部/肩部）

坐标系说明：
  robot_process_node.py 中的路径点 (x, y, z) 属于机器人的 User Frame（作业坐标系）。
  运动学计算在 Base Frame（机器人基坐标系）中进行。
  两者之间的变换矩阵 T_uf 由 make_uf_transform() 根据配置生成。
  若未配置 kin_uf_offset，则假设 User Frame = Base Frame（需安装时校准）。

DH 参数说明：
  使用标准 DH 约定（Craig 版本）：
    T_i = Rz(theta_i) · Tz(d_i) · Tx(a_i) · Rx(alpha_i)
  参数基于 Fanuc R-1000iA/120F 公开规格推导，*使用前必须与实际机器人 MASTERING 数据核对*。
  可通过 config.json 中 kin_dh_params 字段覆盖（6×4 列表）。

依赖：numpy（已在 requirements.txt 中）
"""

import numpy as np
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# 默认 DH 参数  [a(mm), alpha(rad), d(mm), theta_offset(rad)]
# Fanuc R-1000iA/120F 近似值，需现场核对
# ---------------------------------------------------------------------------
_DH_DEFAULT = np.array([
    [  320.0, -np.pi / 2,  670.0,  0.0        ],   # J1 腰部
    [ 1075.0,  0.0,          0.0, -np.pi / 2  ],   # J2 大臂（含 -90° 零位偏移）
    [  250.0, -np.pi / 2,    0.0,  0.0        ],   # J3 小臂
    [    0.0,  np.pi / 2, 1035.0,  0.0        ],   # J4 腕旋转
    [    0.0, -np.pi / 2,    0.0,  0.0        ],   # J5 腕弯曲
    [    0.0,  0.0,         195.0,  0.0        ],   # J6 工具旋转（含法兰）
], dtype=float)

# 关节限位 [min, max] (rad) — R-1000iA/120F 典型值
_JLIM_DEFAULT = np.deg2rad([
    [-185.0,  185.0],   # J1
    [ -90.0,   55.0],   # J2
    [ -70.0,   55.0],   # J3
    [-185.0,  185.0],   # J4
    [-125.0,  125.0],   # J5
    [-360.0,  360.0],   # J6
])

# 腕部奇异阈值：J5 绝对值低于此值（rad）时报警
_WRIST_THRESH = np.deg2rad(5.0)

# 可操作度阈值：Yoshikawa 指标低于此值时报警
# 该值对 DH 参数敏感，初次使用建议在正常工作姿态下采集基准值后调整
_MANIP_THRESH = 1e8


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def make_uf_transform(offset_xyzrxryrz: Optional[List[float]] = None) -> np.ndarray:
    """
    根据 [x, y, z, rx_deg, ry_deg, rz_deg] 构造 User Frame 相对 Base Frame 的
    4×4 齐次变换矩阵（旋转顺序：ZYX / RPY）。

    offset_xyzrxryrz 为 None 时返回单位矩阵（User Frame = Base Frame）。

    示例 config.json 配置：
        "kin_uf_offset": [1200.0, 1100.0, 0.0, 0.0, 0.0, 0.0]
    含义：作业坐标系原点在 Base Frame 中 x=1200mm, y=1100mm 处，无旋转。
    """
    if offset_xyzrxryrz is None:
        return np.eye(4)
    x, y, z, rx, ry, rz = offset_xyzrxryrz
    rx, ry, rz = np.deg2rad(rx), np.deg2rad(ry), np.deg2rad(rz)
    # ZYX 欧拉角 → 旋转矩阵
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def default_tool_rotation() -> np.ndarray:
    """
    默认夹爪姿态旋转矩阵（作业坐标系中）：工具 Z 轴朝下，X 轴朝机器人前方。
    对应 Fanuc XYZWPR 约 W=-180°, P=0°, R=0°。
    可根据实际工具安装方向在 config.json 中替换。
    """
    return np.array([
        [ 1.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0, -1.0],
    ], dtype=float)


# ---------------------------------------------------------------------------
# 核心类
# ---------------------------------------------------------------------------

class FanucKinematics:
    """
    Fanuc R-1000iA/120F 运动学辅助（正运动学 / 数值逆运动学 / 奇异性检测）

    使用示例：
        import json, numpy as np
        from rrt_env.fanuc_kinematics import FanucKinematics, make_uf_transform

        with open('config.json') as f:
            cfg = json.load(f)
        T_uf = make_uf_transform(cfg.get('kin_uf_offset'))
        kin = FanucKinematics(uf_transform=T_uf)

        results = kin.validate_path([x0, x_init, x_app, x_goal])
        for r in results:
            if r['near_singularity']:
                print(r['msg'])
    """

    def __init__(
        self,
        dh: Optional[np.ndarray] = None,
        jlim: Optional[np.ndarray] = None,
        uf_transform: Optional[np.ndarray] = None,
    ):
        self._dh   = dh   if dh   is not None else _DH_DEFAULT.copy()
        self._jlim = jlim if jlim is not None else _JLIM_DEFAULT.copy()
        self._T_uf = uf_transform if uf_transform is not None else np.eye(4)
        self._T_uf_inv = np.linalg.inv(self._T_uf)

    # ── 正运动学 ──────────────────────────────────────────────────────────

    def fk(self, q: np.ndarray) -> np.ndarray:
        """正运动学：q (6,) rad → 末端 4×4 变换矩阵（Base Frame）"""
        T = np.eye(4)
        for i in range(6):
            a, alpha, d, th_off = self._dh[i]
            theta = q[i] + th_off
            ct, st = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(alpha), np.sin(alpha)
            T = T @ np.array([
                [ct, -st * ca,  st * sa, a * ct],
                [st,  ct * ca, -ct * sa, a * st],
                [ 0,       sa,       ca,      d],
                [ 0,        0,        0,      1],
            ])
        return T

    # ── 雅可比矩阵与可操作度 ──────────────────────────────────────────────

    def _jacobian(self, q: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """数值雅可比矩阵 (6×6)，位置部分单位 mm，旋转部分单位 rad"""
        J = np.zeros((6, 6))
        T0 = self.fk(q)
        p0 = T0[:3, 3]
        for i in range(6):
            dq = np.zeros(6)
            dq[i] = eps
            T1 = self.fk(q + dq)
            J[:3, i] = (T1[:3, 3] - p0) / eps
            dR = T1[:3, :3] @ T0[:3, :3].T
            J[3, i] = (dR[2, 1] - dR[1, 2]) / (2 * eps)
            J[4, i] = (dR[0, 2] - dR[2, 0]) / (2 * eps)
            J[5, i] = (dR[1, 0] - dR[0, 1]) / (2 * eps)
        return J

    def manipulability(self, q: np.ndarray) -> float:
        """Yoshikawa 可操作度 w = sqrt(det(J·Jᵀ))，值越大越远离奇异"""
        J = self._jacobian(q)
        return float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))

    # ── 奇异性类型识别 ────────────────────────────────────────────────────

    def singularity_types(self, q: np.ndarray) -> List[str]:
        """
        返回当前关节配置下的奇异类型列表：
          'wrist'    : J5 ≈ 0，腕部奇异（J4/J6 轴对齐，最常见）
          'elbow'    : J3 接近下限，小臂几乎与大臂对齐
          'shoulder' : 腕部中心接近 J1 轴（需改变 J1 位置才能消除）
        """
        types = []
        # 腕部奇异
        if abs(q[4]) < _WRIST_THRESH:
            types.append('wrist')
        # 肘部（J3 接近极限 -70°）
        if q[2] < self._jlim[2, 0] + np.deg2rad(8):
            types.append('elbow')
        # 肩部：腕部中心（J4 原点）的 XY 分量接近零
        T_j4 = np.eye(4)
        for i in range(4):
            a, alpha, d, th_off = self._dh[i]
            theta = q[i] + th_off
            ct, st = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(alpha), np.sin(alpha)
            T_j4 = T_j4 @ np.array([
                [ct, -st * ca,  st * sa, a * ct],
                [st,  ct * ca, -ct * sa, a * st],
                [ 0,       sa,       ca,      d],
                [ 0,        0,        0,      1],
            ])
        wc = T_j4[:2, 3]   # wrist center XY in base frame
        if float(np.linalg.norm(wc)) < 30.0:
            types.append('shoulder')
        return types

    # ── 数值逆运动学 (DLS) ────────────────────────────────────────────────

    def ik(
        self,
        T_target: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        max_iter: int = 400,
        pos_tol: float = 1.0,        # mm
        orient_tol: float = 0.015,   # rad (~0.86°)
        damp: float = 0.05,
    ) -> Optional[np.ndarray]:
        """
        阻尼最小二乘逆运动学（Base Frame）。

        T_target : 目标 4×4 齐次变换矩阵
        q_init   : 初始关节角猜测 (rad)，None 时用竖直姿态（适合码垛工况）
        返回     : 收敛且在限位内的关节角 (rad)，否则返回 None
        """
        if q_init is None:
            # 码垛工况常见初始猜测：大臂斜前伸、腕部朝下
            q = np.deg2rad([0.0, -20.0, -20.0, 0.0, -90.0, 0.0])
        else:
            q = q_init.copy()
        q = np.clip(q, self._jlim[:, 0], self._jlim[:, 1])

        for _ in range(max_iter):
            T = self.fk(q)
            dp = T_target[:3, 3] - T[:3, 3]

            R_err = T_target[:3, :3] @ T[:3, :3].T
            cos_th = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
            th = float(np.arccos(cos_th))
            if th < 1e-9:
                dr = np.zeros(3)
            else:
                dr = (th / (2.0 * np.sin(th))) * np.array([
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1],
                ])

            if np.linalg.norm(dp) < pos_tol and np.linalg.norm(dr) < orient_tol:
                in_lim = np.all((q >= self._jlim[:, 0]) & (q <= self._jlim[:, 1]))
                return q if in_lim else None

            e = np.concatenate([dp, dr])
            J = self._jacobian(q)
            JJT = J @ J.T
            dq = J.T @ np.linalg.solve(JJT + damp ** 2 * np.eye(6), e)
            q = np.clip(q + 0.5 * dq, self._jlim[:, 0], self._jlim[:, 1])

        return None   # 未收敛

    def _ik_multi_seed(
        self,
        T_target: np.ndarray,
        q_seed: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """尝试多个初始猜测，返回第一个成功的逆解（提升复杂位姿下的收敛率）"""
        seeds = []
        if q_seed is not None:
            seeds.append(q_seed)
        # 覆盖高层、低层、左侧、右侧等常见工况的初始猜测
        seeds += [
            np.deg2rad([ 0,  -20, -20,   0, -90,   0]),   # 中部朝下
            np.deg2rad([ 0,  -40, -30,   0, -90,   0]),   # 大臂较平
            np.deg2rad([20,  -20, -20,   0, -90,   0]),   # J1 偏右
            np.deg2rad([-20, -20, -20,   0, -90,   0]),   # J1 偏左
            np.deg2rad([ 0,   10, -50,   0, -90,   0]),   # 高层前伸
        ]
        for q0 in seeds:
            result = self.ik(T_target, q0)
            if result is not None:
                return result
        return None

    # ── 单点验证 ──────────────────────────────────────────────────────────

    def check_pose(
        self,
        xyz_uf: Tuple[float, float, float],
        R_tool: Optional[np.ndarray] = None,
        q_seed: Optional[np.ndarray] = None,
    ) -> dict:
        """
        验证作业坐标系中一个位置点是否可达，返回诊断信息。

        xyz_uf : (x, y, z) mm，作业坐标系（与 robot_process_node.py 中路径点同一坐标系）
        R_tool : 末端工具姿态旋转矩阵（作业坐标系），None 时用夹爪朝下默认值
        q_seed : IK 初始猜测关节角，通常用前一个点的 IK 解

        返回 dict:
            reachable        (bool)   : 是否可达（有满足关节限位的逆解）
            q                (ndarray|None) : 逆解关节角 (rad)
            manipulability   (float)  : Yoshikawa 可操作度
            near_singularity (bool)   : 是否接近奇异
            singularity_type (list)   : 奇异类型列表
            msg              (str)    : 诊断摘要
        """
        if R_tool is None:
            R_tool = default_tool_rotation()

        # 组装 User Frame 中的目标变换矩阵，再转换到 Base Frame
        T_uf_pose = np.eye(4)
        T_uf_pose[:3, :3] = R_tool
        T_uf_pose[:3, 3] = xyz_uf
        T_base = self._T_uf @ T_uf_pose

        q = self._ik_multi_seed(T_base, q_seed)
        if q is None:
            return {
                'reachable': False, 'q': None,
                'manipulability': 0.0, 'near_singularity': True,
                'singularity_type': [], 'msg': '逆解失败（超出工作空间或关节限位）',
            }

        m = self.manipulability(q)
        s_types = self.singularity_types(q)
        near = m < _MANIP_THRESH or bool(s_types)
        if s_types:
            msg = '接近奇异：' + ' + '.join(s_types) + f'  可操作度={m:.2e}'
        elif near:
            msg = f'可操作度偏低（{m:.2e}），建议调整路径点'
        else:
            msg = f'正常  可操作度={m:.2e}'
        return {
            'reachable': True, 'q': q,
            'manipulability': m, 'near_singularity': near,
            'singularity_type': s_types, 'msg': msg,
        }

    # ── 路径点批量验证 ────────────────────────────────────────────────────

    def validate_path(
        self,
        waypoints: List[Tuple[float, float, float]],
        R_tool: Optional[np.ndarray] = None,
    ) -> List[dict]:
        """
        批量验证路径点列表。
        前一个点的 IK 解作为下一个点的种子，保证多解环境中关节角的连续性。

        waypoints : [(x0,y0,z0), (x1,y1,z1), ...]  作业坐标系，mm
        返回      : 每个点的 check_pose 结果列表（含 'pos' 字段）
        """
        results = []
        q_prev = None
        for wp in waypoints:
            res = self.check_pose(wp, R_tool, q_prev)
            res['pos'] = wp
            results.append(res)
            if res['reachable']:
                q_prev = res['q']
        return results

    # ── 实用工具 ──────────────────────────────────────────────────────────

    def q_deg(self, q: np.ndarray) -> np.ndarray:
        """关节角 rad → deg，方便日志输出"""
        return np.round(np.rad2deg(q), 1)

    def joint_limit_margin_deg(self, q: np.ndarray) -> np.ndarray:
        """返回每个关节距离最近限位的余量（deg），负值表示已超限"""
        margin = np.minimum(q - self._jlim[:, 0], self._jlim[:, 1] - q)
        return np.round(np.rad2deg(margin), 1)

    # ── 接近点自动规避 ────────────────────────────────────────────────────

    def resolve_approach_point(
        self,
        xyz_uf: Tuple[float, float, float],
        s_types: List[str],
        q_seed: Optional[np.ndarray] = None,
        R_tool: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[dict]]:
        """
        对有奇异/限位风险的接近点 (x_app) 做自动微调，返回第一个可行替换点。

        调整策略（与奇异类型绑定，优先级从上到下）：

          wrist（J5≈0）
            → 首先尝试翻转腕部构型（同一 Cartesian 位置，J4+180°/J6-180°）
            → 其次逐步抬高 z（每步 +30mm，最多 5 步）
            原因：z 升高使肩/肘角度改变，腕中心位置偏移，J5 脱离 0 附近

          elbow（J3 接近极限）
            → 逐步增大 x（拉向机器人，每步 +50mm，最多 3 步）
            → 再试 z 同步抬高
            原因：减小水平伸展量使 J3 回到正常范围

          shoulder（腕中心接近 J1 轴）
            → 同步增大 x 和 z（±50/80mm 组合）
            原因：需要改变整体手臂配置

          无具体类型但 near_singularity=True（可操作度偏低 / J4 限位）
            → 先用翻转腕部种子重解同一位置（不改 Cartesian，换关节配置）
            → 再小幅 x/z 扰动
            原因：J4 限位往往通过替代 IK 解解决，无需改动路径点

        参数：
          xyz_uf  : 当前接近点（作业坐标系，mm）
          s_types : check_pose 返回的 singularity_type 列表
          q_seed  : 前一路径点的 IK 解，用于保证关节连续性
          R_tool  : 工具姿态，None 时用默认朝下

        返回：
          (adjusted_xyz, check_result)  成功时；
          (None, None)                  无法规避时
        """
        x, y, z = xyz_uf

        # ── 翻转腕部（J4+180°, J6-180°）种子，不改变 Cartesian 位置 ──────
        def _try_flipped_seed(xyz):
            seeds_to_flip = [q_seed] if q_seed is not None else []
            seeds_to_flip += [
                np.deg2rad([ 0,  -20, -20,   0, -90,   0]),
                np.deg2rad([ 0,  -40, -30,   0, -90,   0]),
                np.deg2rad([20,  -20, -20,   0, -90,   0]),
            ]
            for q0 in seeds_to_flip:
                q_flip = q0.copy()
                q_flip[3] += np.pi   # J4 +180°
                q_flip[5] -= np.pi   # J6 -180°（保持工具姿态不变）
                q_flip = np.clip(q_flip, self._jlim[:, 0], self._jlim[:, 1])
                res = self.check_pose(xyz, R_tool, q_flip)
                if res['reachable'] and not res['near_singularity']:
                    return xyz, res
            return None, None

        # ── 生成 Cartesian 扰动候选列表 (dx, dy, dz) ─────────────────────
        candidates: List[Tuple[float, float, float]] = []

        if 'wrist' in s_types:
            # 主策略：抬高 z
            candidates += [(0.0, 0.0, float(dz)) for dz in [30, 60, 90, 120, 150]]
            # 辅助：x 同步后移（车厢深处目标）
            candidates += [(30.0, 0.0, 30.0), (50.0, 0.0, 50.0)]

        if 'elbow' in s_types:
            # 主策略：增大 x（拉向机器人）
            candidates += [(float(dx), 0.0, 0.0) for dx in [50, 80, 120]]
            # 辅助：x 和 z 同时调整
            candidates += [(50.0, 0.0, 30.0), (80.0, 0.0, 50.0)]

        if 'shoulder' in s_types:
            candidates += [
                (50.0, 0.0, 50.0), (80.0, 0.0, 80.0),
                (100.0, 0.0, 50.0), (50.0, 0.0, 100.0),
            ]

        # 无具体奇异类型：小幅扰动兜底
        if not s_types:
            candidates += [
                (0.0, 0.0, 50.0), (30.0, 0.0, 30.0),
                (50.0, 0.0, 0.0), (50.0, 0.0, 50.0),
            ]

        # ── 先尝试翻转腕部（不动 Cartesian 点）────────────────────────────
        if 'wrist' in s_types or not s_types:
            adj_xyz, adj_res = _try_flipped_seed(xyz_uf)
            if adj_xyz is not None:
                return adj_xyz, adj_res

        # ── 再逐个试扰动候选 ─────────────────────────────────────────────
        for dx, dy, dz in candidates:
            new_xyz = (x + dx, y + dy, z + dz)
            res = self.check_pose(new_xyz, R_tool, q_seed)
            if res['reachable'] and not res['near_singularity']:
                return new_xyz, res

        return None, None
