"""断点续传持久化层：原子保存垛型计划 + 进度游标，断电后可恢复。

设计要点：
  - 计划确定性：RobotPosition 由 config_rp 完全决定，重启重建即可，无需存中间状态。
  - JSON 纯文本：plan/cursor 都是 JSON，可直接打开查看，无 pickle 安全风险。
  - 原子替换：写临时文件 → fsync 文件 → os.replace，避免读取到半截 JSON。
    当前未对父目录执行 fsync，因此不承诺最新目录项在突然掉电后一定持久。
  - plan_hash 校验：游标与计划指纹绑定（json sort_keys 确定性），换了垛型则拒绝续传。
  - 先存后发：节点在「发送某抓前」先更新游标，断电宁可漏一抓也不重码（防碰撞）。
"""
import os
import json
import time
import hashlib


def resolve_resume_dir(file_path):
    """定位 <ws>/log/robot_process/resume/，与点云/日志同根。"""
    prefix = os.environ.get('COLCON_PREFIX_PATH', '')
    if prefix:
        ws_root = os.path.dirname(prefix.split(':')[0])
        if os.path.isdir(ws_root):
            return os.path.join(ws_root, 'log', 'robot_process', 'resume')
    parts = os.path.realpath(os.path.abspath(file_path)).split(os.sep)
    for seg in ('src', 'install'):
        if seg in parts:
            ws_root = os.sep.join(parts[:parts.index(seg)])
            return os.path.join(ws_root, 'log', 'robot_process', 'resume')
    return os.path.join(os.path.dirname(os.path.abspath(file_path)), 'resume')


def _atomic_write_bytes(path, data):
    """先同步临时文件再原子替换目标，避免其他读者看到半截文件。"""
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class ResumeStore:
    """以计划文件和进度游标文件维护可恢复的码垛状态。"""

    def __init__(self, base_dir, logger=None):
        """初始化存储路径；目录创建失败只告警，不中断主流程。"""
        self.dir = base_dir
        self.log = logger
        try:
            os.makedirs(self.dir, exist_ok=True)
        except Exception as e:
            self._warn(f"创建续传目录失败：{e}")
        self.plan_path = os.path.join(self.dir, 'resume_plan.json')
        self.cursor_path = os.path.join(self.dir, 'resume_cursor.json')
        self._plan_hash = None

    def _warn(self, msg):
        """保存类失败只告警，绝不抛出（不中断主码垛流程）。"""
        if self.log is not None:
            self.log.warning(f"[断点续传] {msg}")
        else:
            print(f"[断点续传] {msg}")

    @staticmethod
    def plan_hash(config_rp):
        """返回规范化计划 JSON 的短指纹，用于拒绝错配游标。"""
        # sort_keys 保证同一计划跨运行得到稳定序列。
        blob = json.dumps(config_rp, sort_keys=True, ensure_ascii=False).encode('utf-8')
        return hashlib.md5(blob).hexdigest()[:12]

    def save_plan(self, config_rp):
        """机器人连接后存一次，返回 plan_hash（失败返回 None，不抛异常）。"""
        try:
            blob = json.dumps(config_rp, ensure_ascii=False, indent=2).encode('utf-8')
            _atomic_write_bytes(self.plan_path, blob)
            self._plan_hash = self.plan_hash(config_rp)
            return self._plan_hash
        except Exception as e:
            self._warn(f"保存计划失败（不影响码垛）：{e}")
            return None

    def save_cursor(self, rp_idx, box_id, path_id):
        """每抓更新游标（先存后发）。失败只告警不抛异常，不中断码垛。"""
        try:
            data = {
                'rp_idx': int(rp_idx),
                'box_id': int(box_id),
                'path_id': int(path_id),
                'plan_hash': self._plan_hash,
                'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            _atomic_write_bytes(self.cursor_path, json.dumps(data, ensure_ascii=False).encode('utf-8'))
        except Exception as e:
            self._warn(f"保存游标失败（不影响码垛）：{e}")

    def load(self):
        """读回 (config_rp, cursor)；缺文件或 plan_hash 不匹配返回 None。"""
        if not (os.path.isfile(self.plan_path) and os.path.isfile(self.cursor_path)):
            return None
        try:
            with open(self.plan_path, 'r', encoding='utf-8') as f:
                config_rp = json.load(f)
            with open(self.cursor_path, 'r', encoding='utf-8') as f:
                cursor = json.load(f)
        except Exception:
            return None
        h = self.plan_hash(config_rp)
        if cursor.get('plan_hash') != h:
            return None   # 垛型与游标不匹配，拒绝续传
        self._plan_hash = h
        return config_rp, cursor

    def clear(self):
        """全部码垛结束后清除，避免下次误触发续传（失败只告警不抛异常）。"""
        for p in (self.plan_path, self.cursor_path, self.plan_path + '.tmp', self.cursor_path + '.tmp'):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
            except Exception as e:
                self._warn(f"清除续传文件失败：{e}")
