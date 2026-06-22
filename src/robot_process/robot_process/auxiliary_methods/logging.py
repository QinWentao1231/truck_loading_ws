import os
import datetime
import colorlog
import logging
import logging.handlers


def _resolve_log_dir():
    """
    从文件路径推导 <ws_root>/log/<pkg_name>/，兼容两种启动方式：
      - VSCode 直接运行：<ws>/src/<pkg>/<pkg>/...
      - ros2 run 安装路径：<ws>/install/<pkg>/lib/<pkg>/...
    通过查找路径中的 src 或 install 段定位工作空间根，不依赖固定层级。
    """
    real = os.path.realpath(os.path.abspath(__file__))
    parts = real.split(os.sep)
    for i, part in enumerate(parts):
        if part in ('src', 'install'):
            ws_root = os.sep.join(parts[:i]) or os.sep
            pkg_name = parts[i + 1] if i + 1 < len(parts) else 'robot_process'
            return os.path.join(ws_root, 'log', pkg_name)
    # 兜底：文件所在目录上4级
    return os.path.join(os.path.dirname(real), '..', '..', '..', '..', 'log')


_log_dir = _resolve_log_dir()
os.makedirs(_log_dir, exist_ok=True)
_default_filepath = os.path.join(_log_dir, 'log_' + datetime.datetime.now().strftime('%Y%m%d') + '.log')


# >>> 定义Logger类
class Logger(object):
    level_relations = {
        'debug': logging.DEBUG, 'info': logging.INFO,
        'warning': logging.WARNING, 'error': logging.ERROR, 'crit': logging.CRITICAL
    }
    log_colors_config = {
        'DEBUG': 'white', 'INFO': 'green',
        'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'red',
    }

    def __init__(self, filename=_default_filepath, level='info', when='D', back_count=3):
        self.logger = logging.getLogger(filename)
        sh_format = colorlog.ColoredFormatter(
            fmt='%(log_color)s%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            log_colors=self.log_colors_config)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(sh_format)
        th = logging.handlers.TimedRotatingFileHandler(
            filename=filename, when=when, backupCount=back_count, encoding='utf-8')
        th_format = logging.Formatter(
            '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        th.setFormatter(th_format)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)

    def get_log(self):
        return self.logger
