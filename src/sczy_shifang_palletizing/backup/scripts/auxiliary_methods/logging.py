import os
import datetime
import logging
import colorlog
import logging

# >>> 定义Logger类
class Logger(object):
    # 定义类属性
    level_relations = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR,
                       'crit': logging.CRITICAL}  # 日志级别关系映射
    log_colors_config = {
        'DEBUG': 'white',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red',
    }
    # 获取当前工作目录
    current_directory = os.getcwd()

    # 获取上级目录
    parent_directory = os.path.dirname(current_directory)

    # 使用函数exists()对文件存在与否进行判断，存在为True,不存在为False.
    if os.path.exists('logs'):  # 文件存在
        filepath = os.getcwd() + '/logs/log_' + datetime.datetime.now().strftime('%Y%m%d') + '.log'
    else:
        os.makedirs("./logs")  # 创建文件夹
        filepath = os.getcwd() + '/logs/log_' + datetime.datetime.now().strftime('%Y%m%d') + '.log'

    # 定义构造函数  when=D 天数，新生成的文件名上会带上时间 backCount 保留文件生成个数
    def __init__(self, filename=filepath, level='info', when='D', back_count=3):
        # when：描述滚动周期的基本单位;“S”: Seconds ,“M”: Minutes ,“H”: Hours ,“D”: Days ,“W”: Week day (0=Monday) ,
        # “midnight”: Roll over at midnight
        self.logger = logging.getLogger(filename)  # 指定日志文件名
        sh_format = colorlog.ColoredFormatter(
            fmt='%(log_color)s%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            log_colors=self.log_colors_config)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(sh_format)  # 设置屏幕上显示的格式
        th = logging.handlers.TimedRotatingFileHandler(filename=filename, when=when,
                                                       backupCount=back_count, encoding='utf-8')  # 写入文件
        th_format = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        th.setFormatter(th_format)  # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里 屏幕输出
        self.logger.addHandler(th)  # 把对象加到logger里 文件写入

    def get_log(self):
        return self.logger
