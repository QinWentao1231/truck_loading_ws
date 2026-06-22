from setuptools import find_packages, setup

package_name = 'cabin_width_check'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['config.json']),
        ('share/' + package_name + '/rviz', ['rviz/cabin_width.rviz']),
        ('share/' + package_name + '/launch', ['launch/cabin_width.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='qinwentao',
    maintainer_email='qinwentao@todo.todo',
    description='车厢内部点云宽度变形检测：分站采集拼接、逐片测宽、超限告警、RViz 标注',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'width_check_node = cabin_width_check.width_check_node:main',
        ],
    },
)
