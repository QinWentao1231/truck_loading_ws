from setuptools import find_packages, setup

package_name = 'pointcloud_process'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'rclpy', 'open3d'],
    zip_safe=True,
    maintainer='qinwentao',
    maintainer_email='qinwentao@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stacking_detection_node = pointcloud_process.stacking_detection_node:main'
        ],
    },
    # 在这里添加虚拟环境的路径
    python_requires='>=3.12',
    # setup_requires=['colcon-common-extensions'],
    # 添加你的虚拟环境路径
    dependency_links=[
        '/home/qinwentao/workcells/venv/lib/python3.12/site-packages',
    ]
)
