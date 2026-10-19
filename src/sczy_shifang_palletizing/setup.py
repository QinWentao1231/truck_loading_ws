from glob import glob
from setuptools import find_packages, setup

package_name = 'sczy_shifang_palletizing'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(include=[package_name, package_name + '.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md', 'requirements.txt']),
        ('share/' + package_name + '/config', glob('config/*.json')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    package_data={package_name + '.grpc_pkg.interface': ['proto/*.proto']},
    install_requires=['setuptools', 'numpy>=1.24,<2', 'protobuf>=4.24.2,<7',
                      'grpcio>=1.56.2,<2', 'plotly>=5.15,<7'],
    zip_safe=False,
    maintainer='nvidia',
    maintainer_email='nvidia@todo.todo',
    description="For use only at Sichuan China Tobacco's Shifang factory: legacy stacking and rule-based ROS2 paths.",
    license='TODO',
    extras_require={'test': ['pytest']},
    entry_points={'console_scripts': [
        'sczy_shifang_palletizing_node = sczy_shifang_palletizing.node:main',
    ]},
)
