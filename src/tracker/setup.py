from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ajaybhati',
    maintainer_email='ajaybhati@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'kill_pose = tracker.kill_pose:main',
            'goal_sender = tracker.goal_sender:main',
            'chaser_calculator = tracker.chase_calculator:main',
            'evasion_calculator = tracker.evasion_calculator:main',
            'state_machine = tracker.state_machine:main',
        ],
    },
)
