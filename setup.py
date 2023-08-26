# -*- coding = utf-8 -*-
# @Time : 2023/8/26 16:48
# @Author : Xinye Yang
# @File : setup.py
# @Software : PyCharm
from setuptools import setup


setup(
    name='crowdnav',
    version='0.0.1',
    packages=[
        'crowd_nav',
        'crowd_nav.configs',
        'crowd_nav.policy',
        'crowd_nav.utils',
        'crowd_sim',
        'crowd_sim.envs',
        'crowd_sim.envs.policy',
        'crowd_sim.envs.utils',
    ],
    install_requires=[
        'gitpython',
        'gym',
        'matplotlib',
        'numpy',
        'scipy',
        'torch',
        'torchvision',
        'seaborn',
        'tqdm',
        'tensorboardX'
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
)