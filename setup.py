#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# for your packages to be recognized by python
d = generate_distutils_setup(
 packages=['tiago_speech_recognition', 'tiago_speech_recognition_ros'],
 package_dir={'tiago_speech_recognition_ros': 'src/tiago_speech_recognition_ros'}
)

setup(**d)
