#!/usr/bin/env python

import os
import os.path as osp
import time
import subprocess
from numpy.distutils.core import setup

# Create list of all sub-directories with
#   __init__.py files...
packages = []
for subdir, dirs, files in os.walk('nanocavity'):
    if '__init__.py' in files:
        packages.append(subdir.replace(os.sep, '.'))

# Generate configuration
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('nanocavity')
    return config

# Main setup of python modules
setup(name='nanocavity',
      description='Python package for light emission and quantum transport in plasmonic nanocavities',
      author='Andres Bejarano',
      author_email='andres.bejarano@dipc.org',
      license='GPL',
      version='0.1.0',
      packages=packages,
      configuration=configuration)
