#!/usr/bin/env python
#   -*- coding: utf-8 -*-

from setuptools import setup
from setuptools.command.install import install as _install

class install(_install):
    def pre_install_script(self):
        pass

    def post_install_script(self):
        pass

    def run(self):
        self.pre_install_script()

        _install.run(self)

        self.post_install_script()

if __name__ == '__main__':
    setup(
        name = 'lava-dnf',
        version = '0.1.0',
        description = 'A library that provides processes and other software infrastructure to build architectures composed of Dynamic Neural Fields (DNF).',
        long_description = '',
        long_description_content_type = None,
        classifiers = [
            'Development Status :: 3 - Alpha',
            'Programming Language :: Python'
        ],
        keywords = '',

        author = '',
        author_email = '',
        maintainer = '',
        maintainer_email = '',

        license = "['BSD-3-Clause']",

        url = 'https://lava-nc.org',
        project_urls = {},

        scripts = [],
        packages = [],
        namespace_packages = [],
        py_modules = [
            'lava.lib.dnf.connect.connect',
            'lava.lib.dnf.connect.exceptions',
            'lava.lib.dnf.connect.reshape_bool.models',
            'lava.lib.dnf.connect.reshape_bool.process',
            'lava.lib.dnf.connect.reshape_int.models',
            'lava.lib.dnf.connect.reshape_int.process',
            'lava.lib.dnf.inputs.gauss_pattern.models',
            'lava.lib.dnf.inputs.gauss_pattern.process',
            'lava.lib.dnf.inputs.rate_code_spike_gen.models',
            'lava.lib.dnf.inputs.rate_code_spike_gen.process',
            'lava.lib.dnf.kernels.kernels',
            'lava.lib.dnf.operations.enums',
            'lava.lib.dnf.operations.exceptions',
            'lava.lib.dnf.operations.operations',
            'lava.lib.dnf.operations.shape_handlers',
            'lava.lib.dnf.utils.convenience',
            'lava.lib.dnf.utils.math',
            'lava.lib.dnf.utils.plotting',
            'lava.lib.dnf.utils.validation'
        ],
        entry_points = {},
        data_files = [],
        package_data = {},
        install_requires = [
            'lava-nc@https://github.com/lava-nc/lava/releases/download/v0.2.0/lava-nc-0.2.0.tar.gz',
            'numpy',
            'scipy>=1.7.2'
        ],
        dependency_links = [],
        zip_safe = True,
        cmdclass = {'install': install},
        python_requires = '',
        obsoletes = [],
    )
