from setuptools import setup, Extension
import os
import sys
import numpy
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext


__version__ = "0.0.1"


# Mostly referred from Microsoft DiskANNpy. //Licensed under the MIT license.


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {'unix': ['-Ofast', '-m64', '-Wl,--no-as-needed']}
    arch_list = '-march -msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -msse4.2 -mavx -mavx2 -mavx512f'.split()
    no_arch_flag = True

    if 'CFLAGS' in os.environ:
        for flag in arch_list:
            if flag in os.environ["CFLAGS"]:
                no_arch_flag = False
                break

    if no_arch_flag:
        c_opts['unix'].append('-march=native')

    link_opts = {'unix': ['-L/opt/intel/compilers_and_libraries/linux/lib/intel64/', '-liomp5', '-lpthread', '-lm', '-ldl']}
    c_opts['unix'].append('-fopenmp')
    link_opts['unix'].extend(['-fopenmp', '-pthread'])
    #, '-ltcmalloc'])

    def build_extensions(self):
        ct = 'unix'
        opts = self.c_opts.get(ct, [])
        opts.append('-DVERSION_INFO="%s"' %
                    self.distribution.get_version())
        opts.append('-std=c++14')
        opts.append('-fvisibility=hidden')
        print('Extra compilation arguments:', opts)

        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(self.link_opts.get(ct, []))
            ext.include_dirs.extend([
                # Path to pybind11 headers
                pybind11.get_include(False),
                pybind11.get_include(True),
                # Path to numpy headers
                numpy.get_include()
            ])

        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        'bbannpy',
        ['ann_bind.cpp'],
        include_dirs=['../include/',
                      '/usr/include',
                      pybind11.get_include(False),
                      pybind11.get_include(True)],
        libraries=['aio'],
        language='c++',
        extra_objects=[
        '../build/src/lib/libBBAnnLib2_s.a',
        '../build/src/libTimeRecorder.a'],
    )
]

setup(
    name="bbannpy",
    version=__version__,
    author="TBD",
    author_email="TBD",
    url="TBD",
    description="TBD",
    long_description="TBD",
    ext_modules=ext_modules,
    install_requires=['numpy', 'pybind11'],
    cmdclass={"build_ext": BuildExt},
    test_suite="tests",
    zip_safe=False,
)

