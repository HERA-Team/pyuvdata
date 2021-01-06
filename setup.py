# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

import glob
import os
import io
import sys
import platform
from setuptools import setup, Extension, find_namespace_packages

from Cython.Distutils import build_ext
from distutils.sysconfig import get_config_var
from distutils.version import LooseVersion
from Cython.Build import cythonize

# When setting up, the binary extension modules haven't yet been built, so
# without a workaround we can't use the pyuvdata code to get the version.
os.environ["PYUVDATA_IGNORE_EXTMOD_IMPORT_FAIL"] = "1"

# add pyuvdata to our path in order to use the branch_scheme function
sys.path.append("pyuvdata")
from branch_scheme import branch_scheme  # noqa


# this solution works for `pip install .`` but not `python setup.py install`...
class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):

        # Import numpy here, only when headers are needed
        import numpy

        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())

        # Call original build_ext command
        build_ext.run(self)


with io.open("README.md", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()


def is_platform_mac():
    return sys.platform == "darwin"


def is_platform_windows():
    return sys.platform == "win32"


# For mac, ensure extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distuitls behaviour which is to target
# the version that python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.py
# implementation based on pandas, see https://github.com/pandas-dev/pandas/issues/23424
if is_platform_mac():
    if "MACOSX_DEPLOYMENT_TARGET" not in os.environ:
        current_system = LooseVersion(platform.mac_ver()[0])
        python_target = LooseVersion(get_config_var("MACOSX_DEPLOYMENT_TARGET"))
        if python_target < "10.9" and current_system >= "10.9":
            os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.9"

# define the cython compile args, depending on platform
if is_platform_windows():
    extra_compile_args = ["/Ox"]
else:
    extra_compile_args = ["-O3"]

global_c_macros = [
    ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
]

miriad_extension = Extension(
    "pyuvdata._miriad",
    sources=[
        "pyuvdata/uvdata/src/miriad_wrap.pyx",
        "pyuvdata/uvdata/src/uvio.c",
        "pyuvdata/uvdata/src/hio.c",
        "pyuvdata/uvdata/src/pack.c",
        "pyuvdata/uvdata/src/bug.c",
        "pyuvdata/uvdata/src/dio.c",
        "pyuvdata/uvdata/src/headio.c",
        "pyuvdata/uvdata/src/maskio.c",
    ],
    define_macros=global_c_macros,
    include_dirs=["pyuvdata/uvdata/src/"],
)

corr_fits_extension = Extension(
    "pyuvdata._corr_fits",
    sources=["pyuvdata/uvdata/corr_fits_src/corr_fits.pyx"],
    define_macros=global_c_macros,
    include_dirs=["pyuvdata/uvdata/corr_fits_src/"],
    extra_compile_args=extra_compile_args,
)

utils_extension = Extension(
    "pyuvdata._utils",
    sources=["pyuvdata/utils.pyx"],
    define_macros=global_c_macros,
    extra_compile_args=extra_compile_args,
)

extensions = [corr_fits_extension, utils_extension]

# don't build miriad on windows
if not is_platform_windows():
    extensions.append(miriad_extension)

novas_reqs = ["novas", "novas_de405"]
casa_reqs = ["python-casacore"]
healpix_reqs = ["astropy_healpix"]
cst_reqs = ["pyyaml"]
test_reqs = (
    casa_reqs
    + healpix_reqs
    + novas_reqs
    + cst_reqs
    + [
        "pytest>=6.2",
        "pytest-xdist",
        "pytest-cases>=3",
        "pytest-cov",
        "cython",
        "coverage",
        "pre-commit",
    ]
)
doc_reqs = ["sphinx", "pypandoc"]

setup_args = {
    "name": "pyuvdata",
    "author": "Radio Astronomy Software Group",
    "url": "https://github.com/RadioAstronomySoftwareGroup/pyuvdata",
    "license": "BSD",
    "description": "an interface for astronomical interferometeric datasets in python",
    "long_description": readme,
    "long_description_content_type": "text/markdown",
    "package_dir": {"pyuvdata": "pyuvdata"},
    "packages": find_namespace_packages(),
    "cmdclass": {"build_ext": CustomBuildExtCommand},
    "ext_modules": cythonize(extensions, language_level=3),
    "scripts": [fl for fl in glob.glob("scripts/*") if not os.path.isdir(fl)],
    "use_scm_version": {"local_scheme": branch_scheme},
    "include_package_data": True,
    "install_requires": [
        "numpy>=1.18",
        "scipy",
        "astropy>=4.2",
        "h5py",
        "setuptools_scm",
    ],
    "extras_require": {
        "casa": casa_reqs,
        "novas": novas_reqs,
        "healpix": healpix_reqs,
        "cst": cst_reqs,
        "all": casa_reqs + healpix_reqs + cst_reqs + novas_reqs,
        "test": test_reqs,
        "doc": doc_reqs,
        "dev": test_reqs + doc_reqs,
    },
    "classifiers": [
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    "keywords": "radio astronomy interferometry",
}

if __name__ == "__main__":
    setup(**setup_args)
