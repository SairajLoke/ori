import os
import re
import ast
from pathlib import Path
from distutils.core import setup
from setuptools import find_packages


PACKAGE_NAME = "ha_data"
this_dir = os.path.dirname(os.path.abspath(__file__))

def get_package_version():
    with open(Path(this_dir) / PACKAGE_NAME / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("MAMBA_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


setup(
    name='ha_data',
    version=get_package_version(),
    packages=find_packages(),
    author="",
    license='',
    long_description=open('README.md').read(),
)