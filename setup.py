from os.path import dirname, realpath
from setuptools import setup, find_packages, Distribution
from tools.version import __version__


def _read_requirements_file():
    """Return the elements in requirements.txt."""
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


setup(
    name='tools',
    version=__version__,
    packages=find_packages(),
    url='https://github.com/zhz03/dataset_tool',
    license='MIT',
    author='Zhaoliang Zheng',
    author_email='zhz03@g.ucla.edu',
    description='Dataset tool for multiple datasets like CARLA, V2X-Real, etc.',
    long_description=open("README.md").read(),
    install_requires=_read_requirements_file(),
)
