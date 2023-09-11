from setuptools import setup, find_packages
from setuptools.command.install import install

# TODO fix this so we actually have a version.py
__version__ = "0.1.0-0"  # This will get replaced when reading version.py


with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()

setup(
    name='rlgym-distrib-rl-wrapper',
    packages=find_packages(),
    version=__version__,
    description='A wrapper for the rlgym RL environment',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/nexus-rl/rlgym-distrib-rl-wrapper',
    install_requires=[
        'gym',
        'rlgym-sim',
        'rlgym-tools==1.7.0',
    ],
    python_requires='>=3.7',
    license='Apache 2.0',
    license_file='LICENSE',
    keywords=['rocket-league', 'reinforcement-learning', 'reinforcement-learning-algorithms', 'gym', 'machine-learning'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        "Operating System :: Microsoft :: Windows",
    ],
    package_data={}
)
