from setuptools import setup, find_packages

setup(
    name = 'torus_world',
    version = '0.0.1',
    keywords = 'reinforcement learning',
    description = 'a library to assist adventures in a flat torus world',
    license = 'MIT License',
    url = 'https://github.com/mu-zhao',
    author = 'Mu Zhao',
    author_email = 'muzhao.pku@gmail.com',
    packages=find_packages(),
    platforms = 'any',
)