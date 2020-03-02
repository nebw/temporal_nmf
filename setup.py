#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

install_reqs = parse_requirements('requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_reqs]
dep_links = [str(req_line.url) for req_line in install_reqs]


setup(
    name='temporal_nmf',
    version='0.1',
    description='Embedding of temporal networks using approximate parametric NMF',
    author='Benjamin Wild',
    author_email='b.w@fu-berlin.de',
    url='https://github.com/nebw/temporal_nmf/',
    install_requires=reqs,
    dependency_links=dep_links,
    packages=['temporal_nmf'],
)
