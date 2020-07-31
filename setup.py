try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import site
import sys

site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

setup(
    name="temporal_nmf",
    version="0.2",
    description="Embedding of temporal networks using approximate parametric NMF",
    entry_points={"console_scripts": ["tnmf_train = temporal_nmf.scripts.train:train",]},
    author="Benjamin Wild",
    author_email="b.w@fu-berlin.de",
    url="https://github.com/nebw/temporal_nmf/",
    packages=["temporal_nmf", "temporal_nmf.scripts"],
)
