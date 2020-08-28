# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from os import path
here = path.abspath(path.dirname(__file__))

# Get version from the VERSION file
with open(path.join(here, 'VERSION'), encoding='utf-8') as f:
    version = f.read().strip()


setup(
    name='text_category',
    author="feiyang",
    author_email='duanqy1995@gmail.com',
    description="文本分类",
    version=version,
    url='https://github.com/kunpeng199494/text_category',
    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # These classifiers are *not* checked by 'pip install'. See instead
        # 'python_requires' below.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    keywords='text_category',
    packages=find_packages(
        exclude=[
            "data",
            "conf",
            "examples",
            "scripts",
        ]
    ),
    python_requires='>=3.6, <4',
    install_requires=[
        "hao >= 0.2",
        "sklearn",
        "torch >= 1.3.0",
        "numpy",
        "transformers >= 2.11.0"

    ],
    extras_require={  # Optional
        'dev': [],
        'test': ['pytest'],
    },

    include_package_data=True,
    project_urls={  # Optional
        'Source': 'https://github.com/kunpeng199494/text_category',
    },

)
