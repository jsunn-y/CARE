from setuptools import setup
import os
import re


def read_version():
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'CARE/__init__.py')
    with open(path, 'r') as fh:
        return re.search(r'__version__\s?=\s?[\'"](.+)[\'"]', fh.read()).group(1)


def read_author():
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'CARE/__init__.py')
    with open(path, 'r') as fh:
        return re.search(r'__author__\s?=\s?[\'"](.+)[\'"]', fh.read()).group(1)


def read_email():
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'CARE/__init__.py')
    with open(path, 'r') as fh:
        return re.search(r'__author_email__\s?=\s?[\'"](.+)[\'"]', fh.read()).group(1)


def read_git():
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'CARE/__init__.py')
    with open(path, 'r') as fh:
        return re.search(r'__url__\s?=\s?[\'"](.+)[\'"]', fh.read()).group(1)


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='CARE',
      version=read_version(),
      description='',
      long_description=readme(),
      long_description_content_type='text/markdown',
      author=read_author(),
      author_email=read_email(),
      url=read_git(),
      license='GPL3',
      project_urls={
          "Bug Tracker": read_git(),
          "Documentation": read_git(),
          "Source Code": read_git()},
      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
      ],
      keywords=['benchmark', 'protein', 'machine-learning'],
      packages=['CARE'],
      entry_points={
          'console_scripts': [
              'CARE = CARE.cmd:main'
          ]
      },
      install_requires=['jupyterlab',
                        'openai',
                        'matplotlib',
                        'numpy',
                        'pandas',
                        'scipy',
                        'sciutil',
                        'seaborn',
                        'scikit-learn',
                        'statsmodels',
                        'drfp',
                        'tqdm'],
      python_requires='>=3.8',
      data_files=[("", ["LICENSE"])]
      )
