from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages

import numpy

setup(name='igmtools',
      version='0.1.0',
      packages=find_packages(),
      description='Tools for research on the intergalactic medium',
      install_requires=['astropy>=1.0', 'scipy', 'astroML', 'matplotlib'],
      author='Charles Finn',
      author_email='c.w.finn2301@gmail.com',
      license='BSD',
      url='https://github.com/cwfinn/igmtools',
      download_url='https://pypi.python.org/pypi/igmtools/0.1.0',
      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Topic :: Scientific/Engineering :: Physics'
      ],
      provides=['igmtools'],
      zip_safe=False,
      use_2to3=False,
      entry_points={
          'console_scripts': [
              'velplot = igmtools.plot.velplot_utils:main',
              'ivelplot = igmtools.modeling.helpers:main',
              'runvpfit = igmtools.modeling.vpfit:main'
          ]
      },
      include_dirs=[numpy.get_include(), ],
      include_package_data=True)
