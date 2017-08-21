from setuptools import setup

setup(name='piper',
      version='0.1',
      description='Pipeline for machine learning',
      url='http://github.com/eivindbergem/piper',
      author='Eivind Alexander Bergem',
      author_email='eivind.bergem@gmail.com',
      license='GPL',
      packages=['piper'],
      test_suite='nose.collector',
      tests_require=['nose'])
