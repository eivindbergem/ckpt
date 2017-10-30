from setuptools import setup

setup(name='ckpt',
      version='0.1',
      description='Experimental scaffolding for machine learning',
      url='http://github.com/eivindbergem/ckpt',
      author='Eivind Alexander Bergem',
      author_email='eivind.bergem@gmail.com',
      license='GPL',
      packages=['ckpt'],
      scripts=['bin/ckpt'],
      test_suite='nose.collector',
      tests_require=['nose'])
