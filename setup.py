from setuptools import setup, Extension

setup(name='mykmeanssp',
      version='1.0',
      description='capi for kmeans cluster algorithm by Shir and Tom',
      ext_modules=[Extension('mykmeanssp', sources=['kmeans.c'])])