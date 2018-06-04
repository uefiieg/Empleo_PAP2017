# -*- coding: utf-8 -*-

from setuptools import setup


with open('README_FOR_EMPLOYMENT.md') as f:
    readme = f.read()

setup(
    name='employment_model',
    version='1.0.0',
    description='Jalisco employment model',
    long_description=readme,
    author='Ing. Raul Romero Barragan',
    author_email='raul7romero@gmail.com',
    license=license,
    install_requires=[
        'xgboost==0.6a2',
        'category-encoders==1.2.6',
        'scikit-learn==0.18.1',
        'multiprocess==0.70.5',
        'future==0.16.0',
        'numpy==1.12.0',
        'MySQL-python==1.2.5',
        'pandas==0.18.1',
        'scipy==0.18.1',
        'statsmodels==0.8.0',
        'PyPDF2==1.26.0',
        'matplotlib==2.0.0',
        'pyyaml==3.12',
        'pathos==0.2.0'
    ]
)
